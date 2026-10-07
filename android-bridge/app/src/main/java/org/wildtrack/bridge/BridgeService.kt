package org.wildtrack.bridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import dji.v5.common.error.IDJIError
import dji.v5.common.register.DJISDKInitEvent
import dji.v5.manager.SDKManager
import dji.v5.manager.interfaces.SDKManagerCallback
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground service: DJI SDK registration and aircraft connection, the laptop link,
 * 10 Hz telemetry, Virtual Stick, and the RTSP stream. Keeps running with the screen off.
 */
class BridgeService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var prefs: Prefs
    private lateinit var link: LinkServer
    private val telemetry = Telemetry()
    private lateinit var vs: VirtualStick
    private lateinit var live: LiveStream
    private var productUp = false
    private var lowBatterySent = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        prefs = Prefs(this)
        goForeground()
        BridgeState.update { it.copy(axis = prefs.axisMapping.describe()) }
        vs = VirtualStick(scope, { prefs.axisMapping }, ::sendEvent, ::alarm)
        live = LiveStream { prefs.fullHd }
        link = LinkServer(BridgeConfig.WS_PORT, vs::handle, ::onLaptop)
        link.start()
        BridgeState.log("listening on ws://${LiveStream.phoneIp() ?: "<phone-ip>"}:${BridgeConfig.WS_PORT}${BridgeConfig.WS_PATH}")
        scope.launch { telemetryLoop() }
        initSdk()
    }

    private fun initSdk() {
        SDKManager.getInstance().init(applicationContext, object : SDKManagerCallback {
            override fun onRegisterSuccess() {
                BridgeState.update { it.copy(registered = true, sdk = "registered") }
                BridgeState.log("DJI SDK registered")
            }

            override fun onRegisterFailure(error: IDJIError) {
                BridgeState.update { it.copy(sdk = "registration failed: ${error.description()}") }
                BridgeState.log("DJI SDK registration failed: ${error.description()} (check DJI_API_KEY and internet on first run)")
            }

            override fun onProductConnect(productId: Int) {
                BridgeState.update { it.copy(product = "connected ($productId)") }
                BridgeState.log("aircraft connected")
                onProductUp()
            }

            override fun onProductDisconnect(productId: Int) {
                BridgeState.update { it.copy(product = null) }
                BridgeState.log("aircraft disconnected")
            }

            override fun onProductChanged(productId: Int) = Unit

            override fun onInitProcess(event: DJISDKInitEvent, totalProcess: Int) {
                if (event == DJISDKInitEvent.INITIALIZE_COMPLETE) {
                    BridgeState.update { it.copy(sdk = "registering") }
                    SDKManager.getInstance().registerApp()
                }
            }

            override fun onDatabaseDownloadProgress(current: Long, total: Long) = Unit
        })
    }

    private fun onProductUp() {
        if (productUp) return
        productUp = true
        telemetry.start()
        vs.start()
        live.start()
    }

    private fun onLaptop(connected: Boolean, host: String?) {
        BridgeState.update { it.copy(laptop = if (connected) host else null) }
        BridgeState.log(if (connected) "laptop connected: $host" else "laptop disconnected")
        // No action needed on disconnect: the watchdog hovers after 300 ms and hands back after 3 s.
    }

    private suspend fun telemetryLoop() {
        var seq = 0L
        var next = System.nanoTime()
        while (scope.isActive) {
            val snap = telemetry.snapshot(vs.enabled)
            link.send(Protocol.telemetry(snap, ++seq, System.currentTimeMillis()))
            BridgeState.update { it.copy(telemetry = snap) }
            val batt = snap.batteryPct
            if (batt != null && batt < BridgeConfig.LOW_BATTERY_PCT && !lowBatterySent) {
                lowBatterySent = true
                sendEvent("low_battery", "$batt%")
            }
            next += BridgeConfig.TELEMETRY_PERIOD_MS * 1_000_000
            delay(((next - System.nanoTime()) / 1_000_000).coerceAtLeast(0))
        }
    }

    private fun sendEvent(name: String, reason: String) {
        link.send(Protocol.event(name, reason))
    }

    private fun alarm(text: String) {
        BridgeState.log("ALARM: $text")
        try {
            ToneGenerator(AudioManager.STREAM_ALARM, 100).startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 3000)
        } catch (_: RuntimeException) {
            // no audio focus; the vibration and the on-screen log still fire
        }
        val pattern = longArrayOf(0, 400, 200, 400, 200, 800)
        val vib = if (Build.VERSION.SDK_INT >= 31) {
            (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        vib.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }

    fun stopAuto() = vs.stopAuto()

    private fun goForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.notif_channel), NotificationManager.IMPORTANCE_LOW))
        val n: Notification = NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.notif_text))
            .setSmallIcon(R.drawable.ic_bridge)
            .setOngoing(true)
            .build()
        ServiceCompat.startForeground(this, 1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
    }

    override fun onDestroy() {
        vs.stop()
        live.stop()
        telemetry.stop()
        link.stop()
        scope.cancel()
        instance = null
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL = "bridge"

        @Volatile var instance: BridgeService? = null
            private set

        fun start(ctx: Context) = ContextCompat.startForegroundService(ctx, Intent(ctx, BridgeService::class.java))
    }
}
