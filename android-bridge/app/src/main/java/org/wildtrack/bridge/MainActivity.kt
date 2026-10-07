package org.wildtrack.bridge

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.switchmaterial.SwitchMaterial
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import kotlinx.coroutines.launch
import java.util.Locale

/** Video preview, status, and the big STOP AUTO button. */
class MainActivity : AppCompatActivity() {
    private lateinit var prefs: Prefs
    private var surface: android.view.Surface? = null

    private val permissions = buildList {
        add(Manifest.permission.ACCESS_FINE_LOCATION)
        add(Manifest.permission.ACCESS_COARSE_LOCATION)
        add(Manifest.permission.READ_PHONE_STATE)
        if (Build.VERSION.SDK_INT >= 31) {
            add(Manifest.permission.BLUETOOTH_CONNECT)
            add(Manifest.permission.BLUETOOTH_SCAN)
        }
        if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
    }

    private val askPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        BridgeService.start(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        prefs = Prefs(this)

        findViewById<Button>(R.id.stop_auto).setOnClickListener { BridgeService.instance?.stopAuto() }
        findViewById<Button>(R.id.sim_start).setOnClickListener {
            val t = BridgeState.status.value.telemetry
            val lat = t.lat?.takeIf { it != 0.0 } ?: prefs.simLat
            val lon = t.lon?.takeIf { it != 0.0 } ?: prefs.simLon
            SimulatorControl.start(lat, lon)
        }
        findViewById<Button>(R.id.sim_stop).setOnClickListener { SimulatorControl.stop() }
        bindAxisSwitches()
        findViewById<SwitchMaterial>(R.id.full_hd).apply {
            isChecked = prefs.fullHd
            setOnCheckedChangeListener { _, on -> prefs.fullHd = on }
        }
        findViewById<SurfaceView>(R.id.preview).holder.addCallback(previewCallback)

        val missing = permissions.filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) BridgeService.start(this) else askPermissions.launch(missing.toTypedArray())

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                BridgeState.status.collect(::render)
            }
        }
    }

    private fun bindAxisSwitches() {
        val transposed = findViewById<SwitchMaterial>(R.id.axis_transposed)
        val northFlip = findViewById<SwitchMaterial>(R.id.north_flip)
        val eastFlip = findViewById<SwitchMaterial>(R.id.east_flip)
        val m = prefs.axisMapping
        transposed.isChecked = m.northAxis == AxisMapping.Axis.PITCH
        northFlip.isChecked = m.northSign < 0
        eastFlip.isChecked = m.eastSign < 0
        val save = { _: android.widget.CompoundButton, _: Boolean ->
            val mapping = AxisMapping(
                northAxis = if (transposed.isChecked) AxisMapping.Axis.PITCH else AxisMapping.Axis.ROLL,
                northSign = if (northFlip.isChecked) -1 else 1,
                eastSign = if (eastFlip.isChecked) -1 else 1,
            )
            prefs.axisMapping = mapping
            BridgeState.update { it.copy(axis = mapping.describe()) }
            BridgeState.log("axis mapping: ${mapping.describe()}")
        }
        transposed.setOnCheckedChangeListener(save)
        northFlip.setOnCheckedChangeListener(save)
        eastFlip.setOnCheckedChangeListener(save)
    }

    private val previewCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) = Unit

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            surface = holder.surface
            MediaDataCenter.getInstance().cameraStreamManager.putCameraStreamSurface(
                ComponentIndexType.LEFT_OR_MAIN, holder.surface, width, height, ICameraStreamManager.ScaleType.CENTER_INSIDE,
            )
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            MediaDataCenter.getInstance().cameraStreamManager.removeCameraStreamSurface(holder.surface)
            surface = null
        }
    }

    private fun fmt(v: Double?, digits: Int = 1): String = v?.let { String.format(Locale.US, "%.${digits}f", it) } ?: "–"

    private fun render(s: Status) {
        val banner = findViewById<TextView>(R.id.vs_banner)
        when {
            s.vsEnabled && s.hovering -> banner.setBackgroundColor(0xFFB46900.toInt()).also { banner.text = "AUTO: HOVERING (laptop silent)" }
            s.vsEnabled -> banner.setBackgroundColor(0xFF127A3A.toInt()).also { banner.text = "AUTO: LAPTOP IS FLYING" }
            else -> banner.setBackgroundColor(0xFF5D6762.toInt()).also { banner.text = "PILOT FLYING" }
        }
        val t = s.telemetry
        findViewById<TextView>(R.id.status).text = buildString {
            appendLine("DJI SDK   ${s.sdk}")
            appendLine("aircraft  ${s.product ?: "not connected"}")
            appendLine("laptop    ${s.laptop ?: "not connected"}")
            appendLine("link      ws://${LiveStream.phoneIp() ?: "?"}:${BridgeConfig.WS_PORT}${BridgeConfig.WS_PATH}")
            appendLine("video     ${s.rtspUrl ?: "not streaming"}")
            if (s.streamInfo.isNotEmpty()) appendLine("          ${s.streamInfo}")
            appendLine("axis      ${s.axis}")
            appendLine("sim       ${if (s.simulator) "ON" else "off"}")
            appendLine("alt       ${fmt(t.altRel)} m   sats ${t.gpsSats ?: "–"}   batt ${t.batteryPct ?: "–"}%   mode ${t.flightMode}")
            appendLine("gimbal    ${fmt(t.gimbalPitch)}°   yaw ${fmt(t.yaw)}°")
        }
        findViewById<TextView>(R.id.log).text = s.log.joinToString("\n")
    }
}
