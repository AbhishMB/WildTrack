package org.wildtrack.bridge

import android.os.SystemClock
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.sdk.keyvalue.value.flightcontroller.FlightControlAuthorityChangeReason
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickState
import dji.v5.manager.aircraft.virtualstick.VirtualStickStateListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

/**
 * Virtual Stick in advanced mode: ground-frame velocity, angular-velocity yaw, velocity
 * vertical (docs/SPEC.md, "Virtual Stick settings"), the phone watchdog, and the gimbal.
 *
 * Never re-enables Virtual Stick on its own: only an explicit vs_enable from the laptop
 * turns it on. DJI ends it by itself on Pause, mode switch off P, RC loss, return-to-home,
 * low battery and restricted zones; every end is reported to the laptop as vs_exit.
 */
class VirtualStick(
    private val scope: CoroutineScope,
    private val mapping: () -> AxisMapping,
    private val onEvent: (name: String, reason: String) -> Unit,
    private val onAlarm: (String) -> Unit,
    private val clock: () -> Long = SystemClock::elapsedRealtime,
) {
    private val vs get() = VirtualStickManager.getInstance()
    private val watchdog = Watchdog(BridgeConfig.WATCHDOG_HOVER_MS, BridgeConfig.WATCHDOG_HANDBACK_MS)
    private val dispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher() // serialises all SDK calls
    private var lastReason = "UNKNOWN"
    private var lastHoverSendMs = 0L

    @Volatile var enabled = false
        private set

    val hovering: Boolean get() = watchdog.hovering

    private val listener = object : VirtualStickStateListener {
        override fun onVirtualStickStateUpdate(stickState: VirtualStickState) {
            val now = stickState.isVirtualStickEnable
            if (enabled && !now) {
                watchdog.disarm()
                onEvent("vs_exit", lastReason)
                BridgeState.log("Virtual Stick ended: $lastReason")
            }
            enabled = now
            BridgeState.update { it.copy(vsEnabled = now) }
        }

        override fun onChangeReasonUpdate(reason: FlightControlAuthorityChangeReason) {
            lastReason = reason.name
        }
    }

    fun start() {
        vs.setVirtualStickStateListener(listener)
        scope.launch(dispatcher) {
            while (isActive) {
                delay(BridgeConfig.WATCHDOG_TICK_MS)
                watchdogTick()
            }
        }
    }

    fun stop() {
        vs.removeVirtualStickStateListener(listener)
        if (enabled) disable("bridge stopping")
    }

    fun handle(cmd: Command) {
        scope.launch(dispatcher) {
            when (cmd) {
                is Command.Vel -> if (enabled) {
                    watchdog.feed(clock())
                    send(Limits.clamp(cmd))
                }
                is Command.Gimbal -> gimbal(cmd.pitchDeg)
                Command.VsEnable -> enable()
                Command.VsDisable -> if (enabled) disable("laptop vs_disable")
                is Command.Unknown -> BridgeState.log("ignored message type '${cmd.type}'")
            }
            BridgeState.update { it.copy(hovering = watchdog.hovering) }
        }
    }

    /** The phone's STOP AUTO button. */
    fun stopAuto() {
        scope.launch(dispatcher) {
            onEvent("vs_exit", "PHONE_STOP_BUTTON")
            disable("STOP AUTO pressed on the phone")
        }
    }

    private fun enable() {
        vs.enableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                vs.setVirtualStickAdvancedModeEnabled(true)
                watchdog.feed(clock())
                onEvent("vs_enabled", "")
                BridgeState.log("Virtual Stick on (advanced mode)")
            }

            override fun onFailure(error: IDJIError) {
                onEvent("vs_enable_failed", error.description() ?: error.errorCode() ?: "")
                BridgeState.log("Virtual Stick refused: ${error.description()}")
            }
        })
    }

    private fun disable(why: String) {
        watchdog.disarm()
        vs.disableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() = BridgeState.log("Virtual Stick off: $why")
            override fun onFailure(error: IDJIError) = BridgeState.log("Virtual Stick off failed: ${error.description()}")
        })
    }

    private fun send(c: Command.Vel) {
        val s = mapping().map(c.vn, c.ve)
        val p = VirtualStickFlightControlParam()
        p.setRoll(s.roll)
        p.setPitch(s.pitch)
        p.setYaw(c.yawRateDps)
        p.setVerticalThrottle(-c.vd) // wire format is NED (down positive); DJI's throttle is up positive
        p.setRollPitchControlMode(RollPitchControlMode.VELOCITY)
        p.setYawControlMode(YawControlMode.ANGULAR_VELOCITY)
        p.setVerticalControlMode(VerticalControlMode.VELOCITY)
        p.setRollPitchCoordinateSystem(FlightCoordinateSystem.GROUND)
        vs.sendVirtualStickAdvancedParam(p)
    }

    private fun watchdogTick() {
        if (!enabled) return
        val now = clock()
        when (watchdog.check(now)) {
            Watchdog.Action.HOVER -> {
                onEvent("watchdog_hover", "")
                BridgeState.log("watchdog: no laptop command for ${BridgeConfig.WATCHDOG_HOVER_MS} ms, hovering")
            }
            Watchdog.Action.HANDBACK -> {
                onEvent("watchdog_handback", "")
                disable("watchdog: laptop silent for ${BridgeConfig.WATCHDOG_HANDBACK_MS / 1000} s")
                onAlarm("Laptop silent: Virtual Stick OFF. Fly manually.")
            }
            Watchdog.Action.NONE -> Unit
        }
        if (watchdog.hovering && now - lastHoverSendMs >= BridgeConfig.HOVER_RESEND_MS) {
            lastHoverSendMs = now
            send(Command.Vel(0, 0.0, 0.0, 0.0, 0.0))
        }
        BridgeState.update { it.copy(hovering = watchdog.hovering) }
    }

    private fun gimbal(pitchDeg: Double) {
        val r = GimbalAngleRotation()
        r.setMode(GimbalAngleRotationMode.ABSOLUTE_ANGLE)
        r.setPitch(pitchDeg.coerceIn(-90.0, 0.0))
        r.setYawIgnored(true)
        r.setRollIgnored(true)
        r.setDuration(0.5)
        KeyManager.getInstance().performAction(
            KeyTools.createKey(GimbalKey.KeyRotateByAngle),
            r,
            object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) = Unit
                override fun onFailure(error: IDJIError) = BridgeState.log("gimbal: ${error.description()}")
            },
        )
    }
}
