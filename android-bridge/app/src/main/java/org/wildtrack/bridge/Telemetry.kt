package org.wildtrack.bridge

import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.DJIKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.common.callback.CommonCallbacks
import dji.v5.manager.KeyManager

/**
 * Subscribes to the aircraft keys the laptop needs and keeps the latest values.
 *
 * KeyAircraftLocation3D altitude is relative to the takeoff point (confirm on the bench
 * that it reads ~0 on the ground). KeyGimbalAttitude is in north-east-down.
 * KeyAircraftVelocity is north/east/down in m/s.
 */
class Telemetry {
    @Volatile private var snap = TelemetrySnapshot()

    private fun <T> listen(key: DJIKey<T>, onValue: (T) -> Unit) {
        val km = KeyManager.getInstance()
        km.getValue(key)?.let(onValue) // current value, if the SDK has one cached
        km.listen(key, this, CommonCallbacks.KeyListener<T> { _, v -> if (v != null) onValue(v) })
    }

    private inline fun update(f: (TelemetrySnapshot) -> TelemetrySnapshot) {
        synchronized(this) { snap = f(snap) }
    }

    fun start() {
        listen(KeyTools.createKey(FlightControllerKey.KeyAircraftLocation3D)) { v ->
            update { it.copy(lat = v.latitude, lon = v.longitude, altRel = v.altitude) }
        }
        listen(KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude)) { v ->
            update { it.copy(pitch = v.pitch, roll = v.roll, yaw = v.yaw) }
        }
        listen(KeyTools.createKey(FlightControllerKey.KeyAircraftVelocity)) { v ->
            update { it.copy(vn = v.x, ve = v.y, vd = v.z) }
        }
        listen(KeyTools.createKey(FlightControllerKey.KeyGPSSatelliteCount)) { v ->
            update { it.copy(gpsSats = v) }
        }
        listen(KeyTools.createKey(FlightControllerKey.KeyHomeLocation)) { v ->
            update { it.copy(homeLat = v.latitude, homeLon = v.longitude) }
        }
        listen(KeyTools.createKey(FlightControllerKey.KeyRemoteControllerFlightMode)) { v ->
            update { it.copy(flightMode = v.name) }
        }
        listen(KeyTools.createKey(BatteryKey.KeyChargeRemainingInPercent)) { v ->
            update { it.copy(batteryPct = v) }
        }
        listen(KeyTools.createKey(GimbalKey.KeyGimbalAttitude, ComponentIndexType.LEFT_OR_MAIN)) { v ->
            update { it.copy(gimbalPitch = v.pitch, gimbalRoll = v.roll, gimbalYaw = v.yaw) }
        }
    }

    fun stop() {
        KeyManager.getInstance().cancelListen(this)
    }

    fun snapshot(vsEnabled: Boolean): TelemetrySnapshot = snap.copy(vsEnabled = vsEnabled)
}
