package org.wildtrack.bridge

import android.content.Context

/** Phone-side constants. The laptop's config.yaml holds everything else. */
object BridgeConfig {
    const val WS_PORT = 8765
    const val WS_PATH = "/link"
    const val RTSP_PORT = 8554
    const val RTSP_USER = "aaa"
    const val RTSP_PASSWORD = "aaa"

    const val TELEMETRY_PERIOD_MS = 100L // 10 Hz

    // Watchdog (docs/SPEC.md, safety table): no laptop command for 300 ms -> hover; 3 s -> Virtual Stick off.
    const val WATCHDOG_HOVER_MS = 300L
    const val WATCHDOG_HANDBACK_MS = 3000L
    const val WATCHDOG_TICK_MS = 50L
    const val HOVER_RESEND_MS = 100L // keep feeding zero velocity at 10 Hz while hovering

    // Defence in depth: the phone clamps whatever the laptop sends.
    const val MAX_HORIZONTAL_MPS = 5.0
    const val MAX_VERTICAL_MPS = 2.0
    const val MAX_YAW_RATE_DPS = 30.0

    const val LOW_BATTERY_PCT = 35
    const val SIM_SATELLITES = 15
}

/** Settings the operator can change on the phone; persisted across restarts. */
class Prefs(context: Context) {
    private val p = context.getSharedPreferences("bridge", Context.MODE_PRIVATE)

    var axisMapping: AxisMapping
        get() = AxisMapping(
            northAxis = if (p.getBoolean("axis_transposed", false)) AxisMapping.Axis.PITCH else AxisMapping.Axis.ROLL,
            northSign = p.getInt("north_sign", 1),
            eastSign = p.getInt("east_sign", 1),
        )
        set(m) {
            p.edit()
                .putBoolean("axis_transposed", m.northAxis == AxisMapping.Axis.PITCH)
                .putInt("north_sign", m.northSign)
                .putInt("east_sign", m.eastSign)
                .apply()
        }

    var fullHd: Boolean
        get() = p.getBoolean("full_hd", false)
        set(v) = p.edit().putBoolean("full_hd", v).apply()

    var simLat: Double
        get() = p.getFloat("sim_lat", 12.2901f).toDouble()
        set(v) = p.edit().putFloat("sim_lat", v.toFloat()).apply()

    var simLon: Double
        get() = p.getFloat("sim_lon", 76.6402f).toDouble()
        set(v) = p.edit().putFloat("sim_lon", v.toFloat()).apply()
}
