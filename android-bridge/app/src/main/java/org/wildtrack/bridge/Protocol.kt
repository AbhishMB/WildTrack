package org.wildtrack.bridge

import org.json.JSONObject
import kotlin.math.hypot

/**
 * Laptop <-> phone wire format (docs/SPEC.md, "Wire format"): one JSON object per
 * WebSocket text message. Pure: no Android or DJI types, so it is unit-tested on the JVM.
 */
sealed class Command {
    data class Vel(val seq: Long, val vn: Double, val ve: Double, val vd: Double, val yawRateDps: Double) : Command()
    data class Gimbal(val pitchDeg: Double) : Command()
    object VsEnable : Command()
    object VsDisable : Command()
    data class Unknown(val type: String) : Command()
}

/** Everything the 10 Hz telemetry message carries. null = not known yet. */
data class TelemetrySnapshot(
    val lat: Double? = null,
    val lon: Double? = null,
    val altRel: Double? = null,
    val pitch: Double? = null,
    val roll: Double? = null,
    val yaw: Double? = null,
    val vn: Double? = null,
    val ve: Double? = null,
    val vd: Double? = null,
    val gimbalPitch: Double? = null,
    val gimbalRoll: Double? = null,
    val gimbalYaw: Double? = null,
    val batteryPct: Int? = null,
    val gpsSats: Int? = null,
    val vsEnabled: Boolean = false,
    val flightMode: String = "UNKNOWN",
    val homeLat: Double? = null,
    val homeLon: Double? = null,
)

object Protocol {
    fun parse(text: String): Command {
        val o = JSONObject(text)
        return when (val type = o.optString("type")) {
            "vel" -> Command.Vel(
                o.optLong("seq"),
                o.getDouble("vn"),
                o.getDouble("ve"),
                o.optDouble("vd", 0.0),
                o.optDouble("yaw_rate_dps", 0.0),
            )
            "gimbal" -> Command.Gimbal(o.getDouble("pitch_deg"))
            "vs_enable" -> Command.VsEnable
            "vs_disable" -> Command.VsDisable
            else -> Command.Unknown(type)
        }
    }

    fun event(name: String, reason: String = ""): String =
        JSONObject().put("type", "event").put("name", name).put("reason", reason).toString()

    private fun num(v: Double?): Any = if (v == null || v.isNaN() || v.isInfinite()) JSONObject.NULL else v

    fun telemetry(t: TelemetrySnapshot, seq: Long, phoneMs: Long): String {
        val home = if (t.homeLat != null && t.homeLon != null && !(t.homeLat == 0.0 && t.homeLon == 0.0)) {
            JSONObject().put("lat", num(t.homeLat)).put("lon", num(t.homeLon))
        } else {
            JSONObject.NULL
        }
        return JSONObject()
            .put("type", "telemetry")
            .put("seq", seq)
            .put("t_phone_ms", phoneMs)
            .put("lat", num(t.lat))
            .put("lon", num(t.lon))
            .put("alt_rel_m", num(t.altRel))
            .put("att", JSONObject().put("pitch", num(t.pitch)).put("roll", num(t.roll)).put("yaw", num(t.yaw)))
            .put("vel_ned", JSONObject().put("n", num(t.vn)).put("e", num(t.ve)).put("d", num(t.vd)))
            .put("gimbal", JSONObject().put("pitch", num(t.gimbalPitch)).put("roll", num(t.gimbalRoll)).put("yaw", num(t.gimbalYaw)))
            .put("battery_pct", t.batteryPct ?: JSONObject.NULL)
            .put("gps_sats", t.gpsSats ?: 0)
            .put("vs_enabled", t.vsEnabled)
            .put("flight_mode", t.flightMode)
            .put("home", home)
            .toString()
    }
}

/** Clamp a laptop velocity command to the phone's own limits. NaN or infinite input means hover. */
object Limits {
    fun clamp(c: Command.Vel): Command.Vel {
        val bad = listOf(c.vn, c.ve, c.vd, c.yawRateDps).any { it.isNaN() || it.isInfinite() }
        if (bad) return c.copy(vn = 0.0, ve = 0.0, vd = 0.0, yawRateDps = 0.0)
        val h = hypot(c.vn, c.ve)
        val k = if (h > BridgeConfig.MAX_HORIZONTAL_MPS) BridgeConfig.MAX_HORIZONTAL_MPS / h else 1.0
        return c.copy(
            vn = c.vn * k,
            ve = c.ve * k,
            vd = c.vd.coerceIn(-BridgeConfig.MAX_VERTICAL_MPS, BridgeConfig.MAX_VERTICAL_MPS),
            yawRateDps = c.yawRateDps.coerceIn(-BridgeConfig.MAX_YAW_RATE_DPS, BridgeConfig.MAX_YAW_RATE_DPS),
        )
    }
}
