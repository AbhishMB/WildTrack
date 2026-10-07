package org.wildtrack.bridge

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.hypot

class ProtocolTest {
    @Test
    fun parsesTheSpecCommands() {
        assertEquals(
            Command.Vel(88, 1.4, -0.6, 0.0, 6.0),
            Protocol.parse("""{"type": "vel", "seq": 88, "vn": 1.4, "ve": -0.6, "vd": 0.0, "yaw_rate_dps": 6.0}"""),
        )
        assertEquals(Command.Gimbal(-60.0), Protocol.parse("""{"type": "gimbal", "pitch_deg": -60.0}"""))
        assertEquals(Command.VsEnable, Protocol.parse("""{"type": "vs_enable"}"""))
        assertEquals(Command.VsDisable, Protocol.parse("""{"type": "vs_disable"}"""))
        assertEquals(Command.Unknown("nope"), Protocol.parse("""{"type": "nope"}"""))
    }

    @Test
    fun telemetryMatchesTheSpecShape() {
        val t = TelemetrySnapshot(
            lat = 12.2958, lon = 76.6394, altRel = 61.2, pitch = -2.1, roll = 0.8, yaw = 134.5,
            vn = 1.2, ve = -0.4, vd = 0.0, gimbalPitch = -58.0, gimbalRoll = 0.0, gimbalYaw = 131.0,
            batteryPct = 72, gpsSats = 17, vsEnabled = true, flightMode = "P", homeLat = 12.2901, homeLon = 76.6402,
        )
        val o = JSONObject(Protocol.telemetry(t, 1042, 1730000000123))
        assertEquals("telemetry", o.getString("type"))
        assertEquals(1042, o.getLong("seq"))
        assertEquals(61.2, o.getDouble("alt_rel_m"), 0.0)
        assertEquals(134.5, o.getJSONObject("att").getDouble("yaw"), 0.0)
        assertEquals(-0.4, o.getJSONObject("vel_ned").getDouble("e"), 0.0)
        assertEquals(-58.0, o.getJSONObject("gimbal").getDouble("pitch"), 0.0)
        assertEquals(72, o.getInt("battery_pct"))
        assertTrue(o.getBoolean("vs_enabled"))
        assertEquals("P", o.getString("flight_mode"))
        assertEquals(12.2901, o.getJSONObject("home").getDouble("lat"), 0.0)
    }

    @Test
    fun unknownValuesAreJsonNull() {
        val o = JSONObject(Protocol.telemetry(TelemetrySnapshot(lat = Double.NaN), 1, 0))
        assertTrue(o.isNull("lat"))
        assertTrue(o.isNull("alt_rel_m"))
        assertTrue(o.isNull("home"))
        assertTrue(o.getJSONObject("gimbal").isNull("pitch"))
        assertEquals(0, o.getInt("gps_sats"))
    }

    @Test
    fun events() {
        val o = JSONObject(Protocol.event("vs_exit", "RC_PAUSE_STOP"))
        assertEquals("event", o.getString("type"))
        assertEquals("vs_exit", o.getString("name"))
        assertEquals("RC_PAUSE_STOP", o.getString("reason"))
    }

    @Test
    fun phoneClampsCommands() {
        val c = Limits.clamp(Command.Vel(1, 30.0, 40.0, -9.0, 400.0))
        assertEquals(BridgeConfig.MAX_HORIZONTAL_MPS, hypot(c.vn, c.ve), 1e-9)
        assertEquals(0.6, c.vn / hypot(c.vn, c.ve), 1e-9) // direction kept
        assertEquals(-BridgeConfig.MAX_VERTICAL_MPS, c.vd, 0.0)
        assertEquals(BridgeConfig.MAX_YAW_RATE_DPS, c.yawRateDps, 0.0)
        val n = Limits.clamp(Command.Vel(2, Double.NaN, 1.0, 0.0, 0.0))
        assertEquals(Command.Vel(2, 0.0, 0.0, 0.0, 0.0), n) // garbage in -> hover
        val ok = Command.Vel(3, 1.0, 1.0, 0.5, 10.0)
        assertEquals(ok, Limits.clamp(ok))
    }
}
