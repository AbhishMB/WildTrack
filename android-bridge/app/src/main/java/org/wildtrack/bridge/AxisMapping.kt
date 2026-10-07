package org.wildtrack.bridge

/**
 * Ground-frame velocity (north, east) -> Virtual Stick roll/pitch values.
 *
 * In DJI's GROUND coordinate system with VELOCITY control, roll and pitch carry the
 * north and east velocities; which carries which (and the signs) must be checked on the
 * bench in the DJI simulator (build stage 2): command north, watch the aircraft move north.
 * The default follows DJI's documentation (roll = north, pitch = east); one open-source
 * Mini 3 project found it transposed. This is the only place the mapping lives.
 */
data class AxisMapping(
    val northAxis: Axis = Axis.ROLL,
    val northSign: Int = 1,
    val eastSign: Int = 1,
) {
    enum class Axis { ROLL, PITCH }

    data class Sticks(val roll: Double, val pitch: Double)

    init {
        require(northSign == 1 || northSign == -1) { "northSign must be +1 or -1" }
        require(eastSign == 1 || eastSign == -1) { "eastSign must be +1 or -1" }
    }

    fun map(vn: Double, ve: Double): Sticks = when (northAxis) {
        Axis.ROLL -> Sticks(roll = northSign * vn, pitch = eastSign * ve)
        Axis.PITCH -> Sticks(roll = eastSign * ve, pitch = northSign * vn)
    }

    fun describe(): String {
        val n = if (northSign > 0) "+north" else "-north"
        val e = if (eastSign > 0) "+east" else "-east"
        return if (northAxis == Axis.ROLL) "roll=$n pitch=$e" else "roll=$e pitch=$n"
    }
}
