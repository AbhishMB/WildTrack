package org.wildtrack.bridge

import org.junit.Assert.assertEquals
import org.junit.Test

class AxisMappingTest {
    @Test
    fun defaultFollowsDjiDocs_rollIsNorth_pitchIsEast() {
        val s = AxisMapping().map(vn = 1.5, ve = -0.5)
        assertEquals(1.5, s.roll, 0.0)
        assertEquals(-0.5, s.pitch, 0.0)
    }

    @Test
    fun transposedSwapsAxes() {
        val s = AxisMapping(northAxis = AxisMapping.Axis.PITCH).map(vn = 1.5, ve = -0.5)
        assertEquals(-0.5, s.roll, 0.0)
        assertEquals(1.5, s.pitch, 0.0)
    }

    @Test
    fun signFlips() {
        val s = AxisMapping(northSign = -1, eastSign = -1).map(vn = 2.0, ve = 3.0)
        assertEquals(-2.0, s.roll, 0.0)
        assertEquals(-3.0, s.pitch, 0.0)
        val t = AxisMapping(northAxis = AxisMapping.Axis.PITCH, northSign = -1).map(vn = 2.0, ve = 3.0)
        assertEquals(3.0, t.roll, 0.0)
        assertEquals(-2.0, t.pitch, 0.0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsBadSign() {
        AxisMapping(northSign = 2)
    }

    @Test
    fun describe() {
        assertEquals("roll=+north pitch=+east", AxisMapping().describe())
        assertEquals("roll=-east pitch=+north", AxisMapping(AxisMapping.Axis.PITCH, 1, -1).describe())
    }
}
