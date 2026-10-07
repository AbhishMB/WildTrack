package org.wildtrack.bridge

import dji.sdk.keyvalue.value.common.LocationCoordinate2D
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.aircraft.simulator.InitializationSettings
import dji.v5.manager.aircraft.simulator.SimulatorManager

/**
 * DJI's own simulator, for build stage 2 (propellers off): velocity commands move the
 * simulated aircraft, which settles the roll/pitch-to-north/east mapping.
 */
object SimulatorControl {
    val enabled: Boolean get() = SimulatorManager.getInstance().isSimulatorEnabled

    fun start(lat: Double, lon: Double) {
        val settings = InitializationSettings.createInstance(LocationCoordinate2D(lat, lon), BridgeConfig.SIM_SATELLITES)
        SimulatorManager.getInstance().enableSimulator(settings, callback("simulator on at %.5f, %.5f".format(lat, lon), true))
    }

    fun stop() {
        SimulatorManager.getInstance().disableSimulator(callback("simulator off", false))
    }

    private fun callback(ok: String, on: Boolean) = object : CommonCallbacks.CompletionCallback {
        override fun onSuccess() {
            BridgeState.update { it.copy(simulator = on) }
            BridgeState.log(ok)
        }

        override fun onFailure(error: IDJIError) = BridgeState.log("simulator: ${error.description()}")
    }
}
