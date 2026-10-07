package org.wildtrack.bridge

import android.app.Application
import android.content.Context

class WildtrackApp : Application() {
    override fun attachBaseContext(base: Context?) {
        super.attachBaseContext(base)
        // Required by DJI Mobile SDK v5 before anything else touches the SDK.
        com.cySdkyc.clx.Helper.install(this)
    }
}
