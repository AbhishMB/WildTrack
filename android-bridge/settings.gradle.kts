pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral() // DJI Mobile SDK v5 is published here (com.dji:dji-sdk-v5-*)
    }
}

rootProject.name = "wildtrack-bridge"
include(":app")
