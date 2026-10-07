import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// DJI Mobile SDK v5. Check the aircraft is on the supported list (docs/SPEC.md) before flying.
val msdkVersion = "5.18.0"

// Your DJI developer app key (developer.dji.com > Apps), registered for package org.wildtrack.bridge.
// Put DJI_API_KEY=... in android-bridge/local.properties (not committed) or the environment.
val djiApiKey: String = run {
    val f = rootProject.file("local.properties")
    val props = Properties()
    if (f.exists()) f.inputStream().use { props.load(it) }
    props.getProperty("DJI_API_KEY") ?: System.getenv("DJI_API_KEY") ?: ""
}

android {
    namespace = "org.wildtrack.bridge"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.wildtrack.bridge"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        manifestPlaceholders["DJI_API_KEY"] = djiApiKey
        ndk {
            abiFilters += listOf("arm64-v8a") // MSDK v5 ships arm64 native libraries
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        buildConfig = true
    }

    packaging {
        jniLibs {
            // MSDK's native libraries must be extracted and must not be stripped.
            useLegacyPackaging = true
            keepDebugSymbols += "**/*.so"
            pickFirsts += listOf("lib/arm64-v8a/libc++_shared.so", "lib/armeabi-v7a/libc++_shared.so")
        }
        resources {
            excludes += listOf("META-INF/INDEX.LIST", "META-INF/io.netty.versions.properties", "META-INF/{AL2.0,LGPL2.1}")
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation("com.dji:dji-sdk-v5-aircraft:$msdkVersion")
    compileOnly("com.dji:dji-sdk-v5-aircraft-provided:$msdkVersion")
    runtimeOnly("com.dji:dji-sdk-v5-networkImp:$msdkVersion")

    implementation("io.ktor:ktor-server-core:2.3.12")
    implementation("io.ktor:ktor-server-cio:2.3.12")
    implementation("io.ktor:ktor-server-websockets:2.3.12")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("com.google.android.material:material:1.12.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303") // the real org.json, for JVM unit tests
}
