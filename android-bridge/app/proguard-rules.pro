# Release builds are not minified (isMinifyEnabled = false). If you enable it, keep DJI's SDK:
-keep class dji.** { *; }
-keep class com.dji.** { *; }
-keep class com.cySdkyc.** { *; }
-dontwarn dji.**
-dontwarn com.dji.**
