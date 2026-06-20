# Aesthetica Reborn Android WebView Build

This is the simplified Android package.

- No npm
- No pnpm
- No Node dependency install
- No Capacitor sync
- No gradlew requirement

The Android app is a native WebView wrapper loading the same Aesthetica Reborn web app from Android assets.

## Build on GitHub

1. Upload the **contents** of this folder to the repository root.
2. Go to Actions → Build Android APK → Run workflow.
3. Download the `aesthetica-reborn-debug-apk` artifact.
4. Install the APK on Android.

The first real analysis still needs internet because MediaPipe loads model/runtime files from web CDNs.
