# Aestra Android Build

Aestra is a mobile-first Android WebView app for visual aesthetics analysis.

This package intentionally keeps the build chain simple:

- no npm install
- no pnpm
- no Capacitor sync
- no Gradle wrapper requirement
- GitHub Actions builds the APK directly with Gradle

## Build on GitHub

1. Upload the **contents** of this folder to the repository root.
2. Go to **Actions → Build Aestra APK → Run workflow**.
3. Download the `aestra-debug-apk` artifact.
4. Install the APK on Android.

The first analysis needs internet because MediaPipe loads model/runtime files from official CDN URLs.
