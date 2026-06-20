# Aesthetica Reborn Android

This repo is set up to build the Aesthetica Reborn Android APK through GitHub Actions **without npm, pnpm, Node installs, Android Studio, or local Android SDK downloads**.

## Build APK

1. Push this repo to GitHub.
2. Open GitHub → **Actions**.
3. Run **Build Android APK**.
4. Download the artifact named `aesthetica-reborn-debug-apk`.
5. Inside is `app-debug.apk`.

## Why this version is simpler

- The web app is already bundled into `android/app/src/main/assets/public/`.
- Capacitor Android is vendored into `android/capacitor-android/`.
- The workflow runs only Gradle in GitHub's cloud runner.
- No `npm install`, no `pnpm`, no Node version problems.

## Notes

- The APK is a debug APK for personal install/testing.
- The app still needs internet for MediaPipe model/CDN files on first analysis.
