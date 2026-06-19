# Aesthetica Reborn Android Project

This is a Capacitor Android wrapper around the mobile-ready Aesthetica Reborn web app.

## What is included
- `dist/` static web app files
- `android/` native Android project created with Capacitor
- App manifest, service worker, icons, and installable Android wrapper config

## Current status
- Android project generation: complete
- Capacitor sync: complete
- APK build: **not built in this environment** because the Android SDK / Gradle Android toolchain is not available here

## To build APK later
If opened in Android Studio or a full Android build environment, the project can be built into a debug/release APK.

Commands:
```bash
npm install
npx cap sync android
npx cap open android
```

Then build from Android Studio, or with Gradle in a proper Android SDK environment.
