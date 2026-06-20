# Aestra Android

Mobile-first Android WebView build of the Aestra facial aesthetics analyzer.

This project intentionally avoids npm/pnpm/Capacitor during CI. GitHub Actions builds the Android APK directly with Gradle.

## Build

1. Push the repo to GitHub.
2. Open **Actions**.
3. Run **Build Android APK**.
4. Download the uploaded APK artifact.

## Notes

- Main app UI is mobile-first and tab-based.
- In-app camera is supported through Android WebView camera permissions and `getUserMedia`.
- Gallery fallback still works.
- The analysis core still uses MediaPipe models loaded from the network on first use.
