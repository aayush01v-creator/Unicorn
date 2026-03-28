import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

final InAppLocalhostServer localhostServer = InAppLocalhostServer(documentRoot: 'assets/www');

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // WebGPU isn't officially fully supported natively in all WebViews, 
  // but Android System WebView and iOS WKWebView both aggressively roll out WebGL/WebGPU.
  if (!kIsWeb && Platform.isAndroid) {
    await InAppWebViewController.setWebContentsDebuggingEnabled(true);
  }

  // Start the localhost server to seamlessly resolve Vite absolute '/assets/' URLs
  await localhostServer.start();

  runApp(const MaterialApp(
    home: UnicornApp(),
    debugShowCheckedModeBanner: false,
  ));
}

class UnicornApp extends StatefulWidget {
  const UnicornApp({super.key});

  @override
  State<UnicornApp> createState() => _UnicornAppState();
}

class _UnicornAppState extends State<UnicornApp> {
  final GlobalKey webViewKey = GlobalKey();
  InAppWebViewController? webViewController;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black, // Match React index.css dark mode
      body: SafeArea(
        child: InAppWebView(
          key: webViewKey,
          initialUrlRequest: URLRequest(
            url: WebUri("http://localhost:8080/index.html")
          ),
          initialSettings: InAppWebViewSettings(
            javaScriptEnabled: true,
            transparentBackground: true,
            hardwareAcceleration: true,
            allowFileAccessFromFileURLs: true,
            allowUniversalAccessFromFileURLs: true,
          ),
          onWebViewCreated: (controller) {
            webViewController = controller;
          },
          onConsoleMessage: (controller, consoleMessage) {
            if (kDebugMode) {
              print("[WebView Console] ${consoleMessage.message}");
            }
          },
        ),
      ),
    );
  }
}
