//go:build go1.21
// +build go1.21

/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// slog2k8s demonstrates how an application using log/slog for logging
// can include Kubernetes packages.
package main

import (
	"context"
	"log/slog"
	"os"

	"k8s.io/klog/v2"
)

func main() {
	options := slog.HandlerOptions{AddSource: true}
	textHandler := slog.NewTextHandler(os.Stderr, &options)
	textLogger := slog.New(textHandler)

	// Use text output as default logger.
	slog.SetDefault(textLogger)

	// This also needs to be done through klog to ensure that all code
	// using klog uses the text handler. klog.Background/TODO/FromContext
	// will return a thin wrapper around the textHandler, so all that klog
	// still does is manage the global default and retrieval from contexts.
	klog.SetSlogLogger(textLogger)

	textLogger.Info("slog.Logger.Info")
	klog.InfoS("klog.InfoS")
	klog.Background().Info("klog.Background+logr.Logger.Info")
	klog.FromContext(context.Background()).Info("klog.FromContext+logr.Logger.Info")
}
