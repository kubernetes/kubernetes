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

package v1

import (
	"log/slog"

	"github.com/go-logr/logr"
	"k8s.io/klog/v2"
)

// setSlogDefaultLogger sets the global slog default logger to the same default
// that klog currently uses.
func setSlogDefaultLogger() {
	// klog.Background() always returns a valid logr.Logger, regardless of
	// how logging was configured. We just need to turn it into a
	// slog.Handler. SetDefault then needs a slog.Logger.
	handler := logr.ToSlogHandler(klog.Background())
	slog.SetDefault(slog.New(handler))
}
