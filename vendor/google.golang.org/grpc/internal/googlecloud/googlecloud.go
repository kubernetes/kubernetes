/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package googlecloud contains internal helpful functions for google cloud.
package googlecloud

import (
	"runtime"
	"strings"
	"sync"

	"google.golang.org/grpc/grpclog"
	internalgrpclog "google.golang.org/grpc/internal/grpclog"
)

const logPrefix = "[googlecloud]"

var (
	vmOnGCEOnce sync.Once
	vmOnGCE     bool

	logger = internalgrpclog.NewPrefixLogger(grpclog.Component("googlecloud"), logPrefix)
)

// OnGCE returns whether the client is running on GCE.
//
// It provides similar functionality as metadata.OnGCE from the cloud library
// package. We keep this to avoid depending on the cloud library module.
func OnGCE() bool {
	vmOnGCEOnce.Do(func() {
		mf, err := manufacturer()
		if err != nil {
			logger.Infof("failed to read manufacturer, setting onGCE=false: %v")
			return
		}
		vmOnGCE = isRunningOnGCE(mf, runtime.GOOS)
	})
	return vmOnGCE
}

// isRunningOnGCE checks whether the local system, without doing a network request, is
// running on GCP.
func isRunningOnGCE(manufacturer []byte, goos string) bool {
	name := string(manufacturer)
	switch goos {
	case "linux":
		name = strings.TrimSpace(name)
		return name == "Google" || name == "Google Compute Engine"
	case "windows":
		name = strings.ReplaceAll(name, " ", "")
		name = strings.ReplaceAll(name, "\n", "")
		name = strings.ReplaceAll(name, "\r", "")
		return name == "Google"
	default:
		return false
	}
}
