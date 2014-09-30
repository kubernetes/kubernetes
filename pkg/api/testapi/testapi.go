/*
Copyright 2014 Google Inc. All rights reserved.

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

// Package testapi provides a helper for retrieving the KUBE_API_VERSION environment variable.
package testapi

import (
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Version returns the API version to test against as set by the KUBE_API_VERSION env var.
func Version() string {
	version := os.Getenv("KUBE_API_VERSION")
	if version == "" {
		version = latest.Version
	}
	return version
}

func CodecForVersionOrDie() runtime.Codec {
	interfaces, err := latest.InterfacesFor(Version())
	if err != nil {
		panic(err)
	}
	return interfaces.Codec
}
