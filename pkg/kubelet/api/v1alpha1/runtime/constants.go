/*
Copyright 2016 The Kubernetes Authors.

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

package runtime

// This file contains all constants defined in CRI.

// Required runtime condition type.
const (
	// RuntimeReady means the runtime is up and ready to accept basic containers.
	RuntimeReady = "RuntimeReady"
	// NetworkReady means the runtime network is up and ready to accept containers which require network.
	NetworkReady = "NetworkReady"
)
