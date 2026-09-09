/*
Copyright 2018 The Kubernetes Authors.

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

package podresources

const (
	// Socket is the name of the podresources server socket
	Socket = "kubelet"

	// DefaultQPS is determined by empirically reviewing known consumers of the API.
	// It's at least unlikely that there is a legitimate need to query podresources
	// more than 100 times per second, the other subsystems are not guaranteed to react
	// so fast in the first place.
	DefaultQPS = 100

	// DefaultBurstTokens is determined by empirically reviewing known consumers of the API.
	// See the documentation of DefaultQPS, same caveats apply.
	DefaultBurstTokens = 10
)
