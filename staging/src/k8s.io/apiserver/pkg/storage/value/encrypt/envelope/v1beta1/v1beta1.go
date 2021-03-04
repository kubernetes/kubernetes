/*
Copyright 2019 The Kubernetes Authors.

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

// Package v1beta1 contains definition of kms-plugin's gRPC service.
package v1beta1

// IsVersionCheckMethod determines whether the supplied method is a version check against kms-plugin.
func IsVersionCheckMethod(method string) bool {
	return method == "/v1beta1.KeyManagementService/Version"
}
