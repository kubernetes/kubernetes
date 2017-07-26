/*
Copyright 2017 The Kubernetes Authors.

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

package registry

import "k8s.io/apiserver/pkg/endpoints/request"

// FakeRequestInfo constructs a fake RequestInfo of Context
func FakeRequestInfo() *request.RequestInfo {
	return &request.RequestInfo{
		IsResourceRequest: false,
		Path:              "",
		Verb:              "GET",
		APIPrefix:         "",
		APIGroup:          "v1",
		APIVersion:        "v1",
		Namespace:         "test",
		Resource:          "test",
		Subresource:       "test",
		Name:              "",
		Parts:             nil,
	}
}
