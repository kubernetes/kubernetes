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

// +k8s:deepcopy-gen=package
// +k8s:protobuf-gen=package
// +k8s:conversion-gen=k8s.io/kube-aggregator/pkg/apis/apiregistration
// +k8s:openapi-gen=true
// +groupName=apiregistration.k8s.io

// Package v1 contains the API Registration API, which is responsible for
// registering an API `Group`/`Version` with another kubernetes like API server.
// The `APIService` holds information about the other API server in
// `APIServiceSpec` type as well as general `TypeMeta` and `ObjectMeta`. The
// `APIServiceSpec` type have the main configuration needed to do the
// aggregation. Any request coming for specified `Group`/`Version` will be
// directed to the service defined by `ServiceReference` (on port 443) after
// validating the target using provided `CABundle` or skipping validation
// if development flag `InsecureSkipTLSVerify` is set. `Priority` is controlling
// the order of this API group in the overall discovery document.
// The return status is a set of conditions for this aggregation. Currently
// there is only one condition named "Available", if true, it means the
// api/server requests will be redirected to specified API server.
package v1 // import "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
