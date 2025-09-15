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

package v1

const (
	// LabelServiceName is used to indicate the name of a Kubernetes service.
	LabelServiceName = "kubernetes.io/service-name"
	// LabelManagedBy is used to indicate the controller or entity that manages
	// an EndpointSlice. This label aims to enable different EndpointSlice
	// objects to be managed by different controllers or entities within the
	// same cluster. It is highly recommended to configure this label for all
	// EndpointSlices.
	LabelManagedBy = "endpointslice.kubernetes.io/managed-by"
	// LabelSkipMirror can be set to true on an Endpoints resource to indicate
	// that the EndpointSliceMirroring controller should not mirror this
	// resource with EndpointSlices.
	LabelSkipMirror = "endpointslice.kubernetes.io/skip-mirror"
)
