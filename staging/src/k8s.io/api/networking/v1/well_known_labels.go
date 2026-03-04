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

const (

	// TODO: Use IPFamily as field with a field selector,And the value is set based on
	// the name at create time and immutable.
	// LabelIPAddressFamily is used to indicate the IP family of a Kubernetes IPAddress.
	// This label simplify dual-stack client operations allowing to obtain the list of
	// IP addresses filtered by family.
	LabelIPAddressFamily = "ipaddress.kubernetes.io/ip-family"
	// LabelManagedBy is used to indicate the controller or entity that manages
	// an IPAddress. This label aims to enable different IPAddress
	// objects to be managed by different controllers or entities within the
	// same cluster. It is highly recommended to configure this label for all
	// IPAddress objects.
	LabelManagedBy = "ipaddress.kubernetes.io/managed-by"
)
