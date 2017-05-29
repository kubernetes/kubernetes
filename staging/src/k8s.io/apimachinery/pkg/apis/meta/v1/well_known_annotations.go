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

package v1

const (
	// When kubelet is started with the "external" cloud provider, then
	// it sets this annotation on the node to denote an ip address set from the
	// cmd line flag. This ip is verified with the cloudprovider as valid by
	// the cloud-controller-manager
	AnnotationProvidedIPAddr = "alpha.kubernetes.io/provided-node-ip"
)
