/*
Copyright The Kubernetes Authors.

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

package v1alpha1

const (
	// DRAResourceHealthService needs to be listed in the "supported versions"
	// array during plugin registration by a DRA plugin which provides
	// an implementation of the v1alpha1 DRAResourceHealth service.
	//
	// v1alpha1 is superseded by v1beta1. New drivers should implement v1beta1;
	// the kubelet helper serves v1alpha1 as well via a conversion wrapper so
	// that kubelets which only support v1alpha1 can still consume health
	// updates. See the v1beta1 package for the conversion code.
	DRAResourceHealthService = "v1alpha1.DRAResourceHealth"
)
