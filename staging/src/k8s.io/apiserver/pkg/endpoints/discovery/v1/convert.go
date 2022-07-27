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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// The /discovery/v1 endpoint uses slightly different types compared to the
// original discovery types. These functions convert between them

func APIResourcesToDiscoveryAPIResources(rsrc []metav1.APIResource) []metav1.DiscoveryAPIResource {
	res := make([]metav1.DiscoveryAPIResource, len(rsrc))
	for i, v := range rsrc {
		res[i] = APIResourceToDiscoveryAPIResource(v)
	}
	return res
}

func APIResourceToDiscoveryAPIResource(rsrc metav1.APIResource) metav1.DiscoveryAPIResource {
	// Should use json tags to autogenerate this?
	return metav1.DiscoveryAPIResource{
		Name:         rsrc.Name,
		SingularName: rsrc.SingularName,
		Namespaced:   rsrc.Namespaced,
		Group:        rsrc.Group,
		Version:      rsrc.Version,
		Kind:         rsrc.Kind,
		Verbs:        rsrc.Verbs,
		ShortNames:   rsrc.ShortNames,
		Categories:   rsrc.Categories,
	}
}
