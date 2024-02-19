/*
Copyright 2022 The Kubernetes Authors.

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

package aggregated

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var AggregatedDiscoveryGV = schema.GroupVersion{Group: "apidiscovery.k8s.io", Version: "v2beta1"}

// Interface is from "k8s.io/apiserver/pkg/endpoints/handlers/negotiation"

// DiscoveryEndpointRestrictions allows requests to /apis to provide a Content Negotiation GVK for aggregated discovery.
var DiscoveryEndpointRestrictions = discoveryEndpointRestrictions{}

type discoveryEndpointRestrictions struct{}

func (discoveryEndpointRestrictions) AllowsMediaTypeTransform(mimeType string, mimeSubType string, gvk *schema.GroupVersionKind) bool {
	return IsAggregatedDiscoveryGVK(gvk)
}

func (discoveryEndpointRestrictions) AllowsServerVersion(string) bool  { return false }
func (discoveryEndpointRestrictions) AllowsStreamSchema(s string) bool { return s == "watch" }

// IsAggregatedDiscoveryGVK checks if a provided GVK is the GVK for serving aggregated discovery.
func IsAggregatedDiscoveryGVK(gvk *schema.GroupVersionKind) bool {
	if gvk != nil {
		return gvk.Group == "apidiscovery.k8s.io" && gvk.Version == "v2beta1" && gvk.Kind == "APIGroupDiscoveryList"
	}
	return false
}
