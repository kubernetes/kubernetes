/*
Copyright 2024 The Kubernetes Authors.

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

package apply

import (
	"fmt"

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/features"
	"k8s.io/client-go/rest"
)

// NewRequest builds a new server-side apply request. The provided apply configuration object will
// be marshalled to the request's body using the default encoding, and the Content-Type header will
// be set to application/apply-patch with the appropriate structured syntax name suffix (today,
// either +yaml or +cbor, see
// https://www.iana.org/assignments/media-type-structured-suffix/media-type-structured-suffix.xhtml).
func NewRequest(client rest.Interface, applyConfiguration interface{}) (*rest.Request, error) {
	pt := types.ApplyYAMLPatchType
	marshal := json.Marshal

	if features.FeatureGates().Enabled(features.ClientsAllowCBOR) && features.FeatureGates().Enabled(features.ClientsPreferCBOR) {
		pt = types.ApplyCBORPatchType
		marshal = cbor.Marshal
	}

	body, err := marshal(applyConfiguration)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal apply configuration: %w", err)
	}

	return client.Patch(pt).Body(body), nil
}
