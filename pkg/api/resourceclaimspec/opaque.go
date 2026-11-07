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

package resourceclaimspec

import (
	"bytes"
	"encoding/json/jsontext"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// PreserveOpaqueParameters restores stored JSON bytes when only the representation
// changed, so immutable spec validation does not reject metadata-only updates.
func PreserveOpaqueParameters(new, old *resource.ResourceClaimSpec) {
	if len(new.Devices.Config) != len(old.Devices.Config) {
		return
	}
	for i := range new.Devices.Config {
		newConfig, oldConfig := &new.Devices.Config[i], &old.Devices.Config[i]
		if newConfig.Opaque == nil || oldConfig.Opaque == nil {
			continue
		}
		newOpaque, oldOpaque := *newConfig.Opaque, *oldConfig.Opaque
		newRaw, oldRaw := newOpaque.Parameters.Raw, oldOpaque.Parameters.Raw
		if bytes.Equal(newRaw, oldRaw) {
			continue
		}

		// Configurations are atomic and ordered. Only pair entries whose surrounding
		// fields (including any decoded Object) are unchanged.
		newOpaque.Parameters.Raw, oldOpaque.Parameters.Raw = nil, nil
		newIdentity, oldIdentity := *newConfig, *oldConfig
		newIdentity.Opaque, oldIdentity.Opaque = &newOpaque, &oldOpaque
		if !equality.Semantic.DeepEqual(newIdentity, oldIdentity) {
			continue
		}

		// Format private copies for comparison only. Reject duplicate names and invalid
		// UTF-8, and retain number spellings to avoid precision loss or broader equality.
		newJSON, oldJSON := jsontext.Value(newRaw).Clone(), jsontext.Value(oldRaw).Clone()
		if err := newJSON.Format(jsontext.ReorderRawObjects(true)); err != nil {
			continue
		}
		if err := oldJSON.Format(jsontext.ReorderRawObjects(true)); err != nil {
			continue
		}
		if bytes.Equal(newJSON, oldJSON) {
			newConfig.Opaque.Parameters.Raw = bytes.Clone(oldRaw)
		}
	}
}
