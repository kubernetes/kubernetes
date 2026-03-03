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

package kubeletplugin

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
)

// DecodeMetadataFromStream reads through a JSON stream of metadata objects
// and populates dest with the first object that can be successfully decoded
// and converted. A metadata file may contain the same data encoded in
// multiple API versions (e.g., v1alpha2 followed by v1alpha1); the function
// tries each object in order and returns on the first success.
//
// dest must be a pointer to a concrete metadata type registered in the
// metadata scheme (e.g., *v1alpha1.DeviceMetadata) or the internal type
// (*metadata.DeviceMetadata). Decoding into the internal type is recommended
// because it is stable across API versions. The scheme's conversion functions
// handle conversion between any registered version.
func DecodeMetadataFromStream(decoder *json.Decoder, dest runtime.Object) error {
	gvks, _, err := metadataScheme.ObjectKinds(dest)
	if err != nil {
		return fmt.Errorf("determine target type: %w", err)
	}

	deserializer := metadataCodecFactory.UniversalDeserializer()

	for decoder.More() {
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return fmt.Errorf("read metadata object from stream: %w", err)
		}

		obj, _, err := deserializer.Decode(raw, nil, nil)
		if err != nil {
			continue
		}

		if err := metadataScheme.Convert(obj, dest, nil); err != nil {
			continue
		}

		// scheme.Convert does not propagate TypeMeta. Set it explicitly
		// for versioned types so that callers see the expected
		// apiVersion/kind. Internal types keep an empty TypeMeta.
		if gvks[0].Version != runtime.APIVersionInternal {
			dest.GetObjectKind().SetGroupVersionKind(gvks[0])
		}
		return nil
	}
	return fmt.Errorf("no compatible metadata version found in stream")
}
