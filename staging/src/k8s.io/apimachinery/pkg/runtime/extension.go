/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// RawExtension intentionally avoids implementing value.UnstructuredConverter for now because the
// signature of ToUnstructured does not allow returning an error value in cases where the conversion
// is not possible (content type is unrecognized or bytes don't match content type).
func (re RawExtension) toUnstructured(rawContentType string) (interface{}, error) {
	// This replicates the behavior of the original MarshalJSON implementation, which ignores
	// Object when Raw is non-nil (even if both are non-nil).
	if re.Raw != nil {
		switch rawContentType {
		default:
			return nil, fmt.Errorf("cannot convert RawExtension with unrecognized content type to unstructured")
		case ContentTypeJSON:
			var u interface{}
			if err := json.Unmarshal(re.Raw, &u); err != nil {
				return nil, fmt.Errorf("failed to parse RawExtension bytes as JSON: %w", err)
			}
			return u, nil
		case ContentTypeCBOR:
			var u interface{}
			if err := cbor.Unmarshal(re.Raw, &u); err != nil {
				return nil, fmt.Errorf("failed to parse RawExtension bytes as CBOR: %w", err)
			}
			return u, nil
		}
	}

	if re.Object != nil {
		rv := reflect.ValueOf(re.Object)
		u, err := value.TypeReflectEntryOf(rv.Type()).ToUnstructured(rv)
		if err != nil {
			return nil, err
		}
		return u, nil
	}

	return nil, nil
}

func (re RawExtension) guessContentType() string {
	// TODO: Allow for an explicit content type.

	c, unknown := cbor.Sniff(re.Raw)
	if !c || unknown {
		if ok, _ := json.Sniff(re.Raw); ok {
			return ContentTypeJSON
		}
	}
	if c {
		return ContentTypeCBOR
	}

	return ""
}

func (re *RawExtension) UnmarshalJSON(in []byte) error {
	if re == nil {
		return errors.New("runtime.RawExtension: UnmarshalJSON on nil pointer")
	}
	if !bytes.Equal(in, []byte("null")) {
		re.Raw = append(re.Raw[0:0], in...)
	}
	return nil
}

var cborNull = []byte{0xf6}

func (re *RawExtension) UnmarshalCBOR(in []byte) error {
	if re == nil {
		return errors.New("runtime.RawExtension: UnmarshalCBOR on nil pointer")
	}
	if !bytes.Equal(in, cborNull) {
		re.Raw = append(re.Raw[0:0], in...)
	}
	return nil
}

// MarshalJSON may get called on pointers or values, so implement MarshalJSON on value.
// http://stackoverflow.com/questions/21390979/custom-marshaljson-never-gets-called-in-go
func (re RawExtension) MarshalJSON() ([]byte, error) {
	if re.Raw == nil {
		// TODO: this is to support legacy behavior of JSONPrinter and YAMLPrinter, which
		// expect to call json.Marshal on arbitrary versioned objects (even those not in
		// the scheme). pkg/kubectl/resource#AsVersionedObjects and its interaction with
		// kubectl get on objects not in the scheme needs to be updated to ensure that the
		// objects that are not part of the scheme are correctly put into the right form.
		if re.Object != nil {
			return json.Marshal(re.Object)
		}
		return []byte("null"), nil
	}

	contentType := re.guessContentType()
	if contentType == ContentTypeJSON {
		return re.Raw, nil
	}

	u, err := re.toUnstructured(contentType)
	if err != nil {
		return nil, err
	}
	return json.Marshal(u)
}

func (re RawExtension) MarshalCBOR() ([]byte, error) {
	if re.Raw == nil {
		if re.Object != nil {
			return cbor.Marshal(re.Object)
		}
		return cbor.Marshal(nil)
	}

	contentType := re.guessContentType()
	if contentType == ContentTypeCBOR {
		return re.Raw, nil
	}

	u, err := re.toUnstructured(contentType)
	if err != nil {
		return nil, err
	}
	return cbor.Marshal(u)
}
