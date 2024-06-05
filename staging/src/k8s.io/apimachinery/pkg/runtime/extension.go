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
	"encoding/json"
	"errors"

	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
)

func (re *RawExtension) UnmarshalJSON(in []byte) error {
	if re == nil {
		return errors.New("runtime.RawExtension: UnmarshalJSON on nil pointer")
	}
	if !bytes.Equal(in, []byte("null")) {
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
	// TODO: Check whether ContentType is actually JSON before returning it.
	return re.Raw, nil
}

var selfDescribedTagPrefix = []byte{0xd9, 0xd9, 0xf7}

func (re RawExtension) MarshalCBOR() ([]byte, error) {
	if re.Raw == nil {
		if re.Object != nil {
			return cbor.Marshal(re.Object)
		}
		return []byte{0xf6}, nil
	}

	if bytes.HasPrefix(re.Raw, selfDescribedTagPrefix) {
		// The encoding of the head of a "self-described CBOR" tag is invalid as the prefix
		// of a Unicode text, and it prefixes all encoded objects produced by the
		// apimachinery CBOR serializer.
		return re.Raw, nil
	}

	// If the encoded data item is not enclosed in a self-described CBOR tag then assume the
	// bytes are JSON-encoded.
	var u interface{}
	if err := json.Unmarshal(re.Raw, &u); err != nil {
		return nil, err
	}
	return cbor.Marshal(u)
}

func (re *RawExtension) UnmarshalCBOR(in []byte) error {
	// For now, all inputs are transcoded to JSON to remain compatible with programs that assume
	// the Raw field contains JSON.
	var u interface{}
	if err := cbor.Unmarshal(in, &u); err != nil {
		return err
	}
	if u == nil {
		return nil
	}
	j, err := json.Marshal(u)
	if err != nil {
		return err
	}

	re.Raw = j
	return nil
}
