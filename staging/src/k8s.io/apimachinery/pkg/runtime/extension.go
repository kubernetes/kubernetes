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
	gojson "encoding/json"
	"errors"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// RawExtension intentionally avoids implementing value.UnstructuredConverter for now because the
// signature of ToUnstructured does not allow returning an error value in cases where the conversion
// is not possible (content type is unrecognized or bytes don't match content type).
func (re RawExtension) toUnstructured() (interface{}, error) {
	if re.Raw != nil {
		contentType, err := re.rawContentType()
		if err != nil {
			return nil, err
		}

		switch contentType {
		default:
			return nil, fmt.Errorf("cannot convert RawExtension with unrecognized content type %q to unstructured", contentType)
		case ContentTypeJSON:
			var u interface{}
			if err := json.Unmarshal(re.Raw, &u); err != nil {
				return nil, err
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

// rawContentType returns the value of the ContentType field, if not empty. Otherwise, it attempts
// to select an appropriate content type based on the Raw bytes. If the heuristic selection fails, a
// non-nil error is returned.
func (re RawExtension) rawContentType() (string, error) {
	if re.ContentType != "" {
		// Never resort to a heuristic in this case.
		return re.ContentType, nil
	}

	if gojson.Valid(re.Raw) {
		return ContentTypeJSON, nil
	}

	return "", errors.New("unable to infer content type of raw bytes")
}

func (re *RawExtension) UnmarshalJSON(in []byte) error {
	if re == nil {
		return errors.New("runtime.RawExtension: UnmarshalJSON on nil pointer")
	}
	if !bytes.Equal(in, []byte("null")) {
		re.Raw = append(re.Raw[0:0], in...)
		re.ContentType = ContentTypeJSON
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

	if contentType, _ := re.rawContentType(); contentType == ContentTypeJSON {
		return re.Raw, nil
	}

	u, err := re.toUnstructured()
	if err != nil {
		return nil, err
	}
	return json.Marshal(u)
}
