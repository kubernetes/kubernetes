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

package runtime

import (
	"fmt"
	"reflect"

	"k8s.io/client-go/1.4/pkg/api/unversioned"
)

// CheckCodec makes sure that the codec can encode objects like internalType,
// decode all of the external types listed, and also decode them into the given
// object. (Will modify internalObject.) (Assumes JSON serialization.)
// TODO: verify that the correct external version is chosen on encode...
func CheckCodec(c Codec, internalType Object, externalTypes ...unversioned.GroupVersionKind) error {
	_, err := Encode(c, internalType)
	if err != nil {
		return fmt.Errorf("Internal type not encodable: %v", err)
	}
	for _, et := range externalTypes {
		exBytes := []byte(fmt.Sprintf(`{"kind":"%v","apiVersion":"%v"}`, et.Kind, et.GroupVersion().String()))
		obj, err := Decode(c, exBytes)
		if err != nil {
			return fmt.Errorf("external type %s not interpretable: %v", et, err)
		}
		if reflect.TypeOf(obj) != reflect.TypeOf(internalType) {
			return fmt.Errorf("decode of external type %s produced: %#v", et, obj)
		}
		err = DecodeInto(c, exBytes, internalType)
		if err != nil {
			return fmt.Errorf("external type %s not convertable to internal type: %v", et, err)
		}
	}
	return nil
}
