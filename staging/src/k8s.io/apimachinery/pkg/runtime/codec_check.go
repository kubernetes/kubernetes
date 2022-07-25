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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

// CheckCodec makes sure that the codec can encode objects like internalType,
// decode all of the external types listed, and also decode them into the given
// object. (Will modify internalObject.) (Assumes JSON serialization.)
// TODO: verify that the correct external version is chosen on encode...
func CheckCodec(c Codec, internalType Object, externalTypes ...schema.GroupVersionKind) error {
	if _, err := Encode(c, internalType); err != nil {
		return fmt.Errorf("Internal type not encodable: %v", err)
	}
	for _, et := range externalTypes {
		typeMeta := TypeMeta{
			Kind:       et.Kind,
			APIVersion: et.GroupVersion().String(),
		}
		exBytes, err := json.Marshal(&typeMeta)
		if err != nil {
			return err
		}
		obj, err := Decode(c, exBytes)
		if err != nil {
			return fmt.Errorf("external type %s not interpretable: %v", et, err)
		}
		if reflect.TypeOf(obj) != reflect.TypeOf(internalType) {
			return fmt.Errorf("decode of external type %s produced: %#v", et, obj)
		}
		if err = DecodeInto(c, exBytes, internalType); err != nil {
			return fmt.Errorf("external type %s not convertible to internal type: %v", et, err)
		}
	}
	return nil
}
