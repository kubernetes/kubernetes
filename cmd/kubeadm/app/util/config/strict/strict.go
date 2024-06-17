/*
Copyright 2018 The Kubernetes Authors.

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

package strict

import (
	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

// VerifyUnmarshalStrict takes a slice of schems, a JSON/YAML byte slice and a GroupVersionKind
// and verifies if the schema is known and if the byte slice unmarshals with strict mode.
func VerifyUnmarshalStrict(schemes []*runtime.Scheme, gvk schema.GroupVersionKind, bytes []byte) error {
	var scheme *runtime.Scheme
	for _, s := range schemes {
		if _, err := s.New(gvk); err == nil {
			scheme = s
			break
		}
	}
	if scheme == nil {
		return errors.Errorf("unknown configuration %#v", gvk)
	}

	opt := json.SerializerOptions{Yaml: true, Pretty: false, Strict: true}
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, opt)
	_, _, err := serializer.Decode(bytes, &gvk, nil)
	if err != nil {
		return errors.Wrapf(err, "error unmarshaling configuration %#v", gvk)
	}

	return nil
}
