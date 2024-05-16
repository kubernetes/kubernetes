/*
Copyright 2019 The Kubernetes Authors.

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

package codec

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// NewLenientSchemeAndCodecs constructs a CodecFactory with strict decoding
// disabled, that has only the Schemes registered into it which are passed
// and added via AddToScheme functions. This can be used to skip strict decoding
// a specific version only.
func NewLenientSchemeAndCodecs(addToSchemeFns ...func(s *runtime.Scheme) error) (*runtime.Scheme, *serializer.CodecFactory, error) {
	lenientScheme := runtime.NewScheme()
	for _, s := range addToSchemeFns {
		if err := s(lenientScheme); err != nil {
			return nil, nil, fmt.Errorf("unable to add API to lenient scheme: %v", err)
		}
	}
	lenientCodecs := serializer.NewCodecFactory(lenientScheme, serializer.DisableStrict)
	return lenientScheme, &lenientCodecs, nil
}
