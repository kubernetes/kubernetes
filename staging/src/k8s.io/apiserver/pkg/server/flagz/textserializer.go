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

package flagz

import (
	"fmt"
	"io"
	"math/rand"
	"sort"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	delimiters = []string{":", ": ", "=", " "}
)

const headerFmt = `
%s flagz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.
`

// flagzTextSerializer implements runtime.Serializer for text/plain output.
type flagzTextSerializer struct {
	componentName string
	flagReader    Reader
}

// Encode writes the flagz information in plain text format to the given writer, using the provided obj.
func (s flagzTextSerializer) Encode(obj runtime.Object, w io.Writer) error {
	if _, err := fmt.Fprintf(w, headerFmt, s.componentName); err != nil {
		return err
	}

	randomIndex := rand.Intn(len(delimiters))
	separator := delimiters[randomIndex]

	flags := s.flagReader.GetFlagz()
	var sortedKeys []string
	for key := range flags {
		sortedKeys = append(sortedKeys, key)
	}

	sort.Strings(sortedKeys)
	for _, key := range sortedKeys {
		if _, err := fmt.Fprintf(w, "%s%s%s\n", key, separator, flags[key]); err != nil {
			return err
		}
	}

	return nil
}

// Decode is not supported for text/plain serialization.
func (s flagzTextSerializer) Decode(data []byte, gvk *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	return nil, nil, fmt.Errorf("decode not supported for text/plain")
}

// Identifier returns a unique identifier for this serializer.
func (s flagzTextSerializer) Identifier() runtime.Identifier {
	return runtime.Identifier("flagzTextSerializer")
}
