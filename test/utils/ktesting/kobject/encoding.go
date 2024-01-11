/*
Copyright 2024 The Kubernetes Authors.

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

package kobject

import (
	"bytes"
	"fmt"
	"io"
	"os"

	apiextensionsscheme "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	cgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/test/e2e/framework"
)

var scheme = runtime.NewScheme()
var decoder = serializer.NewCodecFactory(scheme).UniversalDeserializer() // no defaulting (not registered anyway), no conversion (= return exactly what is in the input)

func init() {
	metav1.AddToGroupVersion(scheme, schema.GroupVersion{Version: "v1"})
	utilruntime.Must(cgoscheme.AddToScheme(scheme))
	utilruntime.Must(apiextensionsscheme.AddToScheme(scheme))
}

// NamedReader is a reader which can also name what it is reading from
// for error messages. Implemented by os.File.
type NamedReader interface {
	io.Reader
	Name() string
}

var _ NamedReader = &os.File{}

// LoadFromManifests loads .yaml or .json manifest files and returns all items
// that it finds in them. It supports all types that the typed clients
// supported by ktesting support (like Pod, CustomResourceDefinition, etc.).
//
// YAML files may contain multiple items separated by "---".
func LoadFromManifests(files ...NamedReader) ([]interface{}, error) {
	var items []interface{}
	parse := func(data []byte) error {
		object, err := runtime.Decode(decoder, data)
		if err != nil {
			return err
		}
		items = append(items, object)
		return nil
	}

	err := visitManifests(parse, files...)

	return items, err
}

func visitManifests(cb func([]byte) error, files ...NamedReader) error {
	for _, file := range files {
		data, err := io.ReadAll(file)
		if err != nil {
			framework.Failf("reading manifest %q: %v", file.Name(), err)
		}

		// Split at the "---" separator before working on
		// individual item. Only works for .yaml.
		//
		// We need to split ourselves because we need access
		// to each original chunk of data for
		// runtime.Decode. kubectl has its own
		// infrastructure for this, but that is a lot of code
		// with many dependencies.
		items := bytes.Split(data, []byte("\n---"))

		for i, item := range items {
			if err := cb(item); err != nil {
				return fmt.Errorf("item #%d in %q: %w", i, file.Name(), err)
			}
		}
	}
	return nil
}
