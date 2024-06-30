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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	cgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/test/utils/ktesting"
	"sigs.k8s.io/yaml"
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

// NamedReadCloser is a [NamedReader] which also supports closing itself.
type NamedReadCloser interface {
	NamedReader
	Close() error
}

// NameReader associates a name with a reader which itself doesn't have one.
func NameReader(r io.Reader, name string) NamedReader {
	return namedReader{
		Reader: r,
		name:   name,
	}
}

type namedReader struct {
	io.Reader
	name string
}

func (n namedReader) Name() string {
	return n.name
}

// LoadFromManifests loads .yaml or .json manifest files and returns all items
// that it finds in them. It creates native types for everyting that the typed
// clients supported by ktesting support (like Pod, CustomResourceDefinition,
// etc.). Everything else gets decoded into an [unstructured.Unstructured].
//
// YAML files may contain multiple items separated by "---". Empty items (only
// comments) are ignored.
//
// Each item may get patches before parsing. The patch callback may be empty.
func LoadFromManifests(tCtx ktesting.TContext, patch func(ktesting.TContext, []byte) []byte, files ...NamedReader) []runtime.Object {
	tCtx.Helper()

	var objs []runtime.Object
	parse := func(tCtx ktesting.TContext, data []byte) {
		if patch != nil {
			data = patch(ktesting.WithStep(tCtx, "patch"), data)
		}
		obj, err := runtime.Decode(decoder, data)

		if runtime.IsNotRegisteredError(err) {
			var anyObj unstructured.Unstructured
			tCtx.ExpectNoError(runtime.DecodeInto(decoder, data, &anyObj), "decoding into unstructured.Unstructured failed")
			if len(anyObj.Object) > 0 {
				objs = append(objs, &anyObj)
				return
			}
			// Fall through...
		}

		if runtime.IsMissingVersion(err) || runtime.IsMissingKind(err) {
			// Could be an empty item (= only comments).
			// Parse as YAML and if that confirms that hypothesis, then ignore the item.
			var anyObj map[string]any
			tCtx.ExpectNoError(yaml.Unmarshal(data, &anyObj), "decoding as YAML")
			if len(anyObj) == 0 {
				return
			}
			// Fall through...
		}

		tCtx.ExpectNoError(err, "decoding failed")
		objs = append(objs, obj)
	}

	for _, file := range files {
		visitManifest(ktesting.WithStep(tCtx, file.Name()), parse, file)
	}

	return objs
}

func visitManifest(tCtx ktesting.TContext, cb func(ktesting.TContext, []byte), file NamedReader) {
	tCtx.Helper()
	data, err := io.ReadAll(file)
	tCtx.ExpectNoError(err, "reading manifest")

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
		itemCtx := ktesting.WithStep(tCtx, fmt.Sprintf("item #%d", i))
		cb(itemCtx, item)
	}
}
