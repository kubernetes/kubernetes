/*
Copyright 2025 The Kubernetes Authors.

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

// package kyaml provides a printer for Kubernetes objects that formats them
// as KYAML, a strict subset of YAML that is designed to be explicit and
// unambiguous.  KYAML is YAML.
package printers

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/yaml/kyaml"
)

// KYAMLPrinter is an implementation of ResourcePrinter which formats data into
// a specific dialect of YAML, known as KYAML. KYAML is halfway between YAML
// and JSON, but is a strict subset of YAML, and so it should should be
// readable by any YAML parser. It is designed to be explicit and unambiguous,
// and eschews significant whitespace.
type KYAMLPrinter struct {
	encoder kyaml.Encoder
}

// PrintObj prints the data as KYAML to the specified writer.
func (p *KYAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	// We use reflect.Indirect here in order to obtain the actual value from a pointer.
	// We need an actual value in order to retrieve the package path for an object.
	// Using reflect.Indirect indiscriminately is valid here, as all runtime.Objects are supposed to be pointers.
	if InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj)).Type().PkgPath()) {
		return errors.New(InternalObjectPrinterErr)
	}

	switch obj := obj.(type) {
	case *metav1.WatchEvent:
		if InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj.Object.Object)).Type().PkgPath()) {
			return errors.New(InternalObjectPrinterErr)
		}
	case *runtime.Unknown:
		return p.encoder.FromYAML(bytes.NewReader(obj.Raw), w)
	}

	if obj.GetObjectKind().GroupVersionKind().Empty() {
		return fmt.Errorf("missing apiVersion or kind; try GetObjectKind().SetGroupVersionKind() if you know the type")
	}

	return p.encoder.FromObject(obj, w)
}
