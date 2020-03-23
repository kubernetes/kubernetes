/*
Copyright 2017 The Kubernetes Authors.

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

package explain

import (
	"bytes"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestFields(t *testing.T) {
	schema := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "OneKind",
	})
	if schema == nil {
		t.Fatal("Couldn't find schema v1.OneKind")
	}

	want := `field1	<Object> -required-
  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla ut lacus ac
  enim vulputate imperdiet ac accumsan risus. Integer vel accumsan lectus.
  Praesent tempus nulla id tortor luctus, quis varius nulla laoreet. Ut orci
  nisi, suscipit id velit sed, blandit eleifend turpis. Curabitur tempus ante at
  lectus viverra, a mattis augue euismod. Morbi quam ligula, porttitor sit amet
  lacus non, interdum pulvinar tortor. Praesent accumsan risus et ipsum dictum,
  vel ullamcorper lorem egestas.

field2	<[]map[string]string>
  This is an array of object of PrimitiveDef

`

	buf := bytes.Buffer{}
	f := Formatter{
		Writer: &buf,
		Wrap:   80,
	}
	s, err := LookupSchemaForField(schema, []string{})
	if err != nil {
		t.Fatalf("Invalid path %v: %v", []string{}, err)
	}
	if err := (fieldsPrinterBuilder{Recursive: false}).BuildFieldsPrinter(&f).PrintFields(s); err != nil {
		t.Fatalf("Failed to print fields: %v", err)
	}
	got := buf.String()
	if got != want {
		t.Errorf("Got:\n%v\nWant:\n%v\n", buf.String(), want)
	}
}
