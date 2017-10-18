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
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/kube-openapi/pkg/util/proto"
)

type fieldsPrinter interface {
	PrintFields(proto.Schema) error
}

func splitDotNotation(model string) (string, []string) {
	var fieldsPath []string
	dotModel := strings.Split(model, ".")
	if len(dotModel) >= 1 {
		fieldsPath = dotModel[1:]
	}
	return dotModel[0], fieldsPath
}

// SplitAndParseResourceRequest separates the users input into a model and fields
func SplitAndParseResourceRequest(inResource string, mapper meta.RESTMapper) (string, []string, error) {
	inResource, fieldsPath := splitDotNotation(inResource)
	inResource, _ = mapper.ResourceSingularizer(inResource)
	return inResource, fieldsPath, nil
}

// PrintModelDescription prints the description of a specific model or dot path.
// If recursive, all components nested within the fields of the schema will be
// printed.
func PrintModelDescription(fieldsPath []string, w io.Writer, schema proto.Schema, recursive bool) error {
	fieldName := ""
	if len(fieldsPath) != 0 {
		fieldName = fieldsPath[len(fieldsPath)-1]
	}

	// Go down the fieldsPath to find what we're trying to explain
	schema, err := LookupSchemaForField(schema, fieldsPath)
	if err != nil {
		return err
	}
	b := fieldsPrinterBuilder{Recursive: recursive}
	f := &Formatter{Writer: w, Wrap: 80}
	return PrintModel(fieldName, f, b, schema)
}
