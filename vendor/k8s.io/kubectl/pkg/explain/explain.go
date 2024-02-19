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

	"k8s.io/kube-openapi/pkg/util/proto"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type fieldsPrinter interface {
	PrintFields(proto.Schema) error
}

func splitDotNotation(model string) (string, []string) {
	var fieldsPath []string

	// ignore trailing period
	model = strings.TrimSuffix(model, ".")

	dotModel := strings.Split(model, ".")
	if len(dotModel) >= 1 {
		fieldsPath = dotModel[1:]
	}
	return dotModel[0], fieldsPath
}

// SplitAndParseResourceRequest separates the users input into a model and fields
func SplitAndParseResourceRequest(inResource string, mapper meta.RESTMapper) (schema.GroupVersionResource, []string, error) {
	inResource, fieldsPath := splitDotNotation(inResource)
	gvr, err := mapper.ResourceFor(schema.GroupVersionResource{Resource: inResource})
	if err != nil {
		return schema.GroupVersionResource{}, nil, err
	}

	return gvr, fieldsPath, nil
}

// SplitAndParseResourceRequestWithMatchingPrefix separates the users input into a model and fields
// while selecting gvr whose (resource, group) prefix matches the resource
func SplitAndParseResourceRequestWithMatchingPrefix(inResource string, mapper meta.RESTMapper) (gvr schema.GroupVersionResource, fieldsPath []string, err error) {
	// ignore trailing period
	inResource = strings.TrimSuffix(inResource, ".")
	dotParts := strings.Split(inResource, ".")

	gvrs, err := mapper.ResourcesFor(schema.GroupVersionResource{Resource: dotParts[0]})
	if err != nil {
		return schema.GroupVersionResource{}, nil, err
	}

	for _, gvrItem := range gvrs {
		// Find first gvr whose gr prefixes requested resource
		groupResource := gvrItem.GroupResource().String()
		if strings.HasPrefix(inResource, groupResource) {
			resourceSuffix := inResource[len(groupResource):]
			if len(resourceSuffix) > 0 {
				dotParts := strings.Split(resourceSuffix, ".")
				if len(dotParts) > 0 {
					fieldsPath = dotParts[1:]
				}
			}
			return gvrItem, fieldsPath, nil
		}
	}

	// If no match, take the first (the highest priority) gvr
	if len(gvrs) > 0 {
		gvr = gvrs[0]
		_, fieldsPath = splitDotNotation(inResource)
	}

	return gvr, fieldsPath, nil
}

// PrintModelDescription prints the description of a specific model or dot path.
// If recursive, all components nested within the fields of the schema will be
// printed.
func PrintModelDescription(fieldsPath []string, w io.Writer, schema proto.Schema, gvk schema.GroupVersionKind, recursive bool) error {
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
	return PrintModel(fieldName, f, b, schema, gvk)
}
