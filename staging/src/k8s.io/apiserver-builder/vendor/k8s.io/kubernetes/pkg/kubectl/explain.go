/*
Copyright 2014 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"
	"io"
	"strings"

	"github.com/emicklei/go-restful/swagger"

	"k8s.io/apimachinery/pkg/api/meta"
	apiutil "k8s.io/kubernetes/pkg/api/util"
)

var allModels = make(map[string]*swagger.NamedModel)
var recursive = false // this is global for convenience, can become int for multiple levels

// SplitAndParseResourceRequest separates the users input into a model and fields
func SplitAndParseResourceRequest(inResource string, mapper meta.RESTMapper) (string, []string, error) {
	inResource, fieldsPath := splitDotNotation(inResource)
	inResource, _ = mapper.ResourceSingularizer(inResource)
	return inResource, fieldsPath, nil
}

// PrintModelDescription prints the description of a specific model or dot path
func PrintModelDescription(inModel string, fieldsPath []string, w io.Writer, swaggerSchema *swagger.ApiDeclaration, r bool) error {
	recursive = r // this is global for convenience
	apiVer := apiutil.GetVersion(swaggerSchema.ApiVersion) + "."

	var pointedModel *swagger.NamedModel
	for i := range swaggerSchema.Models.List {
		name := swaggerSchema.Models.List[i].Name

		allModels[name] = &swaggerSchema.Models.List[i]
		if strings.ToLower(name) == strings.ToLower(apiVer+inModel) {
			pointedModel = &swaggerSchema.Models.List[i]
		}
	}
	if pointedModel == nil {
		return fmt.Errorf("requested resource %q is not defined", inModel)
	}

	if len(fieldsPath) == 0 {
		return printTopLevelResourceInfo(w, pointedModel)
	}

	var pointedModelAsProp *swagger.NamedModelProperty
	for _, field := range fieldsPath {
		if prop, nextModel, isModel := getField(pointedModel, field); prop != nil {
			if isModel {
				pointedModelAsProp = prop
				pointedModel = allModels[nextModel]
			} else {
				return printPrimitive(w, prop)
			}
		} else {
			return fmt.Errorf("field %q does not exist", field)
		}
	}
	return printModelInfo(w, pointedModel, pointedModelAsProp)
}

func splitDotNotation(model string) (string, []string) {
	var fieldsPath []string
	dotModel := strings.Split(model, ".")
	if len(dotModel) >= 1 {
		fieldsPath = dotModel[1:]
	}
	return dotModel[0], fieldsPath
}

func getPointedModel(prop *swagger.ModelProperty) (string, bool) {
	if prop.Ref != nil {
		return *prop.Ref, true
	} else if *prop.Type == "array" && prop.Items.Ref != nil {
		return *prop.Items.Ref, true
	}
	return "", false
}

func getField(model *swagger.NamedModel, sField string) (*swagger.NamedModelProperty, string, bool) {
	for _, prop := range model.Model.Properties.List {
		if prop.Name == sField {
			pointedModel, isModel := getPointedModel(&prop.Property)
			return &prop, pointedModel, isModel
		}
	}
	return nil, "", false
}

func printModelInfo(w io.Writer, model *swagger.NamedModel, modelProp *swagger.NamedModelProperty) error {
	t, _ := getFieldType(&modelProp.Property)
	fmt.Fprintf(w, "RESOURCE: %s <%s>\n\n", modelProp.Name, t)
	fieldDesc, _ := wrapAndIndentText(modelProp.Property.Description, "    ", 80)
	fmt.Fprintf(w, "DESCRIPTION:\n%s\n\n%s\n", fieldDesc, indentText(model.Model.Description, "    "))
	return printFields(w, model)
}

func printPrimitive(w io.Writer, field *swagger.NamedModelProperty) error {
	t, _ := getFieldType(&field.Property)
	fmt.Fprintf(w, "FIELD: %s <%s>\n\n", field.Name, t)
	d, _ := wrapAndIndentText(field.Property.Description, "    ", 80)
	fmt.Fprintf(w, "DESCRIPTION:\n%s\n", d)
	return nil
}

func printTopLevelResourceInfo(w io.Writer, model *swagger.NamedModel) error {
	fmt.Fprintf(w, "DESCRIPTION:\n%s\n", model.Model.Description)
	return printFields(w, model)
}

func printFields(w io.Writer, model *swagger.NamedModel) error {
	fmt.Fprint(w, "\nFIELDS:\n")
	for _, field := range model.Model.Properties.List {
		fieldType, err := getFieldType(&field.Property)
		if err != nil {
			return err
		}

		if arrayContains(model.Model.Required, field.Name) {
			fmt.Fprintf(w, "   %s\t<%s> -required-\n", field.Name, fieldType)
		} else {
			fmt.Fprintf(w, "   %s\t<%s>\n", field.Name, fieldType)
		}

		if recursive {
			pointedModel, isModel := getPointedModel(&field.Property)
			if isModel {
				for _, nestedField := range allModels[pointedModel].Model.Properties.List {
					t, _ := getFieldType(&nestedField.Property)
					fmt.Fprintf(w, "       %s\t<%s>\n", nestedField.Name, t)
				}
			}
		} else {
			fieldDesc, _ := wrapAndIndentText(field.Property.Description, "    ", 80)
			fmt.Fprintf(w, "%s\n\n", fieldDesc)
		}
	}
	fmt.Fprint(w, "\n")
	return nil
}

func getFieldType(prop *swagger.ModelProperty) (string, error) {
	if prop.Type == nil {
		return "Object", nil
	} else if *prop.Type == "any" {
		// Swagger Spec doesn't return information for maps.
		return "map[string]string", nil
	} else if *prop.Type == "array" {
		if prop.Items == nil {
			return "", fmt.Errorf("error in swagger spec. Property: %v contains an array without type", prop)
		}
		if prop.Items.Ref != nil {
			fieldType := "[]Object"
			return fieldType, nil
		}
		fieldType := "[]" + *prop.Items.Type
		return fieldType, nil
	}
	return *prop.Type, nil
}

func wrapAndIndentText(desc, indent string, lim int) (string, error) {
	words := strings.Split(strings.Replace(strings.TrimSpace(desc), "\n", " ", -1), " ")
	n := len(words)

	for i := 0; i < n; i++ {
		if len(words[i]) > lim {
			if strings.Contains(words[i], "/") {
				s := breakURL(words[i])
				words = append(words[:i], append(s, words[i+1:]...)...)
				i = i + len(s) - 1
			} else {
				fmt.Println(len(words[i]))
				return "", fmt.Errorf("there are words longer that the break limit is")
			}
		}
	}

	var lines []string
	line := []string{indent}
	lineL := len(indent)
	for i := 0; i < len(words); i++ {
		w := words[i]

		if strings.HasSuffix(w, "/") && lineL+len(w)-1 < lim {
			prev := line[len(line)-1]
			if strings.HasSuffix(prev, "/") {
				if i+1 < len(words)-1 && !strings.HasSuffix(words[i+1], "/") {
					w = strings.TrimSuffix(w, "/")
				}

				line[len(line)-1] = prev + w
				lineL += len(w)
			} else {
				line = append(line, w)
				lineL += len(w) + 1
			}
		} else if lineL+len(w) < lim {
			line = append(line, w)
			lineL += len(w) + 1
		} else {
			lines = append(lines, strings.Join(line, " "))
			line = []string{indent, w}
			lineL = len(indent) + len(w)
		}
	}
	lines = append(lines, strings.Join(line, " "))

	return strings.Join(lines, "\n"), nil
}

func breakURL(url string) []string {
	var buf []string
	for _, part := range strings.Split(url, "/") {
		buf = append(buf, part+"/")
	}
	return buf
}

func indentText(text, indent string) string {
	lines := strings.Split(text, "\n")
	for i := range lines {
		lines[i] = indent + lines[i]
	}
	return strings.Join(lines, "\n")
}

func arrayContains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
