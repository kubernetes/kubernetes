// +build codegen

package api

import (
	"bytes"
	"fmt"
	"text/template"
)

// A ShapeValidationType is the type of validation that a shape needs
type ShapeValidationType int

const (
	// ShapeValidationRequired states the shape must be set
	ShapeValidationRequired = iota

	// ShapeValidationMinVal states the shape must have at least a number of
	// elements, or for numbers a minimum value
	ShapeValidationMinVal

	// ShapeValidationNested states the shape has nested values that need
	// to be validated
	ShapeValidationNested
)

// A ShapeValidation contains information about a shape and the type of validation
// that is needed
type ShapeValidation struct {
	// Name of the shape to be validated
	Name string
	// Reference to the shape within the context the shape is referenced
	Ref *ShapeRef
	// Type of validation needed
	Type ShapeValidationType
}

var validationGoCodeTmpls = template.Must(template.New("validationGoCodeTmpls").Parse(`
{{ define "requiredValue" -}}
    if s.{{ .Name }} == nil { 
		invalidParams.Add(request.NewErrParamRequired("{{ .Name }}"))
    }
{{- end }}
{{ define "minLen" -}}
	if s.{{ .Name }} != nil && len(s.{{ .Name }}) < {{ .Ref.Shape.Min }} {
		invalidParams.Add(request.NewErrParamMinLen("{{ .Name }}", {{ .Ref.Shape.Min }}))
	}
{{- end }}
{{ define "minLenString" -}}
	if s.{{ .Name }} != nil && len(*s.{{ .Name }}) < {{ .Ref.Shape.Min }} {
		invalidParams.Add(request.NewErrParamMinLen("{{ .Name }}", {{ .Ref.Shape.Min }}))
	}
{{- end }}
{{ define "minVal" -}}
	if s.{{ .Name }} != nil && *s.{{ .Name }} < {{ .Ref.Shape.Min }} {
		invalidParams.Add(request.NewErrParamMinValue("{{ .Name }}", {{ .Ref.Shape.Min }}))
	}
{{- end }}
{{ define "nestedMapList" -}}
    if s.{{ .Name }} != nil { 
		for i, v := range s.{{ .Name }} {
			if v == nil { continue }
			if err := v.Validate(); err != nil {
				invalidParams.AddNested(fmt.Sprintf("%s[%v]", "{{ .Name }}", i), err.(request.ErrInvalidParams))
			}
		}
	}
{{- end }}
{{ define "nestedStruct" -}}
    if s.{{ .Name }} != nil { 
		if err := s.{{ .Name }}.Validate(); err != nil {
			invalidParams.AddNested("{{ .Name }}", err.(request.ErrInvalidParams))
		}
	}
{{- end }}
`))

// GoCode returns the generated Go code for the Shape with its validation type.
func (sv ShapeValidation) GoCode() string {
	var err error

	w := &bytes.Buffer{}
	switch sv.Type {
	case ShapeValidationRequired:
		err = validationGoCodeTmpls.ExecuteTemplate(w, "requiredValue", sv)
	case ShapeValidationMinVal:
		switch sv.Ref.Shape.Type {
		case "list", "map", "blob":
			err = validationGoCodeTmpls.ExecuteTemplate(w, "minLen", sv)
		case "string":
			err = validationGoCodeTmpls.ExecuteTemplate(w, "minLenString", sv)
		case "integer", "long", "float", "double":
			err = validationGoCodeTmpls.ExecuteTemplate(w, "minVal", sv)
		default:
			panic(fmt.Sprintf("ShapeValidation.GoCode, %s's type %s, no min value handling",
				sv.Name, sv.Ref.Shape.Type))
		}
	case ShapeValidationNested:
		switch sv.Ref.Shape.Type {
		case "map", "list":
			err = validationGoCodeTmpls.ExecuteTemplate(w, "nestedMapList", sv)
		default:
			err = validationGoCodeTmpls.ExecuteTemplate(w, "nestedStruct", sv)
		}
	default:
		panic(fmt.Sprintf("ShapeValidation.GoCode, %s's type %d, unknown validation type",
			sv.Name, sv.Type))
	}

	if err != nil {
		panic(fmt.Sprintf("ShapeValidation.GoCode failed, err: %v", err))
	}

	return w.String()
}

// A ShapeValidations is a collection of shape validations needed nested within
// a parent shape
type ShapeValidations []ShapeValidation

var validateShapeTmpl = template.Must(template.New("ValidateShape").Parse(`
// Validate inspects the fields of the type to determine if they are valid.
func (s *{{ .Shape.ShapeName }}) Validate() error {
	invalidParams := request.ErrInvalidParams{Context: "{{ .Shape.ShapeName }}"}
	{{ range $_, $v := .Validations -}}
		{{ $v.GoCode }}
	{{ end }}
	if invalidParams.Len() > 0 {
		return invalidParams
	}
	return nil
}
`))

// GoCode generates the Go code needed to perform validations for the
// shape and its nested fields.
func (vs ShapeValidations) GoCode(shape *Shape) string {
	buf := &bytes.Buffer{}
	validateShapeTmpl.Execute(buf, map[string]interface{}{
		"Shape":       shape,
		"Validations": vs,
	})
	return buf.String()
}

// Has returns true or false if the ShapeValidations already contains the
// the reference and validation type.
func (vs ShapeValidations) Has(ref *ShapeRef, typ ShapeValidationType) bool {
	for _, v := range vs {
		if v.Ref == ref && v.Type == typ {
			return true
		}
	}
	return false
}
