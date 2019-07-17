package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

const (
	// PathParameterKind = indicator of Request parameter type "path"
	PathParameterKind = iota

	// QueryParameterKind = indicator of Request parameter type "query"
	QueryParameterKind

	// BodyParameterKind = indicator of Request parameter type "body"
	BodyParameterKind

	// HeaderParameterKind = indicator of Request parameter type "header"
	HeaderParameterKind

	// FormParameterKind = indicator of Request parameter type "form"
	FormParameterKind

	// CollectionFormatCSV comma separated values `foo,bar`
	CollectionFormatCSV = CollectionFormat("csv")

	// CollectionFormatSSV space separated values `foo bar`
	CollectionFormatSSV = CollectionFormat("ssv")

	// CollectionFormatTSV tab separated values `foo\tbar`
	CollectionFormatTSV = CollectionFormat("tsv")

	// CollectionFormatPipes pipe separated values `foo|bar`
	CollectionFormatPipes = CollectionFormat("pipes")

	// CollectionFormatMulti corresponds to multiple parameter instances instead of multiple values for a single
	// instance `foo=bar&foo=baz`. This is valid only for QueryParameters and FormParameters
	CollectionFormatMulti = CollectionFormat("multi")
)

type CollectionFormat string

func (cf CollectionFormat) String() string {
	return string(cf)
}

// Parameter is for documententing the parameter used in a Http Request
// ParameterData kinds are Path,Query and Body
type Parameter struct {
	data *ParameterData
}

// ParameterData represents the state of a Parameter.
// It is made public to make it accessible to e.g. the Swagger package.
type ParameterData struct {
	Name, Description, DataType, DataFormat string
	Kind                                    int
	Required                                bool
	AllowableValues                         map[string]string
	AllowMultiple                           bool
	DefaultValue                            string
	CollectionFormat                        string
}

// Data returns the state of the Parameter
func (p *Parameter) Data() ParameterData {
	return *p.data
}

// Kind returns the parameter type indicator (see const for valid values)
func (p *Parameter) Kind() int {
	return p.data.Kind
}

func (p *Parameter) bePath() *Parameter {
	p.data.Kind = PathParameterKind
	return p
}
func (p *Parameter) beQuery() *Parameter {
	p.data.Kind = QueryParameterKind
	return p
}
func (p *Parameter) beBody() *Parameter {
	p.data.Kind = BodyParameterKind
	return p
}

func (p *Parameter) beHeader() *Parameter {
	p.data.Kind = HeaderParameterKind
	return p
}

func (p *Parameter) beForm() *Parameter {
	p.data.Kind = FormParameterKind
	return p
}

// Required sets the required field and returns the receiver
func (p *Parameter) Required(required bool) *Parameter {
	p.data.Required = required
	return p
}

// AllowMultiple sets the allowMultiple field and returns the receiver
func (p *Parameter) AllowMultiple(multiple bool) *Parameter {
	p.data.AllowMultiple = multiple
	return p
}

// AllowableValues sets the allowableValues field and returns the receiver
func (p *Parameter) AllowableValues(values map[string]string) *Parameter {
	p.data.AllowableValues = values
	return p
}

// DataType sets the dataType field and returns the receiver
func (p *Parameter) DataType(typeName string) *Parameter {
	p.data.DataType = typeName
	return p
}

// DataFormat sets the dataFormat field for Swagger UI
func (p *Parameter) DataFormat(formatName string) *Parameter {
	p.data.DataFormat = formatName
	return p
}

// DefaultValue sets the default value field and returns the receiver
func (p *Parameter) DefaultValue(stringRepresentation string) *Parameter {
	p.data.DefaultValue = stringRepresentation
	return p
}

// Description sets the description value field and returns the receiver
func (p *Parameter) Description(doc string) *Parameter {
	p.data.Description = doc
	return p
}

// CollectionFormat sets the collection format for an array type
func (p *Parameter) CollectionFormat(format CollectionFormat) *Parameter {
	p.data.CollectionFormat = format.String()
	return p
}
