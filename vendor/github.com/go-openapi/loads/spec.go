// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package loads

import (
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

// JSONDoc loads a json document from either a file or a remote url
func JSONDoc(path string) (json.RawMessage, error) {
	data, err := swag.LoadFromFileOrHTTP(path)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(data), nil
}

// DocLoader represents a doc loader type
type DocLoader func(string) (json.RawMessage, error)

// DocMatcher represents a predicate to check if a loader matches
type DocMatcher func(string) bool

var loaders = &loader{Match: func(_ string) bool { return true }, Fn: JSONDoc}

// AddLoader for a document
func AddLoader(predicate DocMatcher, load DocLoader) {
	prev := loaders
	loaders = &loader{
		Match: predicate,
		Fn:    load,
		Next:  prev,
	}

}

type loader struct {
	Fn    DocLoader
	Match DocMatcher
	Next  *loader
}

// JSONSpec loads a spec from a json document
func JSONSpec(path string) (*Document, error) {
	data, err := JSONDoc(path)
	if err != nil {
		return nil, err
	}
	// convert to json
	return Analyzed(json.RawMessage(data), "")
}

// Document represents a swagger spec document
type Document struct {
	// specAnalyzer
	Analyzer *analysis.Spec
	spec     *spec.Swagger
	origSpec *spec.Swagger
	schema   *spec.Schema
	raw      json.RawMessage
}

// Spec loads a new spec document
func Spec(path string) (*Document, error) {
	specURL, err := url.Parse(path)
	if err != nil {
		return nil, err
	}
	for l := loaders.Next; l != nil; l = l.Next {
		if loaders.Match(specURL.Path) {
			b, err2 := loaders.Fn(path)
			if err2 != nil {
				return nil, err2
			}
			return Analyzed(b, "")
		}
	}
	b, err := loaders.Fn(path)
	if err != nil {
		return nil, err
	}
	return Analyzed(b, "")
}

var swag20Schema = spec.MustLoadSwagger20Schema()

// Analyzed creates a new analyzed spec document
func Analyzed(data json.RawMessage, version string) (*Document, error) {
	if version == "" {
		version = "2.0"
	}
	if version != "2.0" {
		return nil, fmt.Errorf("spec version %q is not supported", version)
	}

	swspec := new(spec.Swagger)
	if err := json.Unmarshal(data, swspec); err != nil {
		return nil, err
	}

	origsqspec := new(spec.Swagger)
	if err := json.Unmarshal(data, origsqspec); err != nil {
		return nil, err
	}

	d := &Document{
		Analyzer: analysis.New(swspec),
		schema:   swag20Schema,
		spec:     swspec,
		raw:      data,
		origSpec: origsqspec,
	}
	return d, nil
}

// Expanded expands the ref fields in the spec document and returns a new spec document
func (d *Document) Expanded() (*Document, error) {
	swspec := new(spec.Swagger)
	if err := json.Unmarshal(d.raw, swspec); err != nil {
		return nil, err
	}
	if err := spec.ExpandSpec(swspec); err != nil {
		return nil, err
	}

	dd := &Document{
		Analyzer: analysis.New(swspec),
		spec:     swspec,
		schema:   swag20Schema,
		raw:      d.raw,
		origSpec: d.origSpec,
	}
	return dd, nil
}

// BasePath the base path for this spec
func (d *Document) BasePath() string {
	return d.spec.BasePath
}

// Version returns the version of this spec
func (d *Document) Version() string {
	return d.spec.Swagger
}

// Schema returns the swagger 2.0 schema
func (d *Document) Schema() *spec.Schema {
	return d.schema
}

// Spec returns the swagger spec object model
func (d *Document) Spec() *spec.Swagger {
	return d.spec
}

// Host returns the host for the API
func (d *Document) Host() string {
	return d.spec.Host
}

// Raw returns the raw swagger spec as json bytes
func (d *Document) Raw() json.RawMessage {
	return d.raw
}

func (d *Document) OrigSpec() *spec.Swagger {
	return d.origSpec
}

// ResetDefinitions gives a shallow copy with the models reset
func (d *Document) ResetDefinitions() *Document {
	defs := make(map[string]spec.Schema, len(d.origSpec.Definitions))
	for k, v := range d.origSpec.Definitions {
		defs[k] = v
	}

	d.spec.Definitions = defs
	return d
}

// Pristine creates a new pristine document instance based on the input data
func (d *Document) Pristine() *Document {
	dd, _ := Analyzed(d.Raw(), d.Version())
	return dd
}
