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

package untyped

import (
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/errors"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

// NewAPI creates the default untyped API
func NewAPI(spec *loads.Document) *API {
	var an *analysis.Spec
	if spec != nil && spec.Spec() != nil {
		an = analysis.New(spec.Spec())
	}
	return &API{
		spec:            spec,
		analyzer:        an,
		DefaultProduces: runtime.JSONMime,
		DefaultConsumes: runtime.JSONMime,
		consumers: map[string]runtime.Consumer{
			runtime.JSONMime: runtime.JSONConsumer(),
		},
		producers: map[string]runtime.Producer{
			runtime.JSONMime: runtime.JSONProducer(),
		},
		authenticators: make(map[string]runtime.Authenticator),
		operations:     make(map[string]map[string]runtime.OperationHandler),
		ServeError:     errors.ServeError,
		Models:         make(map[string]func() interface{}),
		formats:        strfmt.NewFormats(),
	}
}

// API represents an untyped mux for a swagger spec
type API struct {
	spec            *loads.Document
	analyzer        *analysis.Spec
	DefaultProduces string
	DefaultConsumes string
	consumers       map[string]runtime.Consumer
	producers       map[string]runtime.Producer
	authenticators  map[string]runtime.Authenticator
	operations      map[string]map[string]runtime.OperationHandler
	ServeError      func(http.ResponseWriter, *http.Request, error)
	Models          map[string]func() interface{}
	formats         strfmt.Registry
}

// Formats returns the registered string formats
func (d *API) Formats() strfmt.Registry {
	if d.formats == nil {
		d.formats = strfmt.NewFormats()
	}
	return d.formats
}

// RegisterFormat registers a custom format validator
func (d *API) RegisterFormat(name string, format strfmt.Format, validator strfmt.Validator) {
	if d.formats == nil {
		d.formats = strfmt.NewFormats()
	}
	d.formats.Add(name, format, validator)
}

// RegisterAuth registers an auth handler in this api
func (d *API) RegisterAuth(scheme string, handler runtime.Authenticator) {
	if d.authenticators == nil {
		d.authenticators = make(map[string]runtime.Authenticator)
	}
	d.authenticators[scheme] = handler
}

// RegisterConsumer registers a consumer for a media type.
func (d *API) RegisterConsumer(mediaType string, handler runtime.Consumer) {
	if d.consumers == nil {
		d.consumers = map[string]runtime.Consumer{runtime.JSONMime: runtime.JSONConsumer()}
	}
	d.consumers[strings.ToLower(mediaType)] = handler
}

// RegisterProducer registers a producer for a media type
func (d *API) RegisterProducer(mediaType string, handler runtime.Producer) {
	if d.producers == nil {
		d.producers = map[string]runtime.Producer{runtime.JSONMime: runtime.JSONProducer()}
	}
	d.producers[strings.ToLower(mediaType)] = handler
}

// RegisterOperation registers an operation handler for an operation name
func (d *API) RegisterOperation(method, path string, handler runtime.OperationHandler) {
	if d.operations == nil {
		d.operations = make(map[string]map[string]runtime.OperationHandler)
	}
	um := strings.ToUpper(method)
	if b, ok := d.operations[um]; !ok || b == nil {
		d.operations[um] = make(map[string]runtime.OperationHandler)
	}
	d.operations[um][path] = handler
}

// OperationHandlerFor returns the operation handler for the specified id if it can be found
func (d *API) OperationHandlerFor(method, path string) (runtime.OperationHandler, bool) {
	if d.operations == nil {
		return nil, false
	}
	if pi, ok := d.operations[strings.ToUpper(method)]; ok {
		h, ok := pi[path]
		return h, ok
	}
	return nil, false
}

// ConsumersFor gets the consumers for the specified media types
func (d *API) ConsumersFor(mediaTypes []string) map[string]runtime.Consumer {
	result := make(map[string]runtime.Consumer)
	for _, mt := range mediaTypes {
		if consumer, ok := d.consumers[mt]; ok {
			result[mt] = consumer
		}
	}
	return result
}

// ProducersFor gets the producers for the specified media types
func (d *API) ProducersFor(mediaTypes []string) map[string]runtime.Producer {
	result := make(map[string]runtime.Producer)
	for _, mt := range mediaTypes {
		if producer, ok := d.producers[mt]; ok {
			result[mt] = producer
		}
	}
	return result
}

// AuthenticatorsFor gets the authenticators for the specified security schemes
func (d *API) AuthenticatorsFor(schemes map[string]spec.SecurityScheme) map[string]runtime.Authenticator {
	result := make(map[string]runtime.Authenticator)
	for k := range schemes {
		if a, ok := d.authenticators[k]; ok {
			result[k] = a
		}
	}
	return result
}

// Validate validates this API for any missing items
func (d *API) Validate() error {
	return d.validate()
}

// validateWith validates the registrations in this API against the provided spec analyzer
func (d *API) validate() error {
	var consumes []string
	for k := range d.consumers {
		consumes = append(consumes, k)
	}

	var produces []string
	for k := range d.producers {
		produces = append(produces, k)
	}

	var authenticators []string
	for k := range d.authenticators {
		authenticators = append(authenticators, k)
	}

	var operations []string
	for m, v := range d.operations {
		for p := range v {
			operations = append(operations, fmt.Sprintf("%s %s", strings.ToUpper(m), p))
		}
	}

	var definedAuths []string
	for k := range d.spec.Spec().SecurityDefinitions {
		definedAuths = append(definedAuths, k)
	}

	if err := d.verify("consumes", consumes, d.analyzer.RequiredConsumes()); err != nil {
		return err
	}
	if err := d.verify("produces", produces, d.analyzer.RequiredProduces()); err != nil {
		return err
	}
	if err := d.verify("operation", operations, d.analyzer.OperationIDs()); err != nil {
		return err
	}

	requiredAuths := d.analyzer.RequiredSecuritySchemes()
	if err := d.verify("auth scheme", authenticators, requiredAuths); err != nil {
		return err
	}
	fmt.Printf("comparing %s with %s\n", strings.Join(definedAuths, ","), strings.Join(requiredAuths, ","))
	if err := d.verify("security definitions", definedAuths, requiredAuths); err != nil {
		return err
	}
	return nil
}

func (d *API) verify(name string, registrations []string, expectations []string) error {

	sort.Sort(sort.StringSlice(registrations))
	sort.Sort(sort.StringSlice(expectations))

	expected := map[string]struct{}{}
	seen := map[string]struct{}{}

	for _, v := range expectations {
		expected[v] = struct{}{}
	}

	var unspecified []string
	for _, v := range registrations {
		seen[v] = struct{}{}
		if _, ok := expected[v]; !ok {
			unspecified = append(unspecified, v)
		}
	}

	for k := range seen {
		delete(expected, k)
	}

	var unregistered []string
	for k := range expected {
		unregistered = append(unregistered, k)
	}
	sort.Sort(sort.StringSlice(unspecified))
	sort.Sort(sort.StringSlice(unregistered))

	if len(unregistered) > 0 || len(unspecified) > 0 {
		return &errors.APIVerificationFailed{
			Section:              name,
			MissingSpecification: unspecified,
			MissingRegistration:  unregistered,
		}
	}

	return nil
}
