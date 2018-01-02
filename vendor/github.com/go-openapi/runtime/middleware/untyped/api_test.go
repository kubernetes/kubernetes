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
	"io"
	"sort"
	"testing"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/errors"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/runtime"
	swaggerspec "github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
)

func stubAutenticator() runtime.Authenticator {
	return runtime.AuthenticatorFunc(func(_ interface{}) (bool, interface{}, error) { return false, nil, nil })
}

type stubConsumer struct {
}

func (s *stubConsumer) Consume(_ io.Reader, _ interface{}) error {
	return nil
}

type stubProducer struct {
}

func (s *stubProducer) Produce(_ io.Writer, _ interface{}) error {
	return nil
}

type stubOperationHandler struct {
}

func (s *stubOperationHandler) ParameterModel() interface{} {
	return nil
}

func (s *stubOperationHandler) Handle(params interface{}) (interface{}, error) {
	return nil, nil
}

func TestUntypedAPIRegistrations(t *testing.T) {
	api := NewAPI(new(loads.Document))

	api.RegisterConsumer("application/yada", new(stubConsumer))
	api.RegisterProducer("application/yada-2", new(stubProducer))
	api.RegisterOperation("get", "/{someId}", new(stubOperationHandler))
	api.RegisterAuth("basic", stubAutenticator())

	assert.NotEmpty(t, api.authenticators)

	_, ok := api.authenticators["basic"]
	assert.True(t, ok)
	_, ok = api.consumers["application/yada"]
	assert.True(t, ok)
	_, ok = api.producers["application/yada-2"]
	assert.True(t, ok)
	_, ok = api.consumers["application/json"]
	assert.True(t, ok)
	_, ok = api.producers["application/json"]
	assert.True(t, ok)
	_, ok = api.operations["GET"]["/{someId}"]
	assert.True(t, ok)

	h, ok := api.OperationHandlerFor("get", "/{someId}")
	assert.True(t, ok)
	assert.NotNil(t, h)

	_, ok = api.OperationHandlerFor("doesntExist", "/{someId}")
	assert.False(t, ok)
}

func TestUntypedAppValidation(t *testing.T) {
	invalidSpecStr := `{
  "consumes": ["application/json"],
  "produces": ["application/json"],
  "security": [
    {"apiKey":[]}
  ],
  "parameters": {
    "format": {
      "in": "query",
      "name": "format",
      "type": "string"
    }
  },
  "paths": {
    "/": {
      "parameters": [
        {
          "name": "limit",
          "type": "integer",
          "format": "int32",
          "x-go-name": "Limit"
        }
      ],
      "get": {
        "consumes": ["application/x-yaml"],
        "produces": ["application/x-yaml"],
        "security": [
          {"basic":[]}
        ],
        "operationId": "someOperation",
        "parameters": [
          {
            "name": "skip",
            "type": "integer",
            "format": "int32"
          }
        ]
      }
    }
  }
}`
	specStr := `{
	  "consumes": ["application/json"],
	  "produces": ["application/json"],
	  "security": [
	    {"apiKey":[]}
	  ],
	  "securityDefinitions": {
	    "basic": { "type": "basic" },
	    "apiKey": { "type": "apiKey", "in":"header", "name":"X-API-KEY" }
	  },
	  "parameters": {
	  	"format": {
	  		"in": "query",
	  		"name": "format",
	  		"type": "string"
	  	}
	  },
	  "paths": {
	  	"/": {
	  		"parameters": [
	  			{
	  				"name": "limit",
			  		"type": "integer",
			  		"format": "int32",
			  		"x-go-name": "Limit"
			  	}
	  		],
	  		"get": {
	  			"consumes": ["application/x-yaml"],
	  			"produces": ["application/x-yaml"],
	        "security": [
	          {"basic":[]}
	        ],
	  			"operationId": "someOperation",
	  			"parameters": [
	  				{
				  		"name": "skip",
				  		"type": "integer",
				  		"format": "int32"
				  	}
	  			]
	  		}
	  	}
	  }
	}`
	validSpec, err := loads.Analyzed([]byte(specStr), "")
	assert.NoError(t, err)
	assert.NotNil(t, validSpec)

	spec, err := loads.Analyzed([]byte(invalidSpecStr), "")
	assert.NoError(t, err)
	assert.NotNil(t, spec)

	analyzed := analysis.New(spec.Spec())
	analyzedValid := analysis.New(validSpec.Spec())
	cons := analyzed.ConsumesFor(analyzed.AllPaths()["/"].Get)
	assert.Len(t, cons, 1)
	prods := analyzed.RequiredProduces()
	assert.Len(t, prods, 2)

	api1 := NewAPI(spec)
	err = api1.Validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [application/x-yaml] consumes registrations", err.Error())
	api1.RegisterConsumer("application/x-yaml", new(stubConsumer))
	err = api1.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [application/x-yaml] produces registrations", err.Error())
	api1.RegisterProducer("application/x-yaml", new(stubProducer))
	err = api1.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [GET /] operation registrations", err.Error())
	api1.RegisterOperation("get", "/", new(stubOperationHandler))
	err = api1.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [apiKey, basic] auth scheme registrations", err.Error())
	api1.RegisterAuth("basic", stubAutenticator())
	api1.RegisterAuth("apiKey", stubAutenticator())
	err = api1.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [apiKey, basic] security definitions registrations", err.Error())

	api3 := NewAPI(validSpec)
	api3.RegisterConsumer("application/x-yaml", new(stubConsumer))
	api3.RegisterProducer("application/x-yaml", new(stubProducer))
	api3.RegisterOperation("get", "/", new(stubOperationHandler))
	api3.RegisterAuth("basic", stubAutenticator())
	api3.RegisterAuth("apiKey", stubAutenticator())
	err = api3.validate()
	assert.NoError(t, err)
	api3.RegisterConsumer("application/something", new(stubConsumer))
	err = api3.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing from spec file [application/something] consumes", err.Error())

	api2 := NewAPI(spec)
	api2.RegisterConsumer("application/something", new(stubConsumer))
	err = api2.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [application/x-yaml] consumes registrations\nmissing from spec file [application/something] consumes", err.Error())
	api2.RegisterConsumer("application/x-yaml", new(stubConsumer))
	delete(api2.consumers, "application/something")
	api2.RegisterProducer("application/something", new(stubProducer))
	err = api2.validate()
	assert.Error(t, err)
	assert.Equal(t, "missing [application/x-yaml] produces registrations\nmissing from spec file [application/something] produces", err.Error())
	delete(api2.producers, "application/something")
	api2.RegisterProducer("application/x-yaml", new(stubProducer))

	expected := []string{"application/x-yaml"}
	sort.Sort(sort.StringSlice(expected))
	consumes := analyzed.ConsumesFor(analyzed.AllPaths()["/"].Get)
	sort.Sort(sort.StringSlice(consumes))
	assert.Equal(t, expected, consumes)
	consumers := api1.ConsumersFor(consumes)
	assert.Len(t, consumers, 1)

	produces := analyzed.ProducesFor(analyzed.AllPaths()["/"].Get)
	sort.Sort(sort.StringSlice(produces))
	assert.Equal(t, expected, produces)
	producers := api1.ProducersFor(produces)
	assert.Len(t, producers, 1)

	definitions := analyzedValid.SecurityDefinitionsFor(analyzedValid.AllPaths()["/"].Get)
	expectedSchemes := map[string]swaggerspec.SecurityScheme{"basic": *swaggerspec.BasicAuth()}
	assert.Equal(t, expectedSchemes, definitions)
	authenticators := api3.AuthenticatorsFor(definitions)
	assert.Len(t, authenticators, 1)

	opHandler := runtime.OperationHandlerFunc(func(data interface{}) (interface{}, error) {
		return data, nil
	})
	d, err := opHandler.Handle(1)
	assert.NoError(t, err)
	assert.Equal(t, 1, d)

	authenticator := runtime.AuthenticatorFunc(func(params interface{}) (bool, interface{}, error) {
		if str, ok := params.(string); ok {
			return ok, str, nil
		}
		return true, nil, errors.Unauthenticated("authenticator")
	})
	ok, p, err := authenticator.Authenticate("hello")
	assert.True(t, ok)
	assert.NoError(t, err)
	assert.Equal(t, "hello", p)
}
