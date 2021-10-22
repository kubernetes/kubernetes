/*
Copyright 2019 The Kubernetes Authors.

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

package custom

import (
	"github.com/go-openapi/spec"
	"k8s.io/kube-openapi/pkg/common"
)

// +k8s:openapi-gen=true
type Bal struct{}

func (_ Bal) OpenAPIV3Definition() common.OpenAPIDefinition {
	return common.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type: []string{"string"},
			},
		},
	}
}

// +k8s:openapi-gen=true
type Bac struct{}

func (_ Bac) OpenAPIV3Definition() common.OpenAPIDefinition {
	return common.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type: []string{"object"},
			},
		},
	}
}

func (_ Bac) OpenAPIDefinition() common.OpenAPIDefinition {
	return common.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type: []string{"string"},
			},
		},
	}
}

// +k8s:openapi-gen=true
type Bah struct{}

func (_ Bah) OpenAPIV3Definition() common.OpenAPIDefinition {
	return common.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type: []string{"object"},
			},
		},
	}
}

func (_ Bah) OpenAPISchemaType() []string {
	return []string{"test-type"}
}

func (_ Bah) OpenAPISchemaFormat() string {
	return "test-format"
}
