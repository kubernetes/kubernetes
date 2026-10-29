/*
Copyright The Kubernetes Authors.

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

package cel

import (
	"testing"

	apiextensions "k8s.io/apiextensions-api/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

func BenchmarkCRDValidationCompilation(b *testing.B) {
	s := &schema.Structural{
		Generic: schema.Generic{Type: "object"},
		ValidationExtensions: schema.ValidationExtensions{
			XValidations: apiextensions.ValidationRules{
				{
					Rule:              "self.spec.replicas > 0 && self.spec.template.spec.containers.all(c, c.image != '')",
					Message:           "replicas must be positive and all containers must have an image",
					MessageExpression: `"invalid spec with replicas: " + string(self.spec.replicas)`,
				},
			},
		},
	}
	envSet := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	declType := model.SchemaDeclType(s, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Compile(s, declType, celconfig.PerCallLimit, envSet, NewExpressionsEnvLoader())
		if err != nil {
			b.Fatalf("unexpected compilation error: %v", err)
		}
	}
}
