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

package install

import (
	"strings"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
)

func TestRoundTripTypes(t *testing.T) {
	apitesting.RoundTripTestForAPIGroup(t, Install, apitesting.MergeFuzzerFuncs(t,
		fuzzerFuncs(),
	))
}

func fuzzerFuncs() []interface{} {
	return []interface{}{
		func(obj *apiextensions.CustomResourceDefinitionSpec, c fuzz.Continue) {
			c.FuzzNoCustom(obj)

			// match our defaulter
			if len(obj.Scope) == 0 {
				obj.Scope = apiextensions.NamespaceScoped
			}
			if len(obj.Names.Singular) == 0 {
				obj.Names.Singular = strings.ToLower(obj.Names.Kind)
			}
			if len(obj.Names.ListKind) == 0 && len(obj.Names.Kind) > 0 {
				obj.Names.ListKind = obj.Names.Kind + "List"
			}
		},
	}
}
