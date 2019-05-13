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

package schema

import (
	"math/rand"
	"reflect"
	"regexp"
	"testing"
	"time"

	fuzz "github.com/google/gofuzz"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"
)

var nullTypeRE = regexp.MustCompile(`"type":\["([^"]*)","null"]`)

func TestStructuralRoundtrip(t *testing.T) {
	f := fuzz.New()
	seed := time.Now().UnixNano()
	t.Logf("seed = %v", seed)
	//seed = int64(1549012506261785182)
	f.RandSource(rand.New(rand.NewSource(seed)))
	f.Funcs(
		func(s *JSON, c fuzz.Continue) {
			switch c.Intn(6) {
			case 0:
				s.Object = float64(42.0)
			case 1:
				s.Object = map[string]interface{}{"foo": "bar"}
			case 2:
				s.Object = ""
			case 3:
				s.Object = []string{}
			case 4:
				s.Object = map[string]interface{}{}
			case 5:
				s.Object = nil
			}
		},
		func(g *Generic, c fuzz.Continue) {
			c.FuzzNoCustom(g)

			// TODO: make nullable in case of empty type survive go-openapi JSON -> API schema roundtrip
			// go-openapi does not support nullable. Adding it to a type slice produces OpenAPI v3
			// incompatible JSON which we cannot unmarshal (without string-replace magic to transform
			// null types back into nullable). If type is empty, nullable:true is not preserved
			// at all.
			if len(g.Type) == 0 {
				g.Nullable = false
			}
		},
	)
	f.MaxDepth(3)
	f.NilChance(0.5)

	for i := 0; i < 10000; i++ {
		orig := &Structural{}
		f.Fuzz(orig)

		// normalize Structural.ValueValidation to zero values if it was nil before
		normalizer := Visitor{
			Structural: func(s *Structural) bool {
				if s.ValueValidation == nil {
					s.ValueValidation = &ValueValidation{}
					return true
				}
				return false
			},
		}
		normalizer.Visit(orig)

		goOpenAPI := orig.ToGoOpenAPI()
		bs, err := json.Marshal(goOpenAPI)
		if err != nil {
			t.Fatal(err)
		}
		str := nullTypeRE.ReplaceAllString(string(bs), `"type":"$1","nullable":true`) // unfold nullable type:[<type>,"null"] -> type:<type>,nullable:true
		v1beta1Schema := &apiextensionsv1beta1.JSONSchemaProps{}
		err = json.Unmarshal([]byte(str), v1beta1Schema)
		if err != nil {
			t.Fatal(err)
		}
		internalSchema := &apiextensions.JSONSchemaProps{}
		err = apiextensionsv1beta1.Convert_v1beta1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v1beta1Schema, internalSchema, nil)
		if err != nil {
			t.Fatal(err)
		}
		s, err := NewStructural(internalSchema)
		if err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(orig, s) {
			t.Fatalf("original and result differ: %v", diff.ObjectDiff(orig, s))
		}
	}
}
