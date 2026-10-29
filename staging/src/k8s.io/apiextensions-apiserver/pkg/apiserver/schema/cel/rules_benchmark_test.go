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

// Benchmarks in this file only use Kubernetes APIs (no direct cel-go imports)
// so that the same file builds against different cel-go versions and can be
// used for before/after comparisons of cel-go upgrades.

import (
	"context"
	"fmt"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-api/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

func benchStringSchema(maxLength int64, rules ...string) schema.Structural {
	s := schema.Structural{
		Generic:         schema.Generic{Type: "string"},
		ValueValidation: &schema.ValueValidation{MaxLength: &maxLength},
	}
	s.ValidationExtensions.XValidations = benchRules(rules...)
	return s
}

func benchIntegerSchema() schema.Structural {
	return schema.Structural{Generic: schema.Generic{Type: "integer"}}
}

func benchListSchema(items schema.Structural, maxItems int64, rules ...string) schema.Structural {
	s := schema.Structural{
		Generic:         schema.Generic{Type: "array"},
		Items:           &items,
		ValueValidation: &schema.ValueValidation{MaxItems: &maxItems},
	}
	s.ValidationExtensions.XValidations = benchRules(rules...)
	return s
}

func benchObjectSchema(props map[string]schema.Structural, rules ...string) schema.Structural {
	s := schema.Structural{
		Generic:    schema.Generic{Type: "object"},
		Properties: props,
	}
	s.ValidationExtensions.XValidations = benchRules(rules...)
	return s
}

func benchRules(rules ...string) apiextensionsv1.ValidationRules {
	if len(rules) == 0 {
		return nil
	}
	out := make(apiextensionsv1.ValidationRules, 0, len(rules))
	for _, r := range rules {
		out = append(out, apiextensionsv1.ValidationRule{Rule: r})
	}
	return out
}

// benchManyRulesSchema models a typical CRD: a list of structured items where
// every item carries several x-kubernetes-validations, plus rules at the root.
// Evaluating one object runs (rulesPerItem * len(items)) + rootRules separate
// CEL programs, which is how real CRDs exercise CEL.
func benchManyRulesSchema() schema.Structural {
	item := benchObjectSchema(map[string]schema.Structural{
		"name":     benchStringSchema(63),
		"image":    benchStringSchema(256),
		"replicas": benchIntegerSchema(),
		"tags":     benchListSchema(benchStringSchema(16), 16),
	},
		"self.replicas >= 0 && self.replicas <= 1000",
		"self.name.size() > 0 && self.name.size() <= 63",
		"self.image.startsWith('registry.example.com/')",
		"self.tags.all(t, t.size() <= 16)",
	)
	return benchObjectSchema(map[string]schema.Structural{
		"items": benchListSchema(item, 1000),
	},
		"self.items.all(i, i.replicas <= 1000)",
		"self.items.exists(i, i.name == 'app-0')",
	)
}

func benchManyRulesObject(n int) map[string]interface{} {
	items := make([]interface{}, n)
	for i := range n {
		items[i] = map[string]interface{}{
			"name":     fmt.Sprintf("app-%d", i),
			"image":    fmt.Sprintf("registry.example.com/app-%d:v1", i),
			"replicas": int64(i % 10),
			"tags":     []interface{}{"tier-web", "env-prod", "team-a"},
		}
	}
	return map[string]interface{}{"items": items}
}

// BenchmarkCRDManyRulesValidation measures validating one custom resource whose
// schema has rules on every list item. The cost is dominated by per-program
// evaluation overhead (activation setup, runtime cost tracking), which is the
// area cel-go reworked between v0.29 and v0.33.
func BenchmarkCRDManyRulesValidation(b *testing.B) {
	s := benchManyRulesSchema()
	validator := NewValidator(&s, true, celconfig.PerCallLimit)
	if validator == nil {
		b.Fatal("expected non nil validator")
	}
	ctx := context.TODO()
	root := field.NewPath("root")
	for _, n := range []int{1, 10, 100} {
		b.Run(fmt.Sprintf("items=%d", n), func(b *testing.B) {
			obj := benchManyRulesObject(n)
			if errs, _ := validator.Validate(ctx, root, &s, obj, nil, celconfig.RuntimeCELCostBudget); len(errs) > 0 {
				b.Fatalf("unexpected validation errors: %v", errs)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = validator.Validate(ctx, root, &s, obj, nil, celconfig.RuntimeCELCostBudget)
			}
		})
	}
}

// BenchmarkCRDManyRulesNewValidator measures building the validator for a CRD
// schema, which compiles (parse, type-check, cost-estimate, plan) every rule in
// the schema. This is paid on apiserver start and on every CRD create/update.
func BenchmarkCRDManyRulesNewValidator(b *testing.B) {
	for _, props := range []int{10, 50} {
		b.Run(fmt.Sprintf("properties=%d", props), func(b *testing.B) {
			fields := map[string]schema.Structural{}
			for p := range props {
				fields[fmt.Sprintf("field%d", p)] = benchStringSchema(64,
					"self.size() <= 64",
					"self.startsWith('x') || self.size() > 0",
				)
			}
			list := benchManyRulesSchema()
			fields["items"] = list.Properties["items"]
			s := benchObjectSchema(fields, list.ValidationExtensions.XValidations[0].Rule)
			if NewValidator(&s, true, celconfig.PerCallLimit) == nil {
				b.Fatal("expected non nil validator")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = NewValidator(&s, true, celconfig.PerCallLimit)
			}
		})
	}
}

// BenchmarkCRDRegexRules guards the regex fast path. Kubernetes relies on cel-go
// precompiling constant patterns (OptOptimize for matches(), and Kubernetes'
// own optimizations for find()/findAll()). If that optimization stops applying,
// allocs/op jumps because the pattern is recompiled on every evaluation.
// startsWith is included as a non-regex baseline.
func BenchmarkCRDRegexRules(b *testing.B) {
	const name = "web-123.example-45"
	for _, tc := range []struct {
		name string
		rule string
	}{
		{name: "matches", rule: `self.name.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$')`},
		{name: "find", rule: `self.name.find('[0-9]+') == '123'`},
		{name: "findAll", rule: `self.name.findAll('[0-9]+').size() == 2`},
		{name: "startsWith-baseline", rule: `self.name.startsWith('web-')`},
	} {
		b.Run(tc.name, func(b *testing.B) {
			s := benchObjectSchema(map[string]schema.Structural{
				"name": benchStringSchema(253),
			}, tc.rule)
			validator := NewValidator(&s, true, celconfig.PerCallLimit)
			if validator == nil {
				b.Fatal("expected non nil validator")
			}
			obj := map[string]interface{}{"name": name}
			ctx := context.TODO()
			root := field.NewPath("root")
			if errs, _ := validator.Validate(ctx, root, &s, obj, nil, celconfig.RuntimeCELCostBudget); len(errs) > 0 {
				b.Fatalf("unexpected validation errors: %v", errs)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = validator.Validate(ctx, root, &s, obj, nil, celconfig.RuntimeCELCostBudget)
			}
		})
	}
}
