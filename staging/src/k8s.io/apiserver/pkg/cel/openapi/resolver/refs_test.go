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

package resolver

import (
	"sync"
	"testing"

	"github.com/go-openapi/jsonreference"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

func schemaRef(ref string) *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Ref: spec.MustCreateRef(ref),
		},
	}
}

func newSchema(ref string) *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			ID: ref,
		},
	}
}

// rootSchema returns a schema with references in all places where TestPopulateRefs
// resolves references.
func rootSchema() *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Properties: map[string]spec.Schema{
				"a": *schemaRef("prop"),
			},
			AdditionalProperties: &spec.SchemaOrBool{
				Schema: schemaRef("additional"),
			},
			Items: &spec.SchemaOrArray{
				Schema: schemaRef("item"),
			},
		},
	}
}

func resolvedRootSchema() *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Properties: map[string]spec.Schema{
				"a": *newSchema("prop"),
			},
			AdditionalProperties: &spec.SchemaOrBool{
				Schema: newSchema("additional"),
			},
			Items: &spec.SchemaOrArray{
				Schema: newSchema("item"),
			},
		},
	}
}

func TestPopulateRefs(t *testing.T) {
	schemas := map[string]*spec.Schema{
		"root": rootSchema(),
	}

	// Add one schema for each of the references above, with an ID that allows
	// verifying that the right schema got resolved.
	for _, ref := range []string{"prop", "additional", "item"} {
		schemas[ref] = newSchema(ref)
	}

	schemaOf := func(ref string) (*spec.Schema, bool) {
		schema, ok := schemas[ref]
		return schema, ok
	}

	// Comparing the root schema below detects mutations where the data is semantically different,
	// but it cannot detect undesired writes where the thing being written is the same (unlikely,
	// but it could happen). Therefore we run two PopulateRefs calls and let the data race
	// detector warn about concurrent, uncoordinated writes.
	var wg sync.WaitGroup
	var actualResolved *spec.Schema
	wg.Go(func() {
		schema, err := PopulateRefs(schemaOf, "root")
		if err != nil {
			t.Errorf("first PopulateRefs failed: %v", err)
		}
		actualResolved = schema
	})
	wg.Go(func() {
		_, err := PopulateRefs(schemaOf, "root")
		if err != nil {
			t.Errorf("second PopulateRefs failed: %v", err)
		}
	})
	wg.Wait()

	if diff := cmp.Diff(resolvedRootSchema(), actualResolved, cmpopts.IgnoreUnexported(jsonreference.Ref{})); diff != "" {
		t.Errorf("unexpected resolved schema (- want, + got):\n%s", diff)
	}

	if diff := cmp.Diff(rootSchema(), schemas["root"], cmpopts.IgnoreUnexported(jsonreference.Ref{})); diff != "" {
		t.Errorf("read-only input schema got modified (- original, + modification):\n%s", diff)
	}
}
