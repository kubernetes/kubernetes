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

package aggregator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/go-openapi/spec"
	fuzz "github.com/google/gofuzz"
	"k8s.io/kube-openapi/pkg/util/sets"
)

func fuzzFuncs(f *fuzz.Fuzzer, refFunc func(ref *spec.Ref, c fuzz.Continue, visible bool)) {
	invisible := 0 // == 0 means visible, > 0 means invisible
	depth := 0
	maxDepth := 3
	nilChance := func(depth int) float64 {
		return math.Pow(0.9, math.Max(0.0, float64(maxDepth-depth)))
	}
	updateFuzzer := func(depth int) {
		f.NilChance(nilChance(depth))
		f.NumElements(0, max(0, maxDepth-depth))
	}
	updateFuzzer(depth)
	enter := func(o interface{}, recursive bool, c fuzz.Continue) {
		if recursive {
			depth++
			updateFuzzer(depth)
		}

		invisible++
		c.FuzzNoCustom(o)
		invisible--
	}
	leave := func(recursive bool) {
		if recursive {
			depth--
			updateFuzzer(depth)
		}
	}
	f.Funcs(
		func(ref *spec.Ref, c fuzz.Continue) {
			refFunc(ref, c, invisible == 0)
		},
		func(sa *spec.SchemaOrStringArray, c fuzz.Continue) {
			*sa = spec.SchemaOrStringArray{}
			if c.RandBool() {
				c.Fuzz(&sa.Schema)
			} else {
				c.Fuzz(&sa.Property)
			}
			if sa.Schema == nil && len(sa.Property) == 0 {
				*sa = spec.SchemaOrStringArray{Schema: &spec.Schema{}}
			}
		},
		func(url *spec.SchemaURL, c fuzz.Continue) {
			*url = spec.SchemaURL("http://url")
		},
		func(s *spec.Swagger, c fuzz.Continue) {
			enter(s, false, c)
			defer leave(false)

			// only fuzz those fields we walk into with invisible==false
			c.Fuzz(&s.Parameters)
			c.Fuzz(&s.Responses)
			c.Fuzz(&s.Definitions)
			c.Fuzz(&s.Paths)
		},
		func(p *spec.PathItem, c fuzz.Continue) {
			enter(p, false, c)
			defer leave(false)

			// only fuzz those fields we walk into with invisible==false
			c.Fuzz(&p.Parameters)
			c.Fuzz(&p.Delete)
			c.Fuzz(&p.Get)
			c.Fuzz(&p.Head)
			c.Fuzz(&p.Options)
			c.Fuzz(&p.Patch)
			c.Fuzz(&p.Post)
			c.Fuzz(&p.Put)
		},
		func(p *spec.Parameter, c fuzz.Continue) {
			enter(p, false, c)
			defer leave(false)

			// only fuzz those fields we walk into with invisible==false
			c.Fuzz(&p.Ref)
			c.Fuzz(&p.Schema)
			if c.RandBool() {
				p.Items = &spec.Items{}
				c.Fuzz(&p.Items.Ref)
			} else {
				p.Items = nil
			}
		},
		func(s *spec.Response, c fuzz.Continue) {
			enter(s, false, c)
			defer leave(false)

			// only fuzz those fields we walk into with invisible==false
			c.Fuzz(&s.Ref)
			c.Fuzz(&s.Description)
			c.Fuzz(&s.Schema)
			c.Fuzz(&s.Examples)
		},
		func(s *spec.Dependencies, c fuzz.Continue) {
			enter(s, false, c)
			defer leave(false)

			// and nothing with invisible==false
		},
		func(p *spec.SimpleSchema, c fuzz.Continue) {
			// gofuzz is broken and calls this even for *SimpleSchema fields, ignoring NilChance, leading to infinite recursion
			if c.Float64() > nilChance(depth) {
				return
			}

			enter(p, true, c)
			defer leave(true)

			c.FuzzNoCustom(p)
		},
		func(s *spec.SchemaProps, c fuzz.Continue) {
			// gofuzz is broken and calls this even for *SchemaProps fields, ignoring NilChance, leading to infinite recursion
			if c.Float64() > nilChance(depth) {
				return
			}

			enter(s, true, c)
			defer leave(true)

			c.FuzzNoCustom(s)
		},
		func(i *interface{}, c fuzz.Continue) {
			// do nothing for examples and defaults. These are free form JSON fields.
		},
	)
}

func TestReplaceReferences(t *testing.T) {
	visibleRE, err := regexp.Compile("\"\\$ref\":\"(http://ref-[^\"]*)\"")
	if err != nil {
		t.Fatalf("failed to compile ref regex: %v", err)
	}
	invisibleRE, err := regexp.Compile("\"\\$ref\":\"(http://invisible-[^\"]*)\"")
	if err != nil {
		t.Fatalf("failed to compile ref regex: %v", err)
	}

	for i := 0; i < 1000; i++ {
		var visibleRefs, invisibleRefs sets.String
		var seed int64
		var randSource rand.Source
		var s *spec.Swagger
		for {
			visibleRefs = sets.NewString()
			invisibleRefs = sets.NewString()

			f := fuzz.New()
			seed = time.Now().UnixNano()
			//seed = int64(1549012506261785182)
			randSource = rand.New(rand.NewSource(seed))
			f.RandSource(randSource)

			visibleRefsNum := 0
			invisibleRefsNum := 0
			fuzzFuncs(f,
				func(ref *spec.Ref, c fuzz.Continue, visible bool) {
					var url string
					if visible {
						// this is a ref that is seen by the walker (we have some exceptions where we don't walk into)
						url = fmt.Sprintf("http://ref-%d", visibleRefsNum)
						visibleRefsNum++
					} else {
						// this is a ref that is not seen by the walker (we have some exceptions where we don't walk into)
						url = fmt.Sprintf("http://invisible-%d", invisibleRefsNum)
						invisibleRefsNum++
					}

					r, err := spec.NewRef(url)
					if err != nil {
						t.Fatalf("failed to fuzz ref: %v", err)
					}
					*ref = r
				},
			)

			// create random swagger spec with random URL references, but at least one ref
			s = &spec.Swagger{}
			f.Fuzz(s)

			// clone spec to normalize (fuzz might generate objects which do not roundtrip json marshalling
			var err error
			s, err = cloneSwagger(s)
			if err != nil {
				t.Fatalf("failed to normalize swagger after fuzzing: %v", err)
			}

			// find refs
			bs, err := json.Marshal(s)
			if err != nil {
				t.Fatalf("failed to marshal swagger: %v", err)
			}
			for _, m := range invisibleRE.FindAllStringSubmatch(string(bs), -1) {
				invisibleRefs.Insert(m[1])
			}
			if res := visibleRE.FindAllStringSubmatch(string(bs), -1); len(res) > 0 {
				for _, m := range res {
					visibleRefs.Insert(m[1])
				}
				break
			}
		}

		t.Run(fmt.Sprintf("iteration %d", i), func(t *testing.T) {
			mutatedRefs := sets.NewString()
			mutationProbability := rand.New(randSource).Float64()
			for _, vr := range visibleRefs.List() {
				if rand.New(randSource).Float64() > mutationProbability {
					mutatedRefs.Insert(vr)
				}
			}

			origString, err := json.Marshal(s)
			if err != nil {
				t.Fatalf("failed to marshal swagger: %v", err)
			}
			t.Logf("created schema with %d walked refs, %d invisible refs, mutating %v, seed %d: %s", visibleRefs.Len(), invisibleRefs.Len(), mutatedRefs.List(), seed, string(origString))

			// convert to json string, replace one of the refs, and unmarshal back
			mutatedString := string(origString)
			for _, r := range mutatedRefs.List() {
				mr := strings.Replace(r, "ref", "mutated", -1)
				mutatedString = strings.Replace(mutatedString, "\""+r+"\"", "\""+mr+"\"", -1)
			}
			mutatedViaJSON := &spec.Swagger{}
			if err := json.Unmarshal([]byte(mutatedString), mutatedViaJSON); err != nil {
				t.Fatalf("failed to unmarshal mutated spec: %v", err)
			}

			// replay the same mutation using the mutating walker
			seenRefs := sets.NewString()
			walker := mutatingReferenceWalker{
				walkRefCallback: func(ref *spec.Ref) *spec.Ref {
					seenRefs.Insert(ref.String())
					if mutatedRefs.Has(ref.String()) {
						r, err := spec.NewRef(strings.Replace(ref.String(), "ref", "mutated", -1))
						if err != nil {
							t.Fatalf("failed to create ref: %v", err)
						}
						return &r
					}
					return ref
				},
			}
			mutatedViaWalker := walker.Start(s)

			// compare that we got the same
			if !reflect.DeepEqual(mutatedViaJSON, mutatedViaWalker) {
				t.Errorf("mutation via walker differ from JSON text replacement (got A, expected B): %s", objectDiff(mutatedViaWalker, mutatedViaJSON))
			}
			if !seenRefs.HasAll(visibleRefs.List()...) {
				t.Errorf("expected to see the same refs in the walker as during fuzzing. Not seen: %v", visibleRefs.Difference(seenRefs).List())
			}
			if shouldNotSee := seenRefs.Intersection(invisibleRefs); shouldNotSee.Len() > 0 {
				t.Errorf("refs seen that the walker is not expected to see: %v", shouldNotSee.List())
			}
		})
	}
}

func cloneSwagger(orig *spec.Swagger) (*spec.Swagger, error) {
	bs, err := json.Marshal(orig)
	if err != nil {
		return nil, err
	}
	s := &spec.Swagger{}
	if err := json.Unmarshal(bs, s); err != nil {
		return nil, err
	}
	return s, nil
}

// stringDiff diffs a and b and returns a human readable diff.
func stringDiff(a, b string) string {
	ba := []byte(a)
	bb := []byte(b)
	out := []byte{}
	i := 0
	for ; i < len(ba) && i < len(bb); i++ {
		if ba[i] != bb[i] {
			break
		}
		out = append(out, ba[i])
	}
	out = append(out, []byte("\n\nA: ")...)
	out = append(out, ba[i:]...)
	out = append(out, []byte("\n\nB: ")...)
	out = append(out, bb[i:]...)
	out = append(out, []byte("\n\n")...)
	return string(out)
}

// objectDiff writes the two objects out as JSON and prints out the identical part of
// the objects followed by the remaining part of 'a' and finally the remaining part of 'b'.
// For debugging tests.
func objectDiff(a, b interface{}) string {
	ab, err := json.Marshal(a)
	if err != nil {
		panic(fmt.Sprintf("a: %v", err))
	}
	bb, err := json.Marshal(b)
	if err != nil {
		panic(fmt.Sprintf("b: %v", err))
	}
	return stringDiff(string(ab), string(bb))
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}
