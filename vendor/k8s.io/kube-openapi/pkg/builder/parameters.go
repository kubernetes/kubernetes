/*
Copyright 2023 The Kubernetes Authors.

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

package builder

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sort"
	"strconv"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

// deduplicateParameters finds parameters that are shared across multiple endpoints and replace them with
// references to the shared parameters in order to avoid repetition.
//
// deduplicateParameters does not mutate the source.
func deduplicateParameters(sp *spec.Swagger) (*spec.Swagger, error) {
	names, parameters, err := collectSharedParameters(sp)
	if err != nil {
		return nil, err
	}

	if sp.Parameters != nil {
		return nil, fmt.Errorf("shared parameters already exist") // should not happen with the builder, but to be sure
	}

	clone := *sp
	clone.Parameters = parameters
	return replaceSharedParameters(names, &clone)
}

// collectSharedParameters finds parameters that show up for many endpoints. These
// are basically all parameters with the exceptions of those where we know they are
// endpoint specific, e.g. because they reference the schema of the kind, or have
// the kind or resource name in the description.
func collectSharedParameters(sp *spec.Swagger) (namesByJSON map[string]string, ret map[string]spec.Parameter, err error) {
	if sp == nil || sp.Paths == nil {
		return nil, nil, nil
	}

	countsByJSON := map[string]int{}
	shared := map[string]spec.Parameter{}
	var keys []string

	collect := func(p *spec.Parameter) error {
		if (p.In == "query" || p.In == "path") && p.Name == "name" {
			return nil // ignore name parameter as they are never shared with the Kind in the description
		}
		if p.In == "query" && p.Name == "fieldValidation" {
			return nil // keep fieldValidation parameter unshared because kubectl uses it (until 1.27) to detect server-side field validation support
		}
		if p.In == "query" && p.Name == "dryRun" {
			return nil // keep fieldValidation parameter unshared because kubectl uses it (until 1.26) to detect dry-run support
		}
		if p.Schema != nil && p.In == "body" && p.Name == "body" && !strings.HasPrefix(p.Schema.Ref.String(), "#/definitions/io.k8s.apimachinery") {
			return nil // ignore non-generic body parameters as they reference the custom schema of the kind
		}

		bs, err := json.Marshal(p)
		if err != nil {
			return err
		}

		k := string(bs)
		countsByJSON[k]++
		if count := countsByJSON[k]; count == 1 {
			shared[k] = *p
			keys = append(keys, k)
		}

		return nil
	}

	for _, path := range sp.Paths.Paths {
		// per operation parameters
		for _, op := range operations(&path) {
			if op == nil {
				continue // shouldn't happen, but ignore if it does; tested through unit test
			}
			for _, p := range op.Parameters {
				if p.Ref.String() != "" {
					// shouldn't happen, but ignore if it does
					continue
				}
				if err := collect(&p); err != nil {
					return nil, nil, err
				}
			}
		}

		// per path parameters
		for _, p := range path.Parameters {
			if p.Ref.String() != "" {
				continue // shouldn't happen, but ignore if it does
			}
			if err := collect(&p); err != nil {
				return nil, nil, err
			}
		}
	}

	// name deterministically
	sort.Strings(keys)
	ret = map[string]spec.Parameter{}
	namesByJSON = map[string]string{}
	for _, k := range keys {
		name := shared[k].Name
		if name == "" {
			// this should never happen as the name is a required field. But if it does, let's be safe.
			name = "param"
		}
		name += "-" + base64Hash(k)
		i := 0
		for {
			if _, ok := ret[name]; !ok {
				ret[name] = shared[k]
				namesByJSON[k] = name
				break
			}
			i++ // only on hash conflict, unlikely with our few variants
			name = shared[k].Name + "-" + strconv.Itoa(i)
		}
	}

	return namesByJSON, ret, nil
}

func operations(path *spec.PathItem) []*spec.Operation {
	return []*spec.Operation{path.Get, path.Put, path.Post, path.Delete, path.Options, path.Head, path.Patch}
}

func base64Hash(s string) string {
	hash := fnv.New64()
	hash.Write([]byte(s))                                                      //nolint:errcheck
	return base64.URLEncoding.EncodeToString(hash.Sum(make([]byte, 0, 8))[:6]) // 8 characters
}

func replaceSharedParameters(sharedParameterNamesByJSON map[string]string, sp *spec.Swagger) (*spec.Swagger, error) {
	if sp == nil || sp.Paths == nil {
		return sp, nil
	}

	ret := sp

	firstPathChange := true
	for k, path := range sp.Paths.Paths {
		pathChanged := false

		// per operation parameters
		for _, op := range []**spec.Operation{&path.Get, &path.Put, &path.Post, &path.Delete, &path.Options, &path.Head, &path.Patch} {
			if *op == nil {
				continue
			}

			firstParamChange := true
			for i := range (*op).Parameters {
				p := (*op).Parameters[i]

				if p.Ref.String() != "" {
					// shouldn't happen, but be idem-potent if it does
					continue
				}

				bs, err := json.Marshal(p)
				if err != nil {
					return nil, err
				}

				if name, ok := sharedParameterNamesByJSON[string(bs)]; ok {
					if firstParamChange {
						orig := *op
						*op = &spec.Operation{}
						**op = *orig
						(*op).Parameters = make([]spec.Parameter, len(orig.Parameters))
						copy((*op).Parameters, orig.Parameters)
						firstParamChange = false
					}

					(*op).Parameters[i] = spec.Parameter{
						Refable: spec.Refable{
							Ref: spec.MustCreateRef("#/parameters/" + name),
						},
					}
					pathChanged = true
				}
			}
		}

		// per path parameters
		firstParamChange := true
		for i := range path.Parameters {
			p := path.Parameters[i]

			if p.Ref.String() != "" {
				// shouldn't happen, but be idem-potent if it does
				continue
			}

			bs, err := json.Marshal(p)
			if err != nil {
				return nil, err
			}

			if name, ok := sharedParameterNamesByJSON[string(bs)]; ok {
				if firstParamChange {
					orig := path.Parameters
					path.Parameters = make([]spec.Parameter, len(orig))
					copy(path.Parameters, orig)
					firstParamChange = false
				}

				path.Parameters[i] = spec.Parameter{
					Refable: spec.Refable{
						Ref: spec.MustCreateRef("#/parameters/" + name),
					},
				}
				pathChanged = true
			}
		}

		if pathChanged {
			if firstPathChange {
				clone := *sp
				ret = &clone

				pathsClone := *ret.Paths
				ret.Paths = &pathsClone

				ret.Paths.Paths = make(map[string]spec.PathItem, len(sp.Paths.Paths))
				for k, v := range sp.Paths.Paths {
					ret.Paths.Paths[k] = v
				}

				firstPathChange = false
			}
			ret.Paths.Paths[k] = path
		}
	}

	return ret, nil
}
