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

package analysis

import (
	"fmt"
	"reflect"

	"github.com/go-openapi/spec"
)

// Mixin modifies the primary swagger spec by adding the paths and
// definitions from the mixin specs. Top level parameters and
// responses from the mixins are also carried over. Operation id
// collisions are avoided by appending "Mixin<N>" but only if
// needed.
//
// The following parts of primary are subject to merge, filling empty details
//   - Info
//   - BasePath
//   - Host
//   - ExternalDocs
//
// Consider calling FixEmptyResponseDescriptions() on the modified primary
// if you read them from storage and they are valid to start with.
//
// Entries in "paths", "definitions", "parameters" and "responses" are
// added to the primary in the order of the given mixins. If the entry
// already exists in primary it is skipped with a warning message.
//
// The count of skipped entries (from collisions) is returned so any
// deviation from the number expected can flag a warning in your build
// scripts. Carefully review the collisions before accepting them;
// consider renaming things if possible.
//
// No key normalization takes place (paths, type defs,
// etc). Ensure they are canonical if your downstream tools do
// key normalization of any form.
//
// Merging schemes (http, https), and consumers/producers do not account for
// collisions.
func Mixin(primary *spec.Swagger, mixins ...*spec.Swagger) []string {
	skipped := make([]string, 0, len(mixins))
	opIds := getOpIds(primary)
	initPrimary(primary)

	for i, m := range mixins {
		skipped = append(skipped, mergeSwaggerProps(primary, m)...)

		skipped = append(skipped, mergeConsumes(primary, m)...)

		skipped = append(skipped, mergeProduces(primary, m)...)

		skipped = append(skipped, mergeTags(primary, m)...)

		skipped = append(skipped, mergeSchemes(primary, m)...)

		skipped = append(skipped, mergeSecurityDefinitions(primary, m)...)

		skipped = append(skipped, mergeSecurityRequirements(primary, m)...)

		skipped = append(skipped, mergeDefinitions(primary, m)...)

		// merging paths requires a map of operationIDs to work with
		skipped = append(skipped, mergePaths(primary, m, opIds, i)...)

		skipped = append(skipped, mergeParameters(primary, m)...)

		skipped = append(skipped, mergeResponses(primary, m)...)
	}
	return skipped
}

// getOpIds extracts all the paths.<path>.operationIds from the given
// spec and returns them as the keys in a map with 'true' values.
func getOpIds(s *spec.Swagger) map[string]bool {
	rv := make(map[string]bool)
	if s.Paths == nil {
		return rv
	}
	for _, v := range s.Paths.Paths {
		piops := pathItemOps(v)
		for _, op := range piops {
			rv[op.ID] = true
		}
	}
	return rv
}

func pathItemOps(p spec.PathItem) []*spec.Operation {
	var rv []*spec.Operation
	rv = appendOp(rv, p.Get)
	rv = appendOp(rv, p.Put)
	rv = appendOp(rv, p.Post)
	rv = appendOp(rv, p.Delete)
	rv = appendOp(rv, p.Head)
	rv = appendOp(rv, p.Patch)
	return rv
}

func appendOp(ops []*spec.Operation, op *spec.Operation) []*spec.Operation {
	if op == nil {
		return ops
	}
	return append(ops, op)
}

func mergeSecurityDefinitions(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for k, v := range m.SecurityDefinitions {
		if _, exists := primary.SecurityDefinitions[k]; exists {
			warn := fmt.Sprintf(
				"SecurityDefinitions entry '%v' already exists in primary or higher priority mixin, skipping\n", k)
			skipped = append(skipped, warn)
			continue
		}
		primary.SecurityDefinitions[k] = v
	}
	return
}

func mergeSecurityRequirements(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for _, v := range m.Security {
		found := false
		for _, vv := range primary.Security {
			if reflect.DeepEqual(v, vv) {
				found = true
				break
			}
		}
		if found {
			warn := fmt.Sprintf(
				"Security requirement: '%v' already exists in primary or higher priority mixin, skipping\n", v)
			skipped = append(skipped, warn)
			continue
		}
		primary.Security = append(primary.Security, v)
	}
	return
}

func mergeDefinitions(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for k, v := range m.Definitions {
		// assume name collisions represent IDENTICAL type. careful.
		if _, exists := primary.Definitions[k]; exists {
			warn := fmt.Sprintf(
				"definitions entry '%v' already exists in primary or higher priority mixin, skipping\n", k)
			skipped = append(skipped, warn)
			continue
		}
		primary.Definitions[k] = v
	}
	return
}

func mergePaths(primary *spec.Swagger, m *spec.Swagger, opIds map[string]bool, mixIndex int) (skipped []string) {
	if m.Paths != nil {
		for k, v := range m.Paths.Paths {
			if _, exists := primary.Paths.Paths[k]; exists {
				warn := fmt.Sprintf(
					"paths entry '%v' already exists in primary or higher priority mixin, skipping\n", k)
				skipped = append(skipped, warn)
				continue
			}

			// Swagger requires that operationIds be
			// unique within a spec. If we find a
			// collision we append "Mixin0" to the
			// operatoinId we are adding, where 0 is mixin
			// index.  We assume that operationIds with
			// all the proivded specs are already unique.
			piops := pathItemOps(v)
			for _, piop := range piops {
				if opIds[piop.ID] {
					piop.ID = fmt.Sprintf("%v%v%v", piop.ID, "Mixin", mixIndex)
				}
				opIds[piop.ID] = true
			}
			primary.Paths.Paths[k] = v
		}
	}
	return
}

func mergeParameters(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for k, v := range m.Parameters {
		// could try to rename on conflict but would
		// have to fix $refs in the mixin. Complain
		// for now
		if _, exists := primary.Parameters[k]; exists {
			warn := fmt.Sprintf(
				"top level parameters entry '%v' already exists in primary or higher priority mixin, skipping\n", k)
			skipped = append(skipped, warn)
			continue
		}
		primary.Parameters[k] = v
	}
	return
}

func mergeResponses(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for k, v := range m.Responses {
		// could try to rename on conflict but would
		// have to fix $refs in the mixin. Complain
		// for now
		if _, exists := primary.Responses[k]; exists {
			warn := fmt.Sprintf(
				"top level responses entry '%v' already exists in primary or higher priority mixin, skipping\n", k)
			skipped = append(skipped, warn)
			continue
		}
		primary.Responses[k] = v
	}
	return skipped
}

func mergeConsumes(primary *spec.Swagger, m *spec.Swagger) []string {
	for _, v := range m.Consumes {
		found := false
		for _, vv := range primary.Consumes {
			if v == vv {
				found = true
				break
			}
		}
		if found {
			// no warning here: we just skip it
			continue
		}
		primary.Consumes = append(primary.Consumes, v)
	}
	return []string{}
}

func mergeProduces(primary *spec.Swagger, m *spec.Swagger) []string {
	for _, v := range m.Produces {
		found := false
		for _, vv := range primary.Produces {
			if v == vv {
				found = true
				break
			}
		}
		if found {
			// no warning here: we just skip it
			continue
		}
		primary.Produces = append(primary.Produces, v)
	}
	return []string{}
}

func mergeTags(primary *spec.Swagger, m *spec.Swagger) (skipped []string) {
	for _, v := range m.Tags {
		found := false
		for _, vv := range primary.Tags {
			if v.Name == vv.Name {
				found = true
				break
			}
		}
		if found {
			warn := fmt.Sprintf(
				"top level tags entry with name '%v' already exists in primary or higher priority mixin, skipping\n", v.Name)
			skipped = append(skipped, warn)
			continue
		}
		primary.Tags = append(primary.Tags, v)
	}
	return
}

func mergeSchemes(primary *spec.Swagger, m *spec.Swagger) []string {
	for _, v := range m.Schemes {
		found := false
		for _, vv := range primary.Schemes {
			if v == vv {
				found = true
				break
			}
		}
		if found {
			// no warning here: we just skip it
			continue
		}
		primary.Schemes = append(primary.Schemes, v)
	}
	return []string{}
}

func mergeSwaggerProps(primary *spec.Swagger, m *spec.Swagger) []string {
	var skipped []string
	primary.Extensions, skipped = mergeExtensions(primary.Extensions, m.Extensions)

	// merging details in swagger top properties
	if primary.Host == "" {
		primary.Host = m.Host
	}
	if primary.BasePath == "" {
		primary.BasePath = m.BasePath
	}
	if primary.Info == nil {
		primary.Info = m.Info
	} else if m.Info != nil {
		var sk []string
		primary.Info.Extensions, sk = mergeExtensions(primary.Info.Extensions, m.Info.Extensions)
		skipped = append(skipped, sk...)
		if primary.Info.Description == "" {
			primary.Info.Description = m.Info.Description
		}
		if primary.Info.Title == "" {
			primary.Info.Description = m.Info.Description
		}
		if primary.Info.TermsOfService == "" {
			primary.Info.TermsOfService = m.Info.TermsOfService
		}
		if primary.Info.Version == "" {
			primary.Info.Version = m.Info.Version
		}

		if primary.Info.Contact == nil {
			primary.Info.Contact = m.Info.Contact
		} else if m.Info.Contact != nil {
			if primary.Info.Contact.Name == "" {
				primary.Info.Contact.Name = m.Info.Contact.Name
			}
			if primary.Info.Contact.URL == "" {
				primary.Info.Contact.URL = m.Info.Contact.URL
			}
			if primary.Info.Contact.Email == "" {
				primary.Info.Contact.Email = m.Info.Contact.Email
			}
		}

		if primary.Info.License == nil {
			primary.Info.License = m.Info.License
		} else if m.Info.License != nil {
			if primary.Info.License.Name == "" {
				primary.Info.License.Name = m.Info.License.Name
			}
			if primary.Info.License.URL == "" {
				primary.Info.License.URL = m.Info.License.URL
			}
		}

	}
	if primary.ExternalDocs == nil {
		primary.ExternalDocs = m.ExternalDocs
	} else if m.ExternalDocs != nil {
		if primary.ExternalDocs.Description == "" {
			primary.ExternalDocs.Description = m.ExternalDocs.Description
		}
		if primary.ExternalDocs.URL == "" {
			primary.ExternalDocs.URL = m.ExternalDocs.URL
		}
	}
	return skipped
}

func mergeExtensions(primary spec.Extensions, m spec.Extensions) (result spec.Extensions, skipped []string) {
	if primary == nil {
		result = m
		return
	}
	if m == nil {
		result = primary
		return
	}
	result = primary
	for k, v := range m {
		if _, found := primary[k]; found {
			skipped = append(skipped, k)
			continue
		}
		primary[k] = v
	}
	return
}

func initPrimary(primary *spec.Swagger) {
	if primary.SecurityDefinitions == nil {
		primary.SecurityDefinitions = make(map[string]*spec.SecurityScheme)
	}
	if primary.Security == nil {
		primary.Security = make([]map[string][]string, 0, 10)
	}
	if primary.Produces == nil {
		primary.Produces = make([]string, 0, 10)
	}
	if primary.Consumes == nil {
		primary.Consumes = make([]string, 0, 10)
	}
	if primary.Tags == nil {
		primary.Tags = make([]spec.Tag, 0, 10)
	}
	if primary.Schemes == nil {
		primary.Schemes = make([]string, 0, 10)
	}
	if primary.Paths == nil {
		primary.Paths = &spec.Paths{Paths: make(map[string]spec.PathItem)}
	}
	if primary.Paths.Paths == nil {
		primary.Paths.Paths = make(map[string]spec.PathItem)
	}
	if primary.Definitions == nil {
		primary.Definitions = make(spec.Definitions)
	}
	if primary.Parameters == nil {
		primary.Parameters = make(map[string]spec.Parameter)
	}
	if primary.Responses == nil {
		primary.Responses = make(map[string]spec.Response)
	}
}
