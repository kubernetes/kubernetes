/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import (
	"fmt"
	"sort"

	"k8s.io/gengo/types"
)

const tagUnionMember = "union"
const tagUnionDeprecated = "unionDeprecated"
const tagUnionDiscriminator = "unionDiscriminator"

type union struct {
	discriminator         string
	fieldsToDiscriminated map[string]string
}

// emit prints the union, can be called on a nil union (emits nothing)
func (u *union) emit(g openAPITypeWriter) {
	if u == nil {
		return
	}
	g.Do("map[string]interface{}{\n", nil)
	if u.discriminator != "" {
		g.Do("\"discriminator\": \"$.$\",\n", u.discriminator)
	}
	g.Do("\"fields-to-discriminateBy\": map[string]interface{}{\n", nil)
	keys := []string{}
	for field := range u.fieldsToDiscriminated {
		keys = append(keys, field)
	}
	sort.Strings(keys)
	for _, field := range keys {
		g.Do("\"$.$\": ", field)
		g.Do("\"$.$\",\n", u.fieldsToDiscriminated[field])
	}
	g.Do("},\n", nil)
	g.Do("},\n", nil)
}

// Sets the discriminator if it's not set yet, otherwise return an error
func (u *union) setDiscriminator(value string) []error {
	errors := []error{}
	if u.discriminator != "" {
		errors = append(errors, fmt.Errorf("at least two discriminators found: %v and %v", value, u.discriminator))
	}
	u.discriminator = value
	return errors
}

// Add a new member to the union
func (u *union) addMember(jsonName, variableName string) {
	if _, ok := u.fieldsToDiscriminated[jsonName]; ok {
		panic(fmt.Errorf("same field (%v) found multiple times", jsonName))
	}
	u.fieldsToDiscriminated[jsonName] = variableName
}

// Makes sure that the union is valid, specifically looking for re-used discriminated
func (u *union) isValid() []error {
	errors := []error{}
	// Case 1: discriminator but no fields
	if u.discriminator != "" && len(u.fieldsToDiscriminated) == 0 {
		errors = append(errors, fmt.Errorf("discriminator set with no fields in union"))
	}
	// Case 2: two fields have the same discriminated value
	discriminated := map[string]struct{}{}
	for _, d := range u.fieldsToDiscriminated {
		if _, ok := discriminated[d]; ok {
			errors = append(errors, fmt.Errorf("discriminated value is used twice: %v", d))
		}
		discriminated[d] = struct{}{}
	}
	// Case 3: a field is both discriminator AND part of the union
	if u.discriminator != "" {
		if _, ok := u.fieldsToDiscriminated[u.discriminator]; ok {
			errors = append(errors, fmt.Errorf("%v can't be both discriminator and part of the union", u.discriminator))
		}
	}
	return errors
}

// Find unions either directly on the members (or inlined members, not
// going across types) or on the type itself, or on embedded types.
func parseUnions(t *types.Type) ([]union, []error) {
	errors := []error{}
	unions := []union{}
	su, err := parseUnionStruct(t)
	if su != nil {
		unions = append(unions, *su)
	}
	errors = append(errors, err...)
	eu, err := parseEmbeddedUnion(t)
	unions = append(unions, eu...)
	errors = append(errors, err...)
	mu, err := parseUnionMembers(t)
	if mu != nil {
		unions = append(unions, *mu)
	}
	errors = append(errors, err...)
	return unions, errors
}

// Find unions in embedded types, unions shouldn't go across types.
func parseEmbeddedUnion(t *types.Type) ([]union, []error) {
	errors := []error{}
	unions := []union{}
	for _, m := range t.Members {
		if hasOpenAPITagValue(m.CommentLines, tagValueFalse) {
			continue
		}
		if !shouldInlineMembers(&m) {
			continue
		}
		u, err := parseUnions(m.Type)
		unions = append(unions, u...)
		errors = append(errors, err...)
	}
	return unions, errors
}

// Look for union tag on a struct, and then include all the fields
// (except the discriminator if there is one). The struct shouldn't have
// embedded types.
func parseUnionStruct(t *types.Type) (*union, []error) {
	errors := []error{}
	if types.ExtractCommentTags("+", t.CommentLines)[tagUnionMember] == nil {
		return nil, nil
	}

	u := &union{fieldsToDiscriminated: map[string]string{}}

	for _, m := range t.Members {
		jsonName := getReferableName(&m)
		if jsonName == "" {
			continue
		}
		if shouldInlineMembers(&m) {
			errors = append(errors, fmt.Errorf("union structures can't have embedded fields: %v.%v", t.Name, m.Name))
			continue
		}
		if types.ExtractCommentTags("+", m.CommentLines)[tagUnionDeprecated] != nil {
			errors = append(errors, fmt.Errorf("union struct can't have unionDeprecated members: %v.%v", t.Name, m.Name))
			continue
		}
		if types.ExtractCommentTags("+", m.CommentLines)[tagUnionDiscriminator] != nil {
			errors = append(errors, u.setDiscriminator(jsonName)...)
		} else {
			if !hasOptionalTag(&m) {
				errors = append(errors, fmt.Errorf("union members must be optional: %v.%v", t.Name, m.Name))
			}
			u.addMember(jsonName, m.Name)
		}
	}

	return u, errors
}

// Find unions specifically on members.
func parseUnionMembers(t *types.Type) (*union, []error) {
	errors := []error{}
	u := &union{fieldsToDiscriminated: map[string]string{}}

	for _, m := range t.Members {
		jsonName := getReferableName(&m)
		if jsonName == "" {
			continue
		}
		if shouldInlineMembers(&m) {
			continue
		}
		if types.ExtractCommentTags("+", m.CommentLines)[tagUnionDiscriminator] != nil {
			errors = append(errors, u.setDiscriminator(jsonName)...)
		}
		if types.ExtractCommentTags("+", m.CommentLines)[tagUnionMember] != nil {
			errors = append(errors, fmt.Errorf("union tag is not accepted on struct members: %v.%v", t.Name, m.Name))
			continue
		}
		if types.ExtractCommentTags("+", m.CommentLines)[tagUnionDeprecated] != nil {
			if !hasOptionalTag(&m) {
				errors = append(errors, fmt.Errorf("union members must be optional: %v.%v", t.Name, m.Name))
			}
			u.addMember(jsonName, m.Name)
		}
	}
	if len(u.fieldsToDiscriminated) == 0 {
		return nil, nil
	}
	return u, append(errors, u.isValid()...)
}
