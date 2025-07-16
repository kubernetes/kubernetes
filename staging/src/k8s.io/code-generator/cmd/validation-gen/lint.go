/*
Copyright 2024 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"strings"

	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// linter is a struct that holds the state of the linting process.
// It contains a map of types that have been linted, a list of linting rules,
// and a list of errors that occurred during the linting process.
type linter struct {
	linted map[*types.Type]bool
	rules  []lintRule
	// lintErrors is a list of errors that occurred during the linting process.
	// lintErrors would be in the format:
	// field <field_name>: <lint broken message>
	// type <type_name>: <lint broken message>
	lintErrors []error
}

var defaultRules = []lintRule{
	ruleOptionalAndRequired,
	ruleRequiredAndDefault,
}

func (l *linter) AddError(field, msg string) {
	l.lintErrors = append(l.lintErrors, fmt.Errorf("%s: %s", field, msg))
}

func newLinter(rules ...lintRule) *linter {
	if len(rules) == 0 {
		rules = defaultRules
	}
	return &linter{
		linted:     make(map[*types.Type]bool),
		rules:      rules,
		lintErrors: []error{},
	}
}

func (l *linter) lintType(t *types.Type) error {
	if _, ok := l.linted[t]; ok {
		return nil
	}
	l.linted[t] = true

	if t.CommentLines != nil {
		klog.V(5).Infof("linting type %s", t.Name.String())
		lintErrs, err := l.lintComments(t.CommentLines)
		if err != nil {
			return err
		}
		for _, lintErr := range lintErrs {
			l.AddError("type "+t.Name.String(), lintErr)
		}
	}
	switch t.Kind {
	case types.Alias:
		// Recursively lint the underlying type of the alias.
		if err := l.lintType(t.Underlying); err != nil {
			return err
		}
	case types.Struct:
		// Recursively lint each member of the struct.
		for _, member := range t.Members {
			klog.V(5).Infof("linting comments for field %s of type %s", member.String(), t.Name.String())
			lintErrs, err := l.lintComments(member.CommentLines)
			if err != nil {
				return err
			}
			for _, lintErr := range lintErrs {
				l.AddError("type "+t.Name.String(), lintErr)
			}
			if err := l.lintType(member.Type); err != nil {
				return err
			}
		}
	case types.Slice, types.Array, types.Pointer:
		// Recursively lint the element type of the slice or array.
		if err := l.lintType(t.Elem); err != nil {
			return err
		}
	case types.Map:
		// Recursively lint the key and element types of the map.
		if err := l.lintType(t.Key); err != nil {
			return err
		}
		if err := l.lintType(t.Elem); err != nil {
			return err
		}
	}
	return nil
}

// lintRule is a function that validates a slice of comments.
// It returns a string as an error message if the comments are invalid,
// and an error there is an error happened during the linting process.
type lintRule func(comments []string) (string, error)

// lintComments runs all registered rules on a slice of comments.
func (l *linter) lintComments(comments []string) ([]string, error) {
	var lintErrs []string
	for _, rule := range l.rules {
		if msg, err := rule(comments); err != nil {
			return nil, err
		} else if msg != "" {
			lintErrs = append(lintErrs, msg)
		}
	}

	return lintErrs, nil
}

// conflictingTagsRule checks for conflicting tags in a slice of comments.
func conflictingTagsRule(comments []string, tags ...string) (string, error) {
	if len(tags) < 2 {
		return "", fmt.Errorf("at least two tags must be provided")
	}
	tagCount := make(map[string]bool)
	for _, comment := range comments {
		for _, tag := range tags {
			if strings.HasPrefix(comment, tag) {
				tagCount[tag] = true
			}
		}
	}
	if len(tagCount) > 1 {
		return fmt.Sprintf("conflicting tags: {%s}", strings.Join(tags, ", ")), nil
	}
	return "", nil
}

// ruleOptionalAndRequired checks for conflicting tags +k8s:optional and +k8s:required in a slice of comments.
func ruleOptionalAndRequired(comments []string) (string, error) {
	return conflictingTagsRule(comments, "+k8s:optional", "+k8s:required")
}

// ruleRequiredAndDefault checks for conflicting tags +k8s:required and +default in a slice of comments.
func ruleRequiredAndDefault(comments []string) (string, error) {
	return conflictingTagsRule(comments, "+k8s:required", "+default")
}
