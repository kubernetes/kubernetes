/*
Copyright 2014 The Kubernetes Authors.

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

// Package abac authorizes Kubernetes API actions using an Attribute-based access control scheme.
package abac

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/apis/abac"

	// Import latest API for init/side-effects
	_ "k8s.io/kubernetes/pkg/apis/abac/latest"
	"k8s.io/kubernetes/pkg/apis/abac/v0"
)

type policyLoadError struct {
	path string
	line int
	data []byte
	err  error
}

func (p policyLoadError) Error() string {
	if p.line >= 0 {
		return fmt.Sprintf("error reading policy file %s, line %d: %s: %v", p.path, p.line, string(p.data), p.err)
	}
	return fmt.Sprintf("error reading policy file %s: %v", p.path, p.err)
}

// PolicyList is simply a slice of Policy structs.
type PolicyList []*abac.Policy

// NewFromFile attempts to create a policy list from the given file.
//
// TODO: Have policies be created via an API call and stored in REST storage.
func NewFromFile(path string) (PolicyList, error) {
	// File format is one map per line.  This allows easy concatenation of files,
	// comments in files, and identification of errors by line number.
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	pl := make(PolicyList, 0)

	decoder := abac.Codecs.UniversalDecoder()

	i := 0
	unversionedLines := 0
	for scanner.Scan() {
		i++
		p := &abac.Policy{}
		b := scanner.Bytes()

		// skip comment lines and blank lines
		trimmed := strings.TrimSpace(string(b))
		if len(trimmed) == 0 || strings.HasPrefix(trimmed, "#") {
			continue
		}

		decodedObj, _, err := decoder.Decode(b, nil, nil)
		if err != nil {
			if !(runtime.IsMissingVersion(err) || runtime.IsMissingKind(err) || runtime.IsNotRegisteredError(err)) {
				return nil, policyLoadError{path, i, b, err}
			}
			unversionedLines++
			// Migrate unversioned policy object
			oldPolicy := &v0.Policy{}
			if err := runtime.DecodeInto(decoder, b, oldPolicy); err != nil {
				return nil, policyLoadError{path, i, b, err}
			}
			if err := abac.Scheme.Convert(oldPolicy, p, nil); err != nil {
				return nil, policyLoadError{path, i, b, err}
			}
			pl = append(pl, p)
			continue
		}

		decodedPolicy, ok := decodedObj.(*abac.Policy)
		if !ok {
			return nil, policyLoadError{path, i, b, fmt.Errorf("unrecognized object: %#v", decodedObj)}
		}
		pl = append(pl, decodedPolicy)
	}

	if unversionedLines > 0 {
		klog.Warningf("Policy file %s contained unversioned rules. See docs/admin/authorization.md#abac-mode for ABAC file format details.", path)
	}

	if err := scanner.Err(); err != nil {
		return nil, policyLoadError{path, -1, nil, err}
	}
	return pl, nil
}

func matches(p abac.Policy, a authorizer.Attributes) bool {
	if subjectMatches(p, a.GetUser()) {
		if verbMatches(p, a) {
			// Resource and non-resource requests are mutually exclusive, at most one will match a policy
			if resourceMatches(p, a) {
				return true
			}
			if nonResourceMatches(p, a) {
				return true
			}
		}
	}
	return false
}

// subjectMatches returns true if specified user and group properties in the policy match the attributes
func subjectMatches(p abac.Policy, user user.Info) bool {
	matched := false

	if user == nil {
		return false
	}
	username := user.GetName()
	groups := user.GetGroups()

	// If the policy specified a user, ensure it matches
	if len(p.Spec.User) > 0 {
		if p.Spec.User == "*" {
			matched = true
		} else {
			matched = p.Spec.User == username
			if !matched {
				return false
			}
		}
	}

	// If the policy specified a group, ensure it matches
	if len(p.Spec.Group) > 0 {
		if p.Spec.Group == "*" {
			matched = true
		} else {
			matched = false
			for _, group := range groups {
				if p.Spec.Group == group {
					matched = true
				}
			}
			if !matched {
				return false
			}
		}
	}

	return matched
}

func verbMatches(p abac.Policy, a authorizer.Attributes) bool {
	// TODO: match on verb

	// All policies allow read only requests
	if a.IsReadOnly() {
		return true
	}

	// Allow if policy is not readonly
	if !p.Spec.Readonly {
		return true
	}

	return false
}

func nonResourceMatches(p abac.Policy, a authorizer.Attributes) bool {
	// A non-resource policy cannot match a resource request
	if !a.IsResourceRequest() {
		// Allow wildcard match
		if p.Spec.NonResourcePath == "*" {
			return true
		}
		// Allow exact match
		if p.Spec.NonResourcePath == a.GetPath() {
			return true
		}
		// Allow a trailing * subpath match
		if strings.HasSuffix(p.Spec.NonResourcePath, "*") && strings.HasPrefix(a.GetPath(), strings.TrimRight(p.Spec.NonResourcePath, "*")) {
			return true
		}
	}
	return false
}

func resourceMatches(p abac.Policy, a authorizer.Attributes) bool {
	// A resource policy cannot match a non-resource request
	if a.IsResourceRequest() {
		if p.Spec.Namespace == "*" || p.Spec.Namespace == a.GetNamespace() {
			if p.Spec.Resource == "*" || p.Spec.Resource == a.GetResource() {
				if p.Spec.APIGroup == "*" || p.Spec.APIGroup == a.GetAPIGroup() {
					return true
				}
			}
		}
	}
	return false
}

// Authorize implements authorizer.Authorize
func (pl PolicyList) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	for _, p := range pl {
		if matches(*p, a) {
			return authorizer.DecisionAllow, "", nil
		}
	}
	return authorizer.DecisionNoOpinion, "No policy matched.", nil
	// TODO: Benchmark how much time policy matching takes with a medium size
	// policy file, compared to other steps such as encoding/decoding.
	// Then, add Caching only if needed.
}

// RulesFor returns rules for the given user and namespace.
func (pl PolicyList) RulesFor(user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	var (
		resourceRules    []authorizer.ResourceRuleInfo
		nonResourceRules []authorizer.NonResourceRuleInfo
	)

	for _, p := range pl {
		if subjectMatches(*p, user) {
			if p.Spec.Namespace == "*" || p.Spec.Namespace == namespace {
				if len(p.Spec.Resource) > 0 {
					r := authorizer.DefaultResourceRuleInfo{
						Verbs:     getVerbs(p.Spec.Readonly),
						APIGroups: []string{p.Spec.APIGroup},
						Resources: []string{p.Spec.Resource},
					}
					var resourceRule authorizer.ResourceRuleInfo = &r
					resourceRules = append(resourceRules, resourceRule)
				}
				if len(p.Spec.NonResourcePath) > 0 {
					r := authorizer.DefaultNonResourceRuleInfo{
						Verbs:           getVerbs(p.Spec.Readonly),
						NonResourceURLs: []string{p.Spec.NonResourcePath},
					}
					var nonResourceRule authorizer.NonResourceRuleInfo = &r
					nonResourceRules = append(nonResourceRules, nonResourceRule)
				}
			}
		}
	}
	return resourceRules, nonResourceRules, false, nil
}

func getVerbs(isReadOnly bool) []string {
	if isReadOnly {
		return []string{"get", "list", "watch"}
	}
	return []string{"*"}
}
