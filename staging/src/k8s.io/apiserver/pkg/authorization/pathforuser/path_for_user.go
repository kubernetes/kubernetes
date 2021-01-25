/*
Copyright 2020 The Kubernetes Authors.

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

package pathforuser

import (
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/path"
)

// NewAuthorizer returns an authorizer which accepts a given set of paths.
// Each path is either a fully matching path or it ends in * in case a prefix match is done. A leading / is optional.
func NewAuthorizer(userStrings []string, alwaysAllowPaths []string) (authorizer.Authorizer, error) {
	users := sets.NewString(userStrings...)
	pathMatcher, err := path.NewPrefixMatcher(alwaysAllowPaths)
	if err != nil {
		return nil, err
	}

	var prefixes []string
	paths := sets.NewString()
	for _, p := range alwaysAllowPaths {
		p = strings.TrimPrefix(p, "/")
		if len(p) == 0 {
			// matches "/"
			paths.Insert(p)
			continue
		}
		if strings.ContainsRune(p[:len(p)-1], '*') {
			return nil, fmt.Errorf("only trailing * allowed in %q", p)
		}
		if strings.HasSuffix(p, "*") {
			prefixes = append(prefixes, p[:len(p)-1])
		} else {
			paths.Insert(p)
		}
	}

	return authorizer.AuthorizerFunc(func(attr authorizer.Attributes) (authorizer.Decision, string, error) {
		if attr.IsResourceRequest() {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if attr.GetUser() == nil {
			return authorizer.DecisionNoOpinion, "Error", errors.New("no user on request.")
		}

		// if the request matches our set of users and the paths, then allow the action
		if users.Has(attr.GetUser().GetName()) && pathMatcher.Matches(attr.GetPath()) {
			return authorizer.DecisionAllow, "", nil
		}

		// otherwise this authorizer has no opinion
		return authorizer.DecisionNoOpinion, "", nil
	}), nil
}
