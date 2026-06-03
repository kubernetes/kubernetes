/*
Copyright 2018 The Kubernetes Authors.

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

package path

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// NewAuthorizer returns an authorizer which accepts a given set of paths.
// Each path is either a fully matching path or it ends in * in case a prefix match is done. A leading / is optional.
func NewAuthorizer(alwaysAllowPaths []string) (authorizer.Authorizer, error) {
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

	return authorizer.AuthorizerFunc(func(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
		if a.IsResourceRequest() {
			return authorizer.DecisionNoOpinion, "", nil
		}

		pth := strings.TrimPrefix(a.GetPath(), "/")
		if paths.Has(pth) {
			return authorizer.DecisionAllow, "", nil
		}

		for _, prefix := range prefixes {
			if strings.HasPrefix(pth, prefix) {
				return authorizer.DecisionAllow, "", nil
			}
		}

		return authorizer.DecisionNoOpinion, "", nil
	}), nil
}
