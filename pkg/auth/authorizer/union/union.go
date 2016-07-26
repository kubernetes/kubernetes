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

package union

import (
	"strings"

	"k8s.io/kubernetes/pkg/auth/authorizer"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// unionAuthzHandler authorizer against a chain of authorizer.Authorizer
type unionAuthzHandler []authorizer.Authorizer

// New returns an authorizer that authorizes against a chain of authorizer.Authorizer objects
func New(authorizationHandlers ...authorizer.Authorizer) authorizer.Authorizer {
	return unionAuthzHandler(authorizationHandlers)
}

// Authorizes against a chain of authorizer.Authorizer objects and returns nil if successful and returns error if unsuccessful
func (authzHandler unionAuthzHandler) Authorize(a authorizer.Attributes) (bool, string, error) {
	var (
		errlist    []error
		reasonlist []string
	)
	for _, currAuthzHandler := range authzHandler {
		authorized, reason, err := currAuthzHandler.Authorize(a)

		if err != nil {
			errlist = append(errlist, err)
			continue
		}
		if !authorized {
			if reason != "" {
				reasonlist = append(reasonlist, reason)
			}
			continue
		}
		return true, reason, nil
	}

	return false, strings.Join(reasonlist, "\n"), utilerrors.NewAggregate(errlist)
}
