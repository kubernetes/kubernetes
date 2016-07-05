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
	"net/http"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// unionAuthRequestHandler authenticates requests using a chain of authenticator.Requests
type unionAuthRequestHandler []authenticator.Request

// New returns a request authenticator that validates credentials using a chain of authenticator.Request objects
func New(authRequestHandlers ...authenticator.Request) authenticator.Request {
	return unionAuthRequestHandler(authRequestHandlers)
}

// AuthenticateRequest authenticates the request using a chain of authenticator.Request objects.  The first
// success returns that identity.  Errors are only returned if no matches are found.
func (authHandler unionAuthRequestHandler) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	var errlist []error
	for _, currAuthRequestHandler := range authHandler {
		info, ok, err := currAuthRequestHandler.AuthenticateRequest(req)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}

		if ok {
			return info, true, nil
		}
	}

	return nil, false, utilerrors.NewAggregate(errlist)
}
