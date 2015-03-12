/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

// unionAuthRequestHandler authenticates requests using a chain of authenticator.Requests
type unionAuthRequestHandler []authenticator.Request

// New returns a request authenticator that validates credentials using a chain of authenticator.Request objects
func New(authRequestHandlers ...authenticator.Request) authenticator.Request {
	return unionAuthRequestHandler(authRequestHandlers)
}

// AuthenticateRequest authenticates the request using a chain of authenticator.Request objects.  The first
// success returns that identity.  Errors are only returned if no matches are found.
func (authHandler unionAuthRequestHandler) AuthenticateRequest(req *http.Request) (user.Info, http.Header, bool, error) {
	var errlist []error
	headerlist := http.Header{}
	for _, currAuthRequestHandler := range authHandler {
		info, headers, ok, err := currAuthRequestHandler.AuthenticateRequest(req)
		for key, valset := range headers {
			for _, val := range valset {
				headerlist.Add(key, val)
			}
		}
		if err != nil {
			errlist = append(errlist, err)
			continue
		}

		if ok {
			return info, headers, true, nil
		}
	}

	return nil, headerlist, false, errors.NewAggregate(errlist)
}
