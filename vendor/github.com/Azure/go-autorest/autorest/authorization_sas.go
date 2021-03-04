package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"net/http"
	"strings"
)

// SASTokenAuthorizer implements an authorization for SAS Token Authentication
// this can be used for interaction with Blob Storage Endpoints
type SASTokenAuthorizer struct {
	sasToken string
}

// NewSASTokenAuthorizer creates a SASTokenAuthorizer using the given credentials
func NewSASTokenAuthorizer(sasToken string) (*SASTokenAuthorizer, error) {
	if strings.TrimSpace(sasToken) == "" {
		return nil, fmt.Errorf("sasToken cannot be empty")
	}

	token := sasToken
	if strings.HasPrefix(sasToken, "?") {
		token = strings.TrimPrefix(sasToken, "?")
	}

	return &SASTokenAuthorizer{
		sasToken: token,
	}, nil
}

// WithAuthorization returns a PrepareDecorator that adds a shared access signature token to the
// URI's query parameters.  This can be used for the Blob, Queue, and File Services.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/delegate-access-with-shared-access-signature
func (sas *SASTokenAuthorizer) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err != nil {
				return r, err
			}

			if r.URL.RawQuery == "" {
				r.URL.RawQuery = sas.sasToken
			} else if !strings.Contains(r.URL.RawQuery, sas.sasToken) {
				r.URL.RawQuery = fmt.Sprintf("%s&%s", r.URL.RawQuery, sas.sasToken)
			}

			return Prepare(r)
		})
	}
}
