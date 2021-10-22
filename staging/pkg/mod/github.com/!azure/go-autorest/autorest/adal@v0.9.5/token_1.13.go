// +build go1.13

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

package adal

import (
	"context"
	"net/http"
	"time"
)

func getMSIEndpoint(ctx context.Context, sender Sender) (*http.Response, error) {
	// this cannot fail, the return sig is due to legacy reasons
	msiEndpoint, _ := GetMSIVMEndpoint()
	tempCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	// http.NewRequestWithContext() was added in Go 1.13
	req, _ := http.NewRequestWithContext(tempCtx, http.MethodGet, msiEndpoint, nil)
	q := req.URL.Query()
	q.Add("api-version", msiAPIVersion)
	req.URL.RawQuery = q.Encode()
	return sender.Do(req)
}
