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
	"fmt"
	"net/http"
	"time"
)

func getMSIEndpoint(ctx context.Context, sender Sender) (*http.Response, error) {
	tempCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	// http.NewRequestWithContext() was added in Go 1.13
	req, _ := http.NewRequestWithContext(tempCtx, http.MethodGet, msiEndpoint, nil)
	q := req.URL.Query()
	q.Add("api-version", msiAPIVersion)
	req.URL.RawQuery = q.Encode()
	return sender.Do(req)
}

// EnsureFreshWithContext will refresh the token if it will expire within the refresh window (as set by
// RefreshWithin) and autoRefresh flag is on.  This method is safe for concurrent use.
func (mt *MultiTenantServicePrincipalToken) EnsureFreshWithContext(ctx context.Context) error {
	if err := mt.PrimaryToken.EnsureFreshWithContext(ctx); err != nil {
		return fmt.Errorf("failed to refresh primary token: %w", err)
	}
	for _, aux := range mt.AuxiliaryTokens {
		if err := aux.EnsureFreshWithContext(ctx); err != nil {
			return fmt.Errorf("failed to refresh auxiliary token: %w", err)
		}
	}
	return nil
}

// RefreshWithContext obtains a fresh token for the Service Principal.
func (mt *MultiTenantServicePrincipalToken) RefreshWithContext(ctx context.Context) error {
	if err := mt.PrimaryToken.RefreshWithContext(ctx); err != nil {
		return fmt.Errorf("failed to refresh primary token: %w", err)
	}
	for _, aux := range mt.AuxiliaryTokens {
		if err := aux.RefreshWithContext(ctx); err != nil {
			return fmt.Errorf("failed to refresh auxiliary token: %w", err)
		}
	}
	return nil
}

// RefreshExchangeWithContext refreshes the token, but for a different resource.
func (mt *MultiTenantServicePrincipalToken) RefreshExchangeWithContext(ctx context.Context, resource string) error {
	if err := mt.PrimaryToken.RefreshExchangeWithContext(ctx, resource); err != nil {
		return fmt.Errorf("failed to refresh primary token: %w", err)
	}
	for _, aux := range mt.AuxiliaryTokens {
		if err := aux.RefreshExchangeWithContext(ctx, resource); err != nil {
			return fmt.Errorf("failed to refresh auxiliary token: %w", err)
		}
	}
	return nil
}
