// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// CreateRegistrationToken creates a token that can be used to add a self-hosted runner.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/enterprise-admin/#create-a-registration-token-for-an-enterprise
func (s *EnterpriseService) CreateRegistrationToken(ctx context.Context, enterprise string) (*RegistrationToken, *Response, error) {
	u := fmt.Sprintf("enterprises/%v/actions/runners/registration-token", enterprise)

	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	registrationToken := new(RegistrationToken)
	resp, err := s.client.Do(ctx, req, registrationToken)
	if err != nil {
		return nil, resp, err
	}

	return registrationToken, resp, nil
}

// ListRunners lists all the self-hosted runners for a enterprise.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/enterprise-admin/#list-self-hosted-runners-for-an-enterprise
func (s *EnterpriseService) ListRunners(ctx context.Context, enterprise string, opts *ListOptions) (*Runners, *Response, error) {
	u := fmt.Sprintf("enterprises/%v/actions/runners", enterprise)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	runners := &Runners{}
	resp, err := s.client.Do(ctx, req, &runners)
	if err != nil {
		return nil, resp, err
	}

	return runners, resp, nil
}
