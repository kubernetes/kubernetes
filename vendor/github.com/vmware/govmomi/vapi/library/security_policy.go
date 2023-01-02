/*
Copyright (c) 2022-2022 VMware, Inc. All Rights Reserved.

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

package library

import (
	"context"
	"errors"
	"net/http"

	"github.com/vmware/govmomi/vapi/internal"
)

const (
	OvfDefaultSecurityPolicy = "OVF default policy"
)

// ContentSecurityPoliciesInfo contains information on security policies that can
// be used to describe security for content library items.
type ContentSecurityPoliciesInfo struct {
	// ItemTypeRules are rules governing the policy.
	ItemTypeRules map[string]string `json:"item_type_rules"`
	// Name is a human-readable identifier identifying the policy.
	Name string `json:"name"`
	// Policy is the unique identifier for a policy.
	Policy string `json:"policy"`
}

// ListSecurityPolicies lists security policies
func (c *Manager) ListSecurityPolicies(ctx context.Context) ([]ContentSecurityPoliciesInfo, error) {
	url := c.Resource(internal.SecurityPoliciesPath)
	var res []ContentSecurityPoliciesInfo
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

func (c *Manager) DefaultOvfSecurityPolicy(ctx context.Context) (string, error) {
	res, err := c.ListSecurityPolicies(ctx)

	if err != nil {
		return "", err
	}

	for _, policy := range res {
		if policy.Name == OvfDefaultSecurityPolicy {
			return policy.Policy, nil
		}
	}

	return "", errors.New("failed to find default ovf security policy")
}
