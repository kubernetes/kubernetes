// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is handwritten code. These methods are implemented by hand so they can use
// the iam.Policy type.

package admin

import (
	"context"

	"cloud.google.com/go/iam"
	iampb "google.golang.org/genproto/googleapis/iam/v1"
)

// GetIamPolicy returns the IAM access control policy for a ServiceAccount.
func (c *IamClient) GetIamPolicy(ctx context.Context, req *iampb.GetIamPolicyRequest) (*iam.Policy, error) {
	policy, err := c.getIamPolicy(ctx, req)
	if err != nil {
		return nil, err
	}
	return &iam.Policy{InternalProto: policy}, nil
}

// SetIamPolicyRequest is the request type for the SetIamPolicy method.
type SetIamPolicyRequest struct {
	Resource string
	Policy   *iam.Policy
}

// SetIamPolicy sets the IAM access control policy for a ServiceAccount.
func (c *IamClient) SetIamPolicy(ctx context.Context, req *SetIamPolicyRequest) (*iam.Policy, error) {
	preq := &iampb.SetIamPolicyRequest{
		Resource: req.Resource,
		Policy:   req.Policy.InternalProto,
	}
	policy, err := c.setIamPolicy(ctx, preq)
	if err != nil {
		return nil, err
	}
	return &iam.Policy{InternalProto: policy}, nil
}
