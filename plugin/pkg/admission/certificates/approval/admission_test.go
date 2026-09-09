/*
Copyright 2020 The Kubernetes Authors.

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

package approval

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"

	certificatesapi "k8s.io/kubernetes/pkg/apis/certificates"
)

func TestPlugin_Validate(t *testing.T) {
	tests := map[string]struct {
		attributes  admission.Attributes
		allowedName string
		allowed     bool
		authzErr    error
	}{
		"wrong type": {
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj:      &certificatesapi.CertificateSigningRequestList{},
				operation:   admission.Update,
			},
			allowed: false,
		},
		"reject requests if looking up permissions fails": {
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				operation: admission.Update,
			},
			authzErr: errors.New("forced error"),
			allowed:  false,
		},
		"should allow request if user is authorized for specific signerName": {
			allowedName: "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				operation: admission.Update,
			},
			allowed: true,
		},
		"should allow request if user is authorized with wildcard": {
			allowedName: "abc.com/*",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				operation: admission.Update,
			},
			allowed: true,
		},
		"should deny request if user does not have permission for this signerName": {
			allowedName: "notabc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				operation: admission.Update,
			},
			allowed: false,
		},
		"should deny request if user attempts to update signerName to a new value they *do* have permission to approve for": {
			allowedName: "allowed.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "approval",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "notallowed.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "allowed.com/xyz",
				}},
				operation: admission.Update,
			},
			allowed: false,
		},
	}

	for n, test := range tests {
		t.Run(n, func(t *testing.T) {
			p := Plugin{
				authz: fakeAuthorizer{
					t:           t,
					verb:        "approve",
					allowedName: test.allowedName,
					decision:    authorizer.DecisionAllow,
					err:         test.authzErr,
				},
			}
			err := p.Validate(context.Background(), test.attributes, nil)
			if err == nil && !test.allowed {
				t.Errorf("Expected authorization policy to reject CSR but it was allowed")
			}
			if err != nil && test.allowed {
				t.Errorf("Expected authorization policy to accept CSR but it was rejected: %v", err)
			}
		})
	}
}

type fakeAuthorizer struct {
	t           *testing.T
	verb        string
	allowedName string
	decision    authorizer.Decision
	err         error
}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	if f.err != nil {
		return f.decision, "forced error", f.err
	}
	if a.GetVerb() != f.verb {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised verb '%s'", a.GetVerb()), nil
	}
	if a.GetAPIGroup() != "certificates.k8s.io" {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised groupName '%s'", a.GetAPIGroup()), nil
	}
	if a.GetAPIVersion() != "*" {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised apiVersion '%s'", a.GetAPIVersion()), nil
	}
	if a.GetResource() != "signers" {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised resource '%s'", a.GetResource()), nil
	}
	if a.GetName() != f.allowedName {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised resource name '%s'", a.GetName()), nil
	}
	if !a.IsResourceRequest() {
		return authorizer.DecisionDeny, fmt.Sprintf("unrecognised IsResourceRequest '%t'", a.IsResourceRequest()), nil
	}
	return f.decision, "", nil
}

type testAttributes struct {
	resource    schema.GroupResource
	subresource string
	operation   admission.Operation
	obj, oldObj runtime.Object
	name        string

	admission.Attributes // nil panic if any other methods called
}

func (t *testAttributes) GetResource() schema.GroupVersionResource {
	return t.resource.WithVersion("ignored")
}

func (t *testAttributes) GetSubresource() string {
	return t.subresource
}

func (t *testAttributes) GetObject() runtime.Object {
	return t.obj
}

func (t *testAttributes) GetOldObject() runtime.Object {
	return t.oldObj
}

func (t *testAttributes) GetName() string {
	return t.name
}

func (t *testAttributes) GetOperation() admission.Operation {
	return t.operation
}

func (t *testAttributes) GetUserInfo() user.Info {
	return &user.DefaultInfo{Name: "ignored"}
}
