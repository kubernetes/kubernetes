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

package signing

import (
	"context"
	"errors"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	certificatesapi "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/features"
)

func TestPlugin_Validate(t *testing.T) {
	tests := map[string]struct {
		attributes        admission.Attributes
		pcrFeatureEnabled bool
		allowedName       string
		allowed           bool
		authzErr          error
	}{
		"wrong type": {
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj:      &certificatesapi.CertificateSigningRequestList{},
				obj:         &certificatesapi.CertificateSigningRequestList{},
				operation:   admission.Update,
			},
			allowed: false,
		},
		"allowed if the 'certificate' and conditions field has not changed": {
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Status: certificatesapi.CertificateSigningRequestStatus{
					Certificate: []byte("data"),
				}},
				obj: &certificatesapi.CertificateSigningRequest{Status: certificatesapi.CertificateSigningRequestStatus{
					Certificate: []byte("data"),
				}},
				operation: admission.Update,
			},
			allowed:  true,
			authzErr: errors.New("faked error"),
		},
		"deny request if authz lookup fails on certificate change": {
			allowedName: "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Certificate: []byte("data"),
					},
				},
				operation: admission.Update,
			},
			authzErr: errors.New("test"),
			allowed:  false,
		},
		"deny request if authz lookup fails on condition change": {
			allowedName: "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Conditions: []certificatesapi.CertificateSigningRequestCondition{{Type: certificatesapi.CertificateFailed}},
					},
				},
				operation: admission.Update,
			},
			authzErr: errors.New("test"),
			allowed:  false,
		},
		"allow request if user is authorized for specific signerName": {
			allowedName: "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Certificate: []byte("data"),
					},
				},
				operation: admission.Update,
			},
			allowed: true,
		},
		"allow request if user is authorized with wildcard": {
			allowedName: "abc.com/*",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Certificate: []byte("data"),
					},
				},
				operation: admission.Update,
			},
			allowed: true,
		},
		"should deny request if user does not have permission for this signerName": {
			allowedName: "notabc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Certificate: []byte("data"),
					},
				},
				operation: admission.Update,
			},
			allowed: false,
		},
		"should deny request if user attempts to update signerName to a new value they *do* have permission to sign for": {
			allowedName: "allowed.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("certificatesigningrequests"),
				subresource: "status",
				oldObj: &certificatesapi.CertificateSigningRequest{Spec: certificatesapi.CertificateSigningRequestSpec{
					SignerName: "notallowed.com/xyz",
				}},
				obj: &certificatesapi.CertificateSigningRequest{
					Spec: certificatesapi.CertificateSigningRequestSpec{
						SignerName: "allowed.com/xyz",
					},
					Status: certificatesapi.CertificateSigningRequestStatus{
						Certificate: []byte("data"),
					},
				},
				operation: admission.Update,
			},
			allowed: false,
		},
		"pcr signing allowed if feature gate disabled": {
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj:      &certificatesapi.PodCertificateRequest{Status: certificatesapi.PodCertificateRequestStatus{}},
				obj: &certificatesapi.PodCertificateRequest{Status: certificatesapi.PodCertificateRequestStatus{
					CertificateChain: "data",
				}},
				operation: admission.Update,
			},
			allowed: true,
		},
		"pcr wrong type": {
			pcrFeatureEnabled: true,
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj:      &certificatesapi.CertificateSigningRequestList{},
				obj:         &certificatesapi.CertificateSigningRequestList{},
				operation:   admission.Update,
			},
			allowed: false,
		},
		"pcr update allowed if the 'certificateChain' and conditions field has not changed": {
			pcrFeatureEnabled: true,
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{Status: certificatesapi.PodCertificateRequestStatus{
					CertificateChain: "data",
				}},
				obj: &certificatesapi.PodCertificateRequest{Status: certificatesapi.PodCertificateRequestStatus{
					CertificateChain: "data",
				}},
				operation: admission.Update,
			},
			allowed:  true,
			authzErr: errors.New("faked error"),
		},
		"pcr deny request if authz lookup fails on certificate change": {
			pcrFeatureEnabled: true,
			allowedName:       "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{Spec: certificatesapi.PodCertificateRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						CertificateChain: "data",
					},
				},
				operation: admission.Update,
			},
			authzErr: errors.New("test"),
			allowed:  false,
		},
		"pcr deny request if authz lookup fails on condition change": {
			pcrFeatureEnabled: true,
			allowedName:       "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					}},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						Conditions: []metav1.Condition{
							{Type: certificatesapi.PodCertificateRequestConditionTypeFailed},
						},
					},
				},
				operation: admission.Update,
			},
			authzErr: errors.New("test"),
			allowed:  false,
		},
		"pcr allow request if user is authorized for specific signerName": {
			pcrFeatureEnabled: true,
			allowedName:       "abc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{Spec: certificatesapi.PodCertificateRequestSpec{
					SignerName: "abc.com/xyz",
				}},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						CertificateChain: "data",
					},
				},
				operation: admission.Update,
			},
			allowed: true,
		},
		"pcr allow request if user is authorized with wildcard": {
			pcrFeatureEnabled: true,
			allowedName:       "abc.com/*",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
				},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						CertificateChain: "data",
					},
				},
				operation: admission.Update,
			},
			allowed: true,
		},
		"pcrshould deny request if user does not have permission for this signerName": {
			pcrFeatureEnabled: true,
			allowedName:       "notabc.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
				},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "abc.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						CertificateChain: "data",
					},
				},
				operation: admission.Update,
			},
			allowed: false,
		},
		"pcr should deny request if user attempts to update signerName to a new value they *do* have permission to sign for": {
			pcrFeatureEnabled: true,
			allowedName:       "allowed.com/xyz",
			attributes: &testAttributes{
				resource:    certificatesapi.Resource("podcertificaterequests"),
				subresource: "status",
				oldObj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "notallowed.com/xyz",
					},
				},
				obj: &certificatesapi.PodCertificateRequest{
					Spec: certificatesapi.PodCertificateRequestSpec{
						SignerName: "allowed.com/xyz",
					},
					Status: certificatesapi.PodCertificateRequestStatus{
						CertificateChain: "data",
					},
				},
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
					verb:        "sign",
					allowedName: test.allowedName,
					decision:    authorizer.DecisionAllow,
					err:         test.authzErr,
				},
			}

			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.PodCertificateRequest, test.pcrFeatureEnabled)
			p.InspectFeatureGates(feature.DefaultFeatureGate)

			err := p.Validate(context.Background(), test.attributes, nil)
			if err == nil && !test.allowed {
				t.Errorf("Expected authorization policy to reject CSR/PSR but it was allowed")
			}
			if err != nil && test.allowed {
				t.Errorf("Expected authorization policy to accept CSR/PSR but it was rejected: %v", err)
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
	oldObj, obj runtime.Object
	name        string

	admission.Attributes // nil panic if any other methods called
}

func (t *testAttributes) GetResource() schema.GroupVersionResource {
	return t.resource.WithVersion("ignored")
}

func (t *testAttributes) GetSubresource() string {
	return t.subresource
}

func (t *testAttributes) GetOldObject() runtime.Object {
	return t.oldObj
}

func (t *testAttributes) GetObject() runtime.Object {
	return t.obj
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
