/*
Copyright The Kubernetes Authors.

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

package verify

import (
	"errors"
	"fmt"
	"slices"

	"github.com/coreos/go-oidc/v3/oidc"

	authenticationv1 "k8s.io/api/authentication/v1"
)

type webhookBinding struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}

type webhookPrivateClaims struct {
	Kubernetes struct {
		Pod                            *webhookBinding     `json:"pod"`
		Secret                         *webhookBinding     `json:"secret"`
		Node                           *webhookBinding     `json:"node"`
		ValidatingWebhookConfiguration *webhookBinding     `json:"validatingwebhookconfiguration"`
		MutatingWebhookConfiguration   *webhookBinding     `json:"mutatingwebhookconfiguration"`
		Attestations                   map[string][]string `json:"attestations"`
	} `json:"kubernetes.io"`
}

// TokenGroups checks the Kubernetes claims of an authenticated token.
func TokenGroups(token *oidc.IDToken) ([]string, error) {
	var claims webhookPrivateClaims
	if err := token.Claims(&claims); err != nil {
		return nil, fmt.Errorf("decoding Kubernetes token claims: %w", err)
	}
	vwc, mwc := claims.Kubernetes.ValidatingWebhookConfiguration, claims.Kubernetes.MutatingWebhookConfiguration
	if (vwc == nil) == (mwc == nil) || claims.Kubernetes.Pod != nil || claims.Kubernetes.Secret != nil || claims.Kubernetes.Node != nil {
		return nil, errors.New("token is not bound exclusively to one webhook configuration")
	}
	binding := vwc
	if binding == nil {
		binding = mwc
	}
	if binding.Name == "" || binding.UID == "" {
		return nil, errors.New("webhook configuration binding must include a name and UID")
	}
	groups := claims.Kubernetes.Attestations[authenticationv1.AttestationAdmissionReviewAPIGroups]
	if len(groups) != 1 || groups[0] == "" {
		return nil, fmt.Errorf("token must carry exactly one nonempty %q attestation", authenticationv1.AttestationAdmissionReviewAPIGroups)
	}
	return groups, nil
}

// AllowsAPIGroup checks the token's attestation against the admitted API group.
func AllowsAPIGroup(groups []string, group string) bool {
	return slices.Contains(groups, group) || slices.Contains(groups, "*")
}
