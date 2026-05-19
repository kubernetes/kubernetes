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

package certauthorization

import (
	"context"
	"strings"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/klog/v2"
)

// IsAuthorizedForSignerName returns true if 'info' is authorized to perform the given
// 'verb' on the synthetic 'signers' resource with the given signerName.
// If the user does not have permission to perform the 'verb' on the given signerName,
// it will also perform an authorization check against {domain portion}/*, for example
// `kubernetes.io/*`. This allows an entity to be granted permission to 'verb' on all
// signerNames with a given 'domain portion'.
func IsAuthorizedForSignerName(ctx context.Context, authz authorizer.Authorizer, info user.Info, verb, signerName string) bool {
	// First check if the user has explicit permission to 'verb' for the given signerName.
	attr := buildAttributes(info, verb, signerName)
	decision, reason, err := authz.Authorize(ctx, attr)
	switch {
	case err != nil:
		klog.V(3).Infof("cannot authorize %q %q for policy: %v,%v", verb, attr.GetName(), reason, err)
	case decision == authorizer.DecisionAllow:
		return true
	}

	// If not, check if the user has wildcard permissions to 'verb' for the domain portion of the signerName, e.g.
	// 'kubernetes.io/*'.
	attr = buildWildcardAttributes(info, verb, signerName)
	decision, reason, err = authz.Authorize(ctx, attr)
	switch {
	case err != nil:
		klog.V(3).Infof("cannot authorize %q %q for policy: %v,%v", verb, attr.GetName(), reason, err)
	case decision == authorizer.DecisionAllow:
		return true
	}

	return false
}

func buildAttributes(info user.Info, verb, signerName string) authorizer.Attributes {
	return authorizer.AttributesRecord{
		User:            info,
		Verb:            verb,
		Name:            signerName,
		APIGroup:        "certificates.k8s.io",
		APIVersion:      "*",
		Resource:        "signers",
		ResourceRequest: true,
	}
}

func buildWildcardAttributes(info user.Info, verb, signerName string) authorizer.Attributes {
	parts := strings.Split(signerName, "/")
	domain := parts[0]
	return buildAttributes(info, verb, domain+"/*")
}
