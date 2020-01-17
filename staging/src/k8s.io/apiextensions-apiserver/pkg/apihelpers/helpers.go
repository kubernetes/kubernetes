/*
Copyright 2019 The Kubernetes Authors.

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

package apihelpers

import (
	"fmt"
	"net/url"
	"strings"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

// IsProtectedCommunityGroup returns whether or not a group specified for a CRD is protected for the community and needs
// to have the v1beta1.KubeAPIApprovalAnnotation set.
func IsProtectedCommunityGroup(group string) bool {
	switch {
	case group == "k8s.io" || strings.HasSuffix(group, ".k8s.io"):
		return true
	case group == "kubernetes.io" || strings.HasSuffix(group, ".kubernetes.io"):
		return true
	default:
		return false
	}

}

// APIApprovalState covers the various options for API approval annotation states
type APIApprovalState int

const (
	// APIApprovalInvalid means the annotation doesn't have an expected value
	APIApprovalInvalid APIApprovalState = iota
	// APIApproved if the annotation has a URL (this means the API is approved)
	APIApproved
	// APIApprovalBypassed if the annotation starts with "unapproved" indicating that for whatever reason the API isn't approved, but we should allow its creation
	APIApprovalBypassed
	// APIApprovalMissing means the annotation is empty
	APIApprovalMissing
)

// GetAPIApprovalState returns the state of the API approval and reason for that state
func GetAPIApprovalState(annotations map[string]string) (state APIApprovalState, reason string) {
	annotation := annotations[v1beta1.KubeAPIApprovedAnnotation]

	// we use the result of this parsing in the switch/case below
	url, annotationURLParseErr := url.ParseRequestURI(annotation)
	switch {
	case len(annotation) == 0:
		return APIApprovalMissing, fmt.Sprintf("protected groups must have approval annotation %q, see https://github.com/kubernetes/enhancements/pull/1111", v1beta1.KubeAPIApprovedAnnotation)
	case strings.HasPrefix(annotation, "unapproved"):
		return APIApprovalBypassed, fmt.Sprintf("not approved: %q", annotation)
	case annotationURLParseErr == nil && url != nil && len(url.Host) > 0 && len(url.Scheme) > 0:
		return APIApproved, fmt.Sprintf("approved in %v", annotation)
	default:
		return APIApprovalInvalid, fmt.Sprintf("protected groups must have approval annotation %q with either a URL or a reason starting with \"unapproved\", see https://github.com/kubernetes/enhancements/pull/1111", v1beta1.KubeAPIApprovedAnnotation)
	}
}
