/*
Copyright 2018 The Kubernetes Authors.

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

package handlers

import (
	"net/http"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/types"
)

var uar = regexp.MustCompile(`\w+`)

// parseUserAgent only returns the first word found in the userAgent. If
// no word can be found, "default" is returned.
func parseUserAgent(userAgent string) string {
	m := uar.FindString(userAgent)
	if m == "" {
		return "default"
	}
	return m
}

// buildWorkflowID builds a reasonable default for the workflowID, based
// on user-agent of the incoming request, sub-resource, verb and
// content-type. Always returns a non-empty string.
// TODO(apelisse): We need a way to clean-up obsolete workflow and/or
// override defaulted workflow with specific workflow.
// TODO(apelisse): We also probably need to understand how changing this
// heuristic would impact object that use former values.
// TODO(apelisse): And we'll need a validate function for workflowID
func buildDefaultWorkflowID(subResource, verb, contentType, userAgent string) string {
	workflowID := parseUserAgent(userAgent)

	if subResource != "" {
		workflowID = workflowID + "-" + subResource
	}

	// We only append the verb if it's not a POST or if it's not an
	// apply PATCH.
	switch verb {
	case http.MethodPost:
	case http.MethodPatch:
		if types.PatchType(contentType) == types.ApplyPatchType {
			break
		}
		fallthrough
	default:
		workflowID = workflowID + "-" + verb
	}

	return strings.ToLower(workflowID)
}
