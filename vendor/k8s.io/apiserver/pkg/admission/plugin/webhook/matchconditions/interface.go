/*
Copyright 2023 The Kubernetes Authors.

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

package matchconditions

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
)

type MatchResult struct {
	Matches             bool
	Error               error
	FailedConditionName string
}

// Matcher contains logic for converting Evaluations to bool of matches or does not match
type Matcher interface {
	// Match is used to take cel evaluations and convert into decisions
	Match(ctx context.Context, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object) MatchResult
}
