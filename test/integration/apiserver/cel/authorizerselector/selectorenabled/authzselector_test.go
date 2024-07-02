/*
Copyright 2024 The Kubernetes Authors.

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

package selectorenabled

import (
	"testing"

	"k8s.io/kubernetes/test/integration/apiserver/cel/authorizerselector"
)

// TestAuthzSelectorsLibraryEnabled ensures that the authzselectors library feature enablement works properly.
// CEL envs and compilers cached per process mean this must be the only test in this package.
func TestAuthzSelectorsLibraryEnabled(t *testing.T) {
	authorizerselector.RunAuthzSelectorsLibraryTests(t, true)
}
