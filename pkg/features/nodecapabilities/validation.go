/*
Copyright 2025 The Kubernetes Authors.

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

package nodecapabilities

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation"
)

// ValidateCapability validates the key and value of a node capability.
func ValidateCapability(k, v string) error {
	// All capability keys must have the "compatibility.kubernetes.io/" prefix.
	if !strings.HasPrefix(k, NodeCapabilityPrefix) {
		return fmt.Errorf("invalid capability key %q: must have prefix %q", k, NodeCapabilityPrefix)
	}
	// Validate the capability key and value against the same rules as labels.
	if errs := validation.IsQualifiedName(k); len(errs) > 0 {
		return fmt.Errorf("invalid capability key %q: %s", k, strings.Join(errs, "; "))
	}
	if errs := validation.IsValidLabelValue(v); len(errs) > 0 {
		return fmt.Errorf("invalid capability value %q for key %q: %s", v, k, strings.Join(errs, "; "))
	}
	return nil
}
