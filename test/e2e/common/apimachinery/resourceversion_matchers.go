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

package apimachinery

import (
	"fmt"
	"math/big"

	"github.com/onsi/gomega/types"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// HaveValidResourceVersion returns a Gomega matcher that checks if a kubernetes
// objects resource version is a valid 128-bit unsigned integer.
func HaveValidResourceVersion() types.GomegaMatcher {
	return &HaveValidResourceVersionMatcher{}
}

type HaveValidResourceVersionMatcher struct {
	failureReason string
}

// Match is a gomega matcher that checks whether or not the provided interface
// is a metav1.Object with a uint128 resource version string.
func (m *HaveValidResourceVersionMatcher) Match(actual interface{}) (success bool, err error) {
	// First, ensure the input is an object.
	o, ok := actual.(metav1.Object)
	if !ok {
		return false, fmt.Errorf("HaveValidResourceVersionMatcher matcher expects an api object, but got %T", actual)
	}

	val, ok := new(big.Int).SetString(o.GetResourceVersion(), 10)

	if !ok {
		m.failureReason = "the resource version is not a valid integer"
		return false, nil
	}
	if val.Sign() < 0 {
		m.failureReason = "the resource version is a negative number"
		return false, nil
	}
	if val.BitLen() > 128 {
		m.failureReason = fmt.Sprintf("resource version requires %d bits (more than 128)", val.BitLen())
		return false, nil
	}
	if val.Cmp(big.NewInt(0)) == 0 {
		m.failureReason = "the resource version is zero which is not valid"
		return false, nil
	}

	return true, nil
}

// FailureMessage is used when the assertion is `Expect(...).To(...)`
func (m *HaveValidResourceVersionMatcher) FailureMessage(actual interface{}) (message string) {
	o, ok := actual.(metav1.Object)
	if !ok {
		return fmt.Sprintf("HaveValidResourceVersion matcher expects an api object, but got %T", actual)
	}
	return fmt.Sprintf("Expected resource version to be a valid uint128, but got %q: %s", o.GetResourceVersion(), m.failureReason)
}

// NegatedFailureMessage is used when the assertion is `Expect(...).ToNot(...)`
func (m *HaveValidResourceVersionMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	o, ok := actual.(metav1.Object)
	if !ok {
		return fmt.Sprintf("HaveValidResourceVersion matcher expects an api object, but got %T", actual)
	}
	return fmt.Sprintf("Expected resource version not to be a valid uint128, but got %q", o.GetResourceVersion())
}
