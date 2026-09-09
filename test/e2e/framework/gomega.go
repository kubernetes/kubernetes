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

// Package framework contains provider-independent helper code for
// building and running E2E tests with Ginkgo. The actual Ginkgo test
// suites gets assembled by combining this framework, the optional
// provider support code and specific tests via a separate .go file
// like Kubernetes' test/e2e.go.
package framework

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega/format"
	gtypes "github.com/onsi/gomega/types"
)

// GomegaObject returns a matcher which appends a full dump of the actual value
// to the failure of the matcher that it wraps. This is useful e.g. for
// gomega.HaveField which otherwise only generates a message containing
// the field that it is checking, but not the object in which that field occurs.
func GomegaObject(shouldMatch gtypes.GomegaMatcher) gtypes.GomegaMatcher {
	return &gomegaObjectMatcher{
		shouldMatch: shouldMatch,
	}
}

type gomegaObjectMatcher struct {
	shouldMatch gtypes.GomegaMatcher
}

func (m *gomegaObjectMatcher) Match(actual interface{}) (success bool, err error) {
	return m.shouldMatch.Match(actual)
}

func (m *gomegaObjectMatcher) FailureMessage(actual interface{}) (message string) {
	return m.withDump(actual, m.shouldMatch.FailureMessage(actual))
}

func (m *gomegaObjectMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return m.withDump(actual, m.shouldMatch.NegatedFailureMessage(actual))
}

func (m *gomegaObjectMatcher) withDump(actual any, message string) string {
	dump := format.Object(actual, 1)
	if !strings.HasSuffix(message, "\n") {
		message += "\n"
	}
	message += fmt.Sprintf("\nFull object:\n%s", dump)
	return message
}
