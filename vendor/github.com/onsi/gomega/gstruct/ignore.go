// untested sections: 2

package gstruct

import (
	"github.com/onsi/gomega/types"
)

// Ignore ignores the actual value and always succeeds.
//
//	Expect(nil).To(Ignore())
//	Expect(true).To(Ignore())
func Ignore() types.GomegaMatcher {
	return &IgnoreMatcher{true}
}

// Reject ignores the actual value and always fails. It can be used in conjunction with IgnoreMissing
// to catch problematic elements, or to verify tests are running.
//
//	Expect(nil).NotTo(Reject())
//	Expect(true).NotTo(Reject())
func Reject() types.GomegaMatcher {
	return &IgnoreMatcher{false}
}

// A matcher that either always succeeds or always fails.
type IgnoreMatcher struct {
	Succeed bool
}

func (m *IgnoreMatcher) Match(actual any) (bool, error) {
	return m.Succeed, nil
}

func (m *IgnoreMatcher) FailureMessage(_ any) (message string) {
	return "Unconditional failure"
}

func (m *IgnoreMatcher) NegatedFailureMessage(_ any) (message string) {
	return "Unconditional success"
}
