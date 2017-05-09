/*
Copyright 2016 The Kubernetes Authors.

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

package matchers

import (
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/types"
)

// Convenience function for matching values that can be nil.
func NilOr(ms ...types.GomegaMatcher) types.GomegaMatcher {
	return gomega.Or(append(ms, gomega.BeNil())...)
}

// Check that the actual value is within the expected range.
// lower and upper must be numeric values.
func InRange(lower, upper interface{}) types.GomegaMatcher {
	return gomega.And(gomega.BeNumerically(">=", lower), gomega.BeNumerically("<=", upper))
}

// Applies the matcher m to the value being pointed to.
func Ptr(m types.GomegaMatcher) types.GomegaMatcher {
	return &PtrMatcher{
		Matcher: m,
	}
}

// A matcher that always succeeds.
func Ignore() types.GomegaMatcher {
	return gomega.And()
}

// A matcher that always fails (mostly for development).
func Fail() types.GomegaMatcher {
	return gomega.Or()
}

// A struct matcher where each field must be present and match the expectations.
func StrictStruct(fields Fields) types.GomegaMatcher {
	return &StructMatcher{
		Fields: fields,
		Strict: true,
	}
}

// A struct matcher where present fields must match the expectations, and extra fields are ignored.
func LooseStruct(fields Fields) types.GomegaMatcher {
	return &StructMatcher{
		Fields: fields,
		Strict: false,
	}
}

// A slice matcher where each element identified by the identifier must be present and match the
// expectations.
func StrictSlice(identifier Identifier, elements Elements) types.GomegaMatcher {
	return &SliceMatcher{
		Identifier: identifier,
		Elements:   elements,
		Strict:     true,
	}
}

// A slice matcher where present elements identified by the identifier must match the expectations,
// and extra elements are ignored.
func LooseSlice(identifier Identifier, elements Elements) types.GomegaMatcher {
	return &SliceMatcher{
		Identifier: identifier,
		Elements:   elements,
		Strict:     false,
	}
}

// A matcher that checks that the actual time was within duration d of Now.
func Recent(d time.Duration) types.GomegaMatcher {
	return &RecentMatcher{d}
}
