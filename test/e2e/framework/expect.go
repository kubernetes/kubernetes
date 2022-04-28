/*
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"github.com/onsi/gomega"
)

// ExpectEqual expects the specified two are the same, otherwise an exception raises
func ExpectEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.Equal(extra), explain...)
}

// ExpectNotEqual expects the specified two are not the same, otherwise an exception raises
func ExpectNotEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).NotTo(gomega.Equal(extra), explain...)
}

// ExpectError expects an error happens, otherwise an exception raises
//
// Should not be used anymore. Instead call ExpectErrorExplained
// with an explanation. That helps understand failures when the
// error is not descriptive enough by itself. Even when most errors
// that can occur are descriptive, that is hard to be certain about,
// so it's better to err on the side of caution and always provide
// an explanation.
func ExpectError(err error, explain ...interface{}) {
	gomega.ExpectWithOffset(1, err).To(gomega.HaveOccurred(), explain...)
}

// ExpectErrorExplained checks that an error happened, otherwise the test fails.
// An additional printf-style explanation must be provided because often errors
// cannot be understood without it. If you're unsure what to put for the explanation,
// try to explain the error to a hypothetical person reading failure logs without any other context about the test
func ExpectErrorExplained(err error, explainFormatStr string, explainArgs ...interface{}) {
	ExpectNotEqual(explainFormatStr, "", "An explanation for the error is required.")
	explainArgs = append([]interface{}{explainFormatStr}, explainArgs...)
	gomega.ExpectWithOffset(1, err).To(gomega.HaveOccurred(), explainArgs...)
}

// ExpectNoError checks if "err" is set, and if so, fails assertion while logging the error.
//
// Should not be used anymore. Instead call ExpectNoErrorExplained
// with an explanation. That helps understand failures when the
// error is not descriptive enough by itself. Even when most errors
// that can occur are descriptive, that is hard to be certain about,
// so it's better to err on the side of caution and always provide
// an explanation.
func ExpectNoError(err error, explain ...interface{}) {
	ExpectNoErrorWithOffset(1, err, explain...)
}

// ExpectNoErrorExplained checks if "err" is set, and if so, fails the assertion while logging the error.
// An additional printf-style explanation must be provided because often errors
// cannot be understood without it. If you're unsure what to put for the explanation,
// try to explain the error to a hypothetical person reading failure logs without any other context about the test
func ExpectNoErrorExplained(err error, explainFormatStr string, explainArgs ...interface{}) {
	ExpectNoErrorWithOffsetExplained(1, err, explainFormatStr, explainArgs...)
}

// ExpectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
//
// Should not be used anymore. Instead call ExpectNoErrorWithOffsetExplained
// with an explanation. That helps understand failures when the
// error is not descriptive enough by itself. Even when most errors
// that can occur are descriptive, that is hard to be certain about,
// so it's better to err on the side of caution and always provide
// an explanation.
func ExpectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explain...)
}

// ExpectNoErrorWithOffsetExplained checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffsetExplained(1, ...) error would be logged for "f").
// An additional printf-style explanation must be provided because often errors
// cannot be understood without it. If you're unsure what to put for the explanation,
// try to explain the error to a hypothetical person reading failure logs without any other context about the test
func ExpectNoErrorWithOffsetExplained(offset int, err error, explainFormatStr string, explainArgs ...interface{}) {
	ExpectNotEqual(explainFormatStr, "", "An explanation for the error is required.")
	explainArgs = append([]interface{}{explainFormatStr}, explainArgs...)
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explainArgs...)
}

// ExpectConsistOf expects actual contains precisely the extra elements.  The ordering of the elements does not matter.
func ExpectConsistOf(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.ConsistOf(extra), explain...)
}

// ExpectHaveKey expects the actual map has the key in the keyset
func ExpectHaveKey(actual interface{}, key interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.HaveKey(key), explain...)
}

// ExpectEmpty expects actual is empty
func ExpectEmpty(actual interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.BeEmpty(), explain...)
}
