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

package ktesting

// Must takes a value and an error (or, equivalently, a single function call returning a
// value and an error), asserts that the error did not occur, and returns the value. Use
// this to discard "can't happen" errors that would otherwise complicate test setup, and
// that would indicate a bug in the test case itself rather than a bug in the code being
// tested if they occurred:
//
//    {
//            description: "empty input returns first IP in range",
//            input:       "",
//            expected:    ktesting.Must(netip.ParseAddr("10.0.0.1")),
//    }
func Must[T any](val T, err error) T {
	if err != nil {
		panic(err.Error())
	}
	return val
}
