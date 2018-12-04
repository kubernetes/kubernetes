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

package testfiles

var (
	testFiles = make(map[string]bool)
)

// RegisterFiles must be called at init time. It takes the same
// test file paths as Read and remembers that they must be available.
// To simplify that usage, RegisterFiles returns the parameters in a slice and
// thus one can write:
//
// var myFiles = testfile.RegisterFiles("http://example.com/foo", "https://example.com/bar")
// func myFunc() {
//     testfile.Read(myFiles[0])
// }
//
// The recorded information is used in two ways in the Kubernetes E2E test suite:
// - files referenced by URL are cached locally (see test/e2e/e2e_test.go)
// - a missing file causes an error at the start of a test run
//   and aborts the test run; this makes it impossible to
//   accidentally use the wrong path in a test, which otherwise
//   would only get detected when executing the test
func RegisterFiles(filePaths ...string) []string {
	for _, path := range filePaths {
		testFiles[path] = true
	}
	return filePaths
}

// GetRegisteredFiles return all file paths registered so far.
// The order is random.
func GetRegisteredFiles() []string {
	filePaths := make([]string, 0, len(testFiles))
	for path := range testFiles {
		filePaths = append(filePaths, path)
	}
	return filePaths
}
