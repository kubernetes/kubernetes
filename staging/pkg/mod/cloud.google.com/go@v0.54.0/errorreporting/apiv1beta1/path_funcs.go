// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package errorreporting

// ResultPath returns the path for the result resource.
//
// Deprecated: Use
//   fmt.Sprintf("inspect/results/%s", result)
// instead.
func ResultPath(result string) string {
	return "" +
		"inspect/results/" +
		result +
		""
}

// ErrorStatsProjectPath returns the path for the project resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s", project)
// instead.
func ErrorStatsProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}

// ReportErrorsProjectPath returns the path for the project resource.
//
// Deprecated: Use
//   fmt.Sprintf("projects/%s", project)
// instead.
func ReportErrorsProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}
