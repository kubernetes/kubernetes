// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package admin

// IamProjectPath returns the path for the project resource.
func IamProjectPath(project string) string {
	return "" +
		"projects/" +
		project +
		""
}

// IamServiceAccountPath returns the path for the service account resource.
func IamServiceAccountPath(project, serviceAccount string) string {
	return "" +
		"projects/" +
		project +
		"/serviceAccounts/" +
		serviceAccount +
		""
}

// IamKeyPath returns the path for the key resource.
func IamKeyPath(project, serviceAccount, key string) string {
	return "" +
		"projects/" +
		project +
		"/serviceAccounts/" +
		serviceAccount +
		"/keys/" +
		key +
		""
}
