/*
Package ownership manages access to resources
Copyright 2019 Portworx

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
package api

// IsMatch returns true if the ownership has at least one similar
// owner, group, or collaborator
func (g *Group) IsMatch(check *Group) bool {
	if check == nil {
		return false
	}

	// Check user
	if g.Id == check.GetId() {
		return true
	}

	return false
}
