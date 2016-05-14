/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package keystone

// validator is an interface describes an actor to check whether the given token
// is valid or not
type validator interface {
	// check whether the token is valid, if not, tell why, and if it is valid, then
	// extract the information that has been encoded in this token.
	validate(token string) (string, string, []string, bool, error)
	// support tell whether this kind of token is supported or not
	support(token string) bool
}
