/*
Copyright 2014 The Go4 Authors

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

// Package legal provides in-process storage for compiled-in licenses.
package legal // import "go4.org/legal"

var licenses []string

// RegisterLicense stores the license text.
// It doesn't check whether the text was already present.
func RegisterLicense(text string) {
	licenses = append(licenses, text)
	return
}

// Licenses returns a slice of the licenses.
func Licenses() []string {
	return licenses
}
