// +build !linux

/*
Copyright 2020 The Kubernetes Authors.

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

package system

// NOOP for non-Linux OSes.

// packageValidator implements the Validator interface. It validates packages
// and their versions.
type packageValidator struct {
	reporter Reporter
}

// Name returns the name of the package validator.
func (validator *packageValidator) Name() string {
	return "package"
}

// Validate checks packages and their versions against the packageSpecs using
// the packageManager, and returns an error on any package/version mismatch.
func (validator *packageValidator) Validate(spec SysSpec) ([]error, []error) {
	return nil, nil
}
