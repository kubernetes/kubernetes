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

package error

import "fmt"

// SecretError represents error with a secret.
type SecretError struct {
	KustomizationPath string
	// ErrorMsg is an error message
	ErrorMsg string
}

func (e SecretError) Error() string {
	return fmt.Sprintf("Kustomization file [%s] encounters a secret error: %s\n", e.KustomizationPath, e.ErrorMsg)
}
