/*
Copyright The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"strings"
)

// EnhanceRuntimeError provides a more helpful error message for "no such key" errors.
// For all other errors, it returns the error unchanged.
func EnhanceRuntimeError(err error) error {
	if err == nil {
		return nil
	}

	if strings.HasPrefix(err.Error(), "no such key:") {
		return fmt.Errorf("%s. Consider using CEL optional chaining (?), has() macro, or orValue() for optional fields", err)
	}

	// Not a "no such key" error, return unchanged.
	return err
}
