/*
Copyright 2019 The Kubernetes Authors.

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

import (
	"fmt"
	"strings"
)

// TokenFromIDAndSecret returns the full token which is of the form "{id}.{secret}"
func TokenFromIDAndSecret(id, secret string) string {
	return fmt.Sprintf("%s.%s", id, secret)
}

// MarshalJSON implements the json.Marshaler interface.
func (bts BootstrapTokenString) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%s"`, bts.String())), nil
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (bts *BootstrapTokenString) UnmarshalJSON(b []byte) error {
	// If the token is represented as "", just return quickly without an error
	if len(b) == 0 {
		return nil
	}

	// Remove unnecessary " characters coming from the JSON parser
	token := strings.Replace(string(b), `"`, ``, -1)
	// Convert the string Token to a BootstrapTokenString object
	newbts, err := NewBootstrapTokenString(token)
	if err != nil {
		return err
	}
	bts.ID = newbts.ID
	bts.Secret = newbts.Secret
	return nil
}

// String returns the string representation of the BootstrapTokenString
func (bts BootstrapTokenString) String() string {
	if len(bts.ID) > 0 && len(bts.Secret) > 0 {
		return TokenFromIDAndSecret(bts.ID, bts.Secret)
	}
	return ""
}
