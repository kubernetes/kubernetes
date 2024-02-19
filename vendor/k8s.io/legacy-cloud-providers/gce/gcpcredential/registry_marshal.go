/*
Copyright 2021 The Kubernetes Authors.

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

package gcpcredential

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/cloud-provider/credentialconfig"
)

// registryConfigEntryWithAuth is used solely for deserializing the Auth field
// into a dockerConfigEntry during JSON deserialization.
type registryConfigEntryWithAuth struct {
	// +optional
	Username string `json:"username,omitempty"`
	// +optional
	Password string `json:"password,omitempty"`
	// +optional
	Email string `json:"email,omitempty"`
	// +optional
	Auth string `json:"auth,omitempty"`
}

// RegistryConfigEntry is a serializable wrapper around credentialconfig.RegistryConfigEntry.
type RegistryConfigEntry struct {
	credentialconfig.RegistryConfigEntry
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (ident *RegistryConfigEntry) UnmarshalJSON(data []byte) error {
	var tmp registryConfigEntryWithAuth
	err := json.Unmarshal(data, &tmp)
	if err != nil {
		return err
	}

	ident.Username = tmp.Username
	ident.Password = tmp.Password
	ident.Email = tmp.Email

	if len(tmp.Auth) == 0 {
		return nil
	}

	ident.Username, ident.Password, err = decodeRegistryConfigFieldAuth(tmp.Auth)
	return err
}

// MarshalJSON implements the json.Marshaler interface.
func (ident RegistryConfigEntry) MarshalJSON() ([]byte, error) {
	toEncode := registryConfigEntryWithAuth{ident.Username, ident.Password, ident.Email, ""}
	toEncode.Auth = encodeRegistryConfigFieldAuth(ident.Username, ident.Password)

	return json.Marshal(toEncode)
}

// decodeRegistryConfigFieldAuth deserializes the "auth" field from dockercfg into a
// username and a password. The format of the auth field is base64(<username>:<password>).
func decodeRegistryConfigFieldAuth(field string) (username, password string, err error) {

	var decoded []byte

	// StdEncoding can only decode padded string
	// RawStdEncoding can only decode unpadded string
	if strings.HasSuffix(strings.TrimSpace(field), "=") {
		// decode padded data
		decoded, err = base64.StdEncoding.DecodeString(field)
	} else {
		// decode unpadded data
		decoded, err = base64.RawStdEncoding.DecodeString(field)
	}

	if err != nil {
		return
	}

	parts := strings.SplitN(string(decoded), ":", 2)
	if len(parts) != 2 {
		err = fmt.Errorf("unable to parse auth field, must be formatted as base64(username:password)")
		return
	}

	username = parts[0]
	password = parts[1]

	return
}

func encodeRegistryConfigFieldAuth(username, password string) string {
	fieldValue := username + ":" + password

	return base64.StdEncoding.EncodeToString([]byte(fieldValue))
}
