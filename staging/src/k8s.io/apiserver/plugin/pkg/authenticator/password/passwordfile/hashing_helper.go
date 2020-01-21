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

package passwordfile

import (
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"regexp"
	"strconv"

	"golang.org/x/crypto/argon2"
)

type passwordChecker interface {
	checkPassword(password string) error
}

type argon2IDChecker struct {
	passwordHash []byte
	salt         []byte
	iterations   uint32
	memory       uint32
	threads      uint8
}

var argon2IDRegEx = regexp.MustCompile(`^\$argon2id\$v=(?P<version>[0-9]+)\$m=(?P<memory>[0-9]+),t=(?P<iterations>[0-9]+),p=(?P<threads>[0-9]+)\$(?P<salt>[a-zA-Z0-9+\/]{22,})\$(?P<hash>[a-zA-Z0-9+\/]{43,})$`)

func extractParams(encodedHash string) (map[string]string, error) {
	match := argon2IDRegEx.FindStringSubmatch(encodedHash)
	if match == nil {
		return nil, fmt.Errorf("input string format did not match format, want: %q", "$argon2id$v=%%d$m=%%d,t=%%d,p=%%d$salt$hash")
	}
	result := make(map[string]string)
	for i, name := range argon2IDRegEx.SubexpNames() {
		result[name] = match[i]
	}
	return result, nil
}

func newArgon2IDChecker(encodedHash string) (*argon2IDChecker, error) {
	params, err := extractParams(encodedHash)
	if err != nil {
		return nil, err
	}
	version, err := strconv.ParseInt(params["version"], 10, 0)
	if err != nil {
		return nil, err
	}
	if version != argon2.Version {
		return nil, fmt.Errorf("unsupported argon2id version: %d, expected version: %d", version, argon2.Version)
	}
	memory, err := strconv.ParseUint(params["memory"], 10, 32)
	if err != nil {
		return nil, err
	}
	iterations, err := strconv.ParseUint(params["iterations"], 10, 32)
	if err != nil {
		return nil, err
	}
	threads, err := strconv.ParseUint(params["threads"], 10, 8)
	if err != nil {
		return nil, err
	}
	salt, err := base64.RawStdEncoding.DecodeString(params["salt"])
	if err != nil {
		return nil, err
	}
	passwordHash, err := base64.RawStdEncoding.DecodeString(params["hash"])
	if err != nil {
		return nil, err
	}
	return &argon2IDChecker{
		passwordHash: passwordHash,
		salt:         salt,
		iterations:   uint32(iterations),
		memory:       uint32(memory),
		threads:      uint8(threads),
	}, nil
}

func (a *argon2IDChecker) checkPassword(password string) error {
	got := argon2.IDKey([]byte(password), a.salt, a.iterations, a.memory, a.threads, uint32(len(a.passwordHash)))
	if subtle.ConstantTimeCompare(a.passwordHash, got) == 0 {
		return fmt.Errorf("password did not match password hash")
	}
	return nil
}
