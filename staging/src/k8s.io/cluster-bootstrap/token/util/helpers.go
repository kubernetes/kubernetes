/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"bufio"
	"crypto/rand"
	"fmt"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cluster-bootstrap/token/api"
)

// validBootstrapTokenChars defines the characters a bootstrap token can consist of
const validBootstrapTokenChars = "0123456789abcdefghijklmnopqrstuvwxyz"

var (
	// BootstrapTokenRegexp is a compiled regular expression of TokenRegexpString
	BootstrapTokenRegexp = regexp.MustCompile(api.BootstrapTokenPattern)
	// BootstrapTokenIDRegexp is a compiled regular expression of TokenIDRegexpString
	BootstrapTokenIDRegexp = regexp.MustCompile(api.BootstrapTokenIDPattern)
	// BootstrapGroupRegexp is a compiled regular expression of BootstrapGroupPattern
	BootstrapGroupRegexp = regexp.MustCompile(api.BootstrapGroupPattern)
)

// GenerateBootstrapToken generates a new, random Bootstrap Token.
func GenerateBootstrapToken() (string, error) {
	tokenID, err := randBytes(api.BootstrapTokenIDBytes)
	if err != nil {
		return "", err
	}

	tokenSecret, err := randBytes(api.BootstrapTokenSecretBytes)
	if err != nil {
		return "", err
	}

	return TokenFromIDAndSecret(tokenID, tokenSecret), nil
}

// randBytes returns a random string consisting of the characters in
// validBootstrapTokenChars, with the length customized by the parameter
func randBytes(length int) (string, error) {
	// len("0123456789abcdefghijklmnopqrstuvwxyz") = 36 which doesn't evenly divide
	// the possible values of a byte: 256 mod 36 = 4. Discard any random bytes we
	// read that are >= 252 so the bytes we evenly divide the character set.
	const maxByteValue = 252

	var (
		b     byte
		err   error
		token = make([]byte, length)
	)

	reader := bufio.NewReaderSize(rand.Reader, length*2)
	for i := range token {
		for {
			if b, err = reader.ReadByte(); err != nil {
				return "", err
			}
			if b < maxByteValue {
				break
			}
		}

		token[i] = validBootstrapTokenChars[int(b)%len(validBootstrapTokenChars)]
	}

	return string(token), nil
}

// TokenFromIDAndSecret returns the full token which is of the form "{id}.{secret}"
func TokenFromIDAndSecret(id, secret string) string {
	return fmt.Sprintf("%s.%s", id, secret)
}

// IsValidBootstrapToken returns whether the given string is valid as a Bootstrap Token and
// in other words satisfies the BootstrapTokenRegexp
func IsValidBootstrapToken(token string) bool {
	return BootstrapTokenRegexp.MatchString(token)
}

// IsValidBootstrapTokenID returns whether the given string is valid as a Bootstrap Token ID and
// in other words satisfies the BootstrapTokenIDRegexp
func IsValidBootstrapTokenID(tokenID string) bool {
	return BootstrapTokenIDRegexp.MatchString(tokenID)
}

// BootstrapTokenSecretName returns the expected name for the Secret storing the
// Bootstrap Token in the Kubernetes API.
func BootstrapTokenSecretName(tokenID string) string {
	return fmt.Sprintf("%s%s", api.BootstrapTokenSecretPrefix, tokenID)
}

// ValidateBootstrapGroupName checks if the provided group name is a valid
// bootstrap group name. Returns nil if valid or a validation error if invalid.
func ValidateBootstrapGroupName(name string) error {
	if BootstrapGroupRegexp.Match([]byte(name)) {
		return nil
	}
	return fmt.Errorf("bootstrap group %q is invalid (must match %s)", name, api.BootstrapGroupPattern)
}

// ValidateUsages validates that the passed in string are valid usage strings for bootstrap tokens.
func ValidateUsages(usages []string) error {
	validUsages := sets.NewString(api.KnownTokenUsages...)
	invalidUsages := sets.NewString()
	for _, usage := range usages {
		if !validUsages.Has(usage) {
			invalidUsages.Insert(usage)
		}
	}
	if len(invalidUsages) > 0 {
		return fmt.Errorf("invalid bootstrap token usage string: %s, valid usage options: %s", strings.Join(invalidUsages.List(), ","), strings.Join(api.KnownTokenUsages, ","))
	}
	return nil
}
