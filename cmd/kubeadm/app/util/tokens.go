/*
Copyright 2016 The Kubernetes Authors.

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
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

const (
	TokenIDLen = 6
	TokenBytes = 8
)

func RandBytes(length int) ([]byte, string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return nil, "", err
	}
	// It's only the tokenID that doesn't care about raw byte slice,
	// so we just encoded it in place and ignore bytes slice where we
	// do not want it
	return b, hex.EncodeToString(b), nil
}

func GenerateToken(s *kubeadmapi.Secrets) error {
	_, tokenID, err := RandBytes(TokenIDLen / 2)
	if err != nil {
		return err
	}

	tokenBytes, token, err := RandBytes(TokenBytes)
	if err != nil {
		return err
	}

	s.TokenID = tokenID
	s.BearerToken = token
	s.Token = tokenBytes
	s.GivenToken = fmt.Sprintf("%s.%s", tokenID, token)
	return nil
}

func UseGivenTokenIfValid(s *kubeadmapi.Secrets) (bool, error) {
	if s.GivenToken == "" {
		return false, nil // not given
	}
	fmt.Println("<util/tokens> validating provided token")
	givenToken := strings.Split(strings.ToLower(s.GivenToken), ".")
	// TODO(phase1+) could also print more specific messages in each case
	invalidErr := "<util/tokens> provided token does not match expected <6 characters>.<16 characters> format - %s"
	if len(givenToken) != 2 {
		return false, fmt.Errorf(invalidErr, "not in 2-part dot-separated format")
	}
	if len(givenToken[0]) != TokenIDLen {
		return false, fmt.Errorf(invalidErr, fmt.Sprintf(
			"length of first part is incorrect [%d (given) != %d (expected) ]",
			len(givenToken[0]), TokenIDLen))
	}
	tokenBytes := []byte(givenToken[1])
	s.TokenID = givenToken[0]
	s.BearerToken = givenToken[1]
	s.Token = tokenBytes
	return true, nil // given and valid
}
