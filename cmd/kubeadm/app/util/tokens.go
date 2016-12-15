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
	"regexp"
	"strconv"
	"strings"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
)

const (
	TokenIDBytes = 3
	TokenBytes   = 8
)

func RandBytes(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func GenerateToken(d *kubeadmapi.TokenDiscovery) error {
	tokenID, err := RandBytes(TokenIDBytes)
	if err != nil {
		return err
	}

	token, err := RandBytes(TokenBytes)
	if err != nil {
		return err
	}

	d.ID = tokenID
	d.Secret = token
	return nil
}

var (
	tokenRegexpString = "^([a-zA-Z0-9]{6})\\.([a-zA-Z0-9]{16})$"
	tokenRegexp       = regexp.MustCompile(tokenRegexpString)
)

func ParseToken(s string) (string, string, error) {
	split := tokenRegexp.FindStringSubmatch(s)
	if len(split) != 3 {
		return "", "", fmt.Errorf("token %q was not of form %q", s, tokenRegexpString)
	}
	return split[1], split[2], nil

}

func BearerToken(d *kubeadmapi.TokenDiscovery) string {
	return fmt.Sprintf("%s.%s", d.ID, d.Secret)
}

func IsTokenValid(d *kubeadmapi.TokenDiscovery) (bool, error) {
	if len(d.ID)+len(d.Secret) == 0 {
		return false, nil
	}
	if _, _, err := ParseToken(d.ID + "." + d.Secret); err != nil {
		return false, err
	}
	return true, nil
}

func DiscoveryPort(d *kubeadmapi.TokenDiscovery) int32 {
	if len(d.Addresses) == 0 {
		return kubeadmapiext.DefaultDiscoveryBindPort
	}
	split := strings.Split(d.Addresses[0], ":")
	if len(split) == 1 {
		return kubeadmapiext.DefaultDiscoveryBindPort
	}
	if i, err := strconv.Atoi(split[1]); err != nil {
		return int32(i)
	}
	return kubeadmapiext.DefaultDiscoveryBindPort
}
