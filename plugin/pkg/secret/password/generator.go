/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package password

import (
	"fmt"
	"io"
	"math/rand"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/secret"
)

const (
	// GeneratorName is the name of the password generator
	GeneratorName = "kubernetes.io/password"

	GeneratedPasswordKey = "password"

	// LengthAnnotation is the name of the annotation for password length
	LengthAnnotation = GeneratorName + "-length"

	// CharsAnnotation is the name of the annotation for password chars
	CharsAnnotation = GeneratorName + "-chars"

	DefaultLength = 16

	Alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	Numerals = "0123456789"
	ASCII    = Alphabet + Numerals + "~!@#$%^&*()-_+={}[]\\|<,>.?/\"';:`"
)

func init() {
	secret.RegisterPlugin(GeneratorName, func(client client.Interface, config io.Reader) (secret.Interface, error) {
		passwordGenerator := New(client)
		return passwordGenerator, nil
	})
}

var _ = secret.Interface(&password{})

// New returns an secret.Interface implementation which generates passwords.
func New(cl client.Interface) *password {
	return &password{}
}

type password struct{}

func (s *password) GenerateValues(req *api.GenerateSecretRequest) (map[string][]byte, error) {
	l := DefaultLength
	if req.Annotations[LengthAnnotation] != "" {
		reql, err := strconv.Atoi(req.Annotations[LengthAnnotation])
		if err != nil && reql <= 0 {
			return nil, fmt.Errorf("invalid password length '%s'", req.Annotations[LengthAnnotation])
		}
		l = reql
	}

	chars := ASCII
	if req.Annotations[CharsAnnotation] != "" {
		chars = req.Annotations[CharsAnnotation]
	}

	return map[string][]byte{
		GeneratedPasswordKey: randString(l, chars),
	}, nil
}

// randString makes a random string l characters long.
func randString(length int, chars string) []byte {
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		result[i] = chars[rand.Intn(len(chars))]
	}
	return result
}
