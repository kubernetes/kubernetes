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

package versioned

import (
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/kubectl/pkg/generate"
	"k8s.io/kubectl/pkg/util/hash"
)

// SecretForDecryptImageGeneratorV1 supports stable generation of a image decrypt secret
type SecretForDecryptImageGeneratorV1 struct {
	// Name of secret (required)
	Name string
	// PrivateKey is represed by the base64 version of it along with the password or in case
	// PKCS base64 of the certificate file content. So a typical gpg private key will look like,
	// <base64 of private key:password>
	PrivateKeyPasswds []string
	// AppendHash; if true, derive a hash from the Secret and append it to the name
	AppendHash bool
}

// Ensure it supports the generator pattern that uses parameter injection
var _ generate.Generator = &SecretForDecryptImageGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ generate.StructuredGenerator = &SecretForDecryptImageGeneratorV1{}

// Generate returns a secret using the specified parameters
func (s SecretForDecryptImageGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := generate.ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &SecretForDecryptImageGeneratorV1{}
	hashParam, found := genericParams["append-hash"]
	if found {
		hashBool, isBool := hashParam.(bool)
		if !isBool {
			return nil, fmt.Errorf("expected bool, found :%v", hashParam)
		}
		delegate.AppendHash = hashBool
		delete(genericParams, "append-hash")
	}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	delegate.Name = params["name"]
	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs a secret object using the configured fields
func (s SecretForDecryptImageGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	secret := &v1.Secret{}
	secret.Name = s.Name
	secret.Type = v1.SecretTypeDecryptKey
	secret.Data = map[string][]byte{}

	privateKeyContent, err := handleDecryptCfgJsonContent(s.PrivateKeyPasswds)
	if err != nil {
		return nil, err
	}
	secret.Data[v1.ImageDecryptionKey] = privateKeyContent

	if s.AppendHash {
		h, err := hash.SecretHash(secret)
		if err != nil {
			return nil, err
		}
		secret.Name = fmt.Sprintf("%s-%s", secret.Name, h)
	}
	return secret, nil
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern
func (s SecretForDecryptImageGeneratorV1) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "name", Required: true},
		{Name: "decrypt-secret", Required: true},
		{Name: "append-hash", Required: false},
	}
}

// validate validates required fields are set to support structured generation
func (s SecretForDecryptImageGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}

	if len(s.PrivateKeyPasswds) == 0 {
		return fmt.Errorf("private key must be specified")
	}
	return nil
}

// handleDecryptCfgJsonContent serializes a decryption key
func handleDecryptCfgJsonContent(privateKeyPasswds []string) ([]byte, error) {
	DecryptParams := DecryptConfigEntry{
		PrivateKeyPasswds: privateKeyPasswds,
	}
	return json.Marshal(DecryptParams)
}

// DecryptConfigEntry represents the base64 of the private key along
// with it's corresponding password
type DecryptConfigEntry struct {
	PrivateKeyPasswds []string `json:"privatekey"`
}
