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

// Package vault implements envelop encryption provider based on Vault KMS
package vault

import (
	"errors"
	"fmt"
	"net/http"
	"path"
	"reflect"

	"github.com/hashicorp/vault/api"
)

const defaultTransitPath = "transit"
const defaultAuthPath = "auth"

// Handle all communication with Vault server.
type clientWrapper struct {
	client      *api.Client
	encryptPath string
	decryptPath string
	authPath    string
}

// Initialize a client wrapper for vault kms provider.
func newClientWrapper(config *EnvelopeConfig) (*clientWrapper, error) {
	client, err := newVaultClient(config)
	if err != nil {
		return nil, err
	}

	// Vault transit path is configurable. "path", "/path", "path/" and "/path/"
	// are the same.
	transit := defaultTransitPath
	if config.TransitPath != "" {
		transit = config.TransitPath
	}

	// auth path is configurable. "path", "/path", "path/" and "/path/" are the same.
	auth := defaultAuthPath
	if config.AuthPath != "" {
		auth = config.AuthPath
	}
	wrapper := &clientWrapper{
		client:      client,
		encryptPath: path.Join("v1", transit, "encrypt"),
		decryptPath: path.Join("v1", transit, "decrypt"),
		authPath:    path.Join(auth),
	}

	// Set token for the api.client.
	if config.Token != "" {
		client.SetToken(config.Token)
	} else {
		err = wrapper.refreshToken(config)
	}
	if err != nil {
		return nil, err
	}

	return wrapper, nil
}

func newVaultClient(config *EnvelopeConfig) (*api.Client, error) {
	vaultConfig := api.DefaultConfig()
	vaultConfig.Address = config.Address

	tlsConfig := &api.TLSConfig{
		CACert:        config.CACert,
		ClientCert:    config.ClientCert,
		ClientKey:     config.ClientKey,
		TLSServerName: config.TLSServerName,
	}
	if err := vaultConfig.ConfigureTLS(tlsConfig); err != nil {
		return nil, err
	}

	return api.NewClient(vaultConfig)
}

// Get token by login and set the value to api.Client.
func (c *clientWrapper) refreshToken(config *EnvelopeConfig) error {
	switch {
	case config.ClientCert != "" && config.ClientKey != "":
		token, err := c.tlsToken(config)
		if err != nil {
			return err
		}
		c.client.SetToken(token)
	case config.RoleID != "":
		token, err := c.appRoleToken(config)
		if err != nil {
			return err
		}
		c.client.SetToken(token)
	default:
		// configuration has already been validated, flow should not reach here
		return errors.New("the Vault authentication configuration is invalid")
	}

	return nil
}

func (c *clientWrapper) tlsToken(config *EnvelopeConfig) (string, error) {
	resp, err := c.client.Logical().Write("/"+path.Join(c.authPath, "cert", "login"), nil)
	if err != nil {
		return "", err
	}

	return resp.Auth.ClientToken, nil
}

func (c *clientWrapper) appRoleToken(config *EnvelopeConfig) (string, error) {
	data := map[string]interface{}{
		"role_id":   config.RoleID,
		"secret_id": config.SecretID,
	}
	resp, err := c.client.Logical().Write("/"+path.Join(c.authPath, "approle", "login"), data)
	if err != nil {
		return "", err
	}

	return resp.Auth.ClientToken, nil
}

func (c *clientWrapper) decrypt(keyName string, cipher string) (string, error) {
	var result string

	data := map[string]string{"ciphertext": cipher}
	resp, err := c.request(path.Join(c.decryptPath, keyName), data)
	if err != nil {
		return result, err
	}

	result, ok := resp.Data["plaintext"].(string)
	if !ok {
		return result, fmt.Errorf("failed type assertion of vault decrypt response type: %v to string", reflect.TypeOf(resp.Data["plaintext"]))
	}

	return result, nil
}

func (c *clientWrapper) encrypt(keyName string, plain string) (string, error) {
	var result string

	data := map[string]string{"plaintext": plain}
	resp, err := c.request(path.Join(c.encryptPath, keyName), data)
	if err != nil {
		return result, err
	}

	result, ok := resp.Data["ciphertext"].(string)
	if !ok {
		return result, fmt.Errorf("failed type assertion of vault encrypt response type: %v to string", reflect.TypeOf(resp.Data["ciphertext"]))
	}

	return result, nil
}

// This request check the response status code. If get code 403, it sets forbidden true.
func (c *clientWrapper) request(path string, data interface{}) (*api.Secret, error) {
	req := c.client.NewRequest("POST", "/"+path)
	if err := req.SetJSONBody(data); err != nil {
		return nil, err
	}

	resp, err := c.client.RawRequest(req)
	if resp != nil {
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusForbidden {
			return nil, &forbiddenError{err: err}
		}

		if resp.StatusCode == http.StatusOK {
			secret, err := api.ParseSecret(resp.Body)
			if err != nil {
				return nil, err
			}
			return secret, nil
		}
		return nil, fmt.Errorf("unexpected response code: %v received for POST request to %v", resp.StatusCode, path)
	}
	if err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no response received for POST request to %v", path)
}

// Return this error when get HTTP code 403.
type forbiddenError struct {
	err error
}

func (e *forbiddenError) Error() string {
	return fmt.Sprintf("error %s", e.err)
}
