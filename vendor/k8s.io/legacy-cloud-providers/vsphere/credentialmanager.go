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

package vsphere

import (
	"errors"
	"fmt"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/listers/core/v1"
	"k8s.io/klog"
	"net/http"
	"strings"
	"sync"
)

// Error Messages
const (
	CredentialsNotFoundErrMsg = "Credentials not found"
	CredentialMissingErrMsg   = "Username/Password is missing"
	UnknownSecretKeyErrMsg    = "Unknown secret key"
)

// Error constants
var (
	ErrCredentialsNotFound = errors.New(CredentialsNotFoundErrMsg)
	ErrCredentialMissing   = errors.New(CredentialMissingErrMsg)
	ErrUnknownSecretKey    = errors.New(UnknownSecretKeyErrMsg)
)

type SecretCache struct {
	cacheLock     sync.Mutex
	VirtualCenter map[string]*Credential
	Secret        *corev1.Secret
}

type Credential struct {
	User     string `gcfg:"user"`
	Password string `gcfg:"password"`
}

type SecretCredentialManager struct {
	SecretName      string
	SecretNamespace string
	SecretLister    v1.SecretLister
	Cache           *SecretCache
}

// GetCredential returns credentials for the given vCenter Server.
// GetCredential returns error if Secret is not added.
// GetCredential return error is the secret doesn't contain any credentials.
func (secretCredentialManager *SecretCredentialManager) GetCredential(server string) (*Credential, error) {
	err := secretCredentialManager.updateCredentialsMap()
	if err != nil {
		statusErr, ok := err.(*apierrors.StatusError)
		if (ok && statusErr.ErrStatus.Code != http.StatusNotFound) || !ok {
			return nil, err
		}
		// Handle secrets deletion by finding credentials from cache
		klog.Warningf("secret %q not found in namespace %q", secretCredentialManager.SecretName, secretCredentialManager.SecretNamespace)
	}

	credential, found := secretCredentialManager.Cache.GetCredential(server)
	if !found {
		klog.Errorf("credentials not found for server %q", server)
		return nil, ErrCredentialsNotFound
	}
	return &credential, nil
}

func (secretCredentialManager *SecretCredentialManager) updateCredentialsMap() error {
	if secretCredentialManager.SecretLister == nil {
		return fmt.Errorf("SecretLister is not initialized")
	}
	secret, err := secretCredentialManager.SecretLister.Secrets(secretCredentialManager.SecretNamespace).Get(secretCredentialManager.SecretName)
	if err != nil {
		klog.Errorf("Cannot get secret %s in namespace %s. error: %q", secretCredentialManager.SecretName, secretCredentialManager.SecretNamespace, err)
		return err
	}
	cacheSecret := secretCredentialManager.Cache.GetSecret()
	if cacheSecret != nil &&
		cacheSecret.GetResourceVersion() == secret.GetResourceVersion() {
		klog.V(4).Infof("VCP SecretCredentialManager: Secret %q will not be updated in cache. Since, secrets have same resource version %q", secretCredentialManager.SecretName, cacheSecret.GetResourceVersion())
		return nil
	}
	secretCredentialManager.Cache.UpdateSecret(secret)
	return secretCredentialManager.Cache.parseSecret()
}

func (cache *SecretCache) GetSecret() *corev1.Secret {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	return cache.Secret
}

func (cache *SecretCache) UpdateSecret(secret *corev1.Secret) {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	cache.Secret = secret
}

func (cache *SecretCache) GetCredential(server string) (Credential, bool) {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	credential, found := cache.VirtualCenter[server]
	if !found {
		return Credential{}, found
	}
	return *credential, found
}

func (cache *SecretCache) parseSecret() error {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	return parseConfig(cache.Secret.Data, cache.VirtualCenter)
}

// parseConfig returns vCenter ip/fdqn mapping to its credentials viz. Username and Password.
func parseConfig(data map[string][]byte, config map[string]*Credential) error {
	if len(data) == 0 {
		return ErrCredentialMissing
	}
	for credentialKey, credentialValue := range data {
		credentialKey = strings.ToLower(credentialKey)
		vcServer := ""
		if strings.HasSuffix(credentialKey, "password") {
			vcServer = strings.Split(credentialKey, ".password")[0]
			if _, ok := config[vcServer]; !ok {
				config[vcServer] = &Credential{}
			}
			config[vcServer].Password = string(credentialValue)
		} else if strings.HasSuffix(credentialKey, "username") {
			vcServer = strings.Split(credentialKey, ".username")[0]
			if _, ok := config[vcServer]; !ok {
				config[vcServer] = &Credential{}
			}
			config[vcServer].User = string(credentialValue)
		} else {
			klog.Errorf("Unknown secret key %s", credentialKey)
			return ErrUnknownSecretKey
		}
	}
	for vcServer, credential := range config {
		if credential.User == "" || credential.Password == "" {
			klog.Errorf("Username/Password is missing for server %s", vcServer)
			return ErrCredentialMissing
		}
	}
	return nil
}
