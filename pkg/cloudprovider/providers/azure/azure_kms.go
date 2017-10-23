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

package azure

import (
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/dataplane/keyvault"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
)

const (
	defaultKeyName   = "k8s-secret-key"
	azureKeyVaultKms = "azure-keyvault"
)

type azureKmsConfig struct {
	VaultName  string `json:"vaultName,omitempty"`
	KeyName    string `json:"keyName,omitempty"`
	KeyVersion string `json:"keyVersion,omitempty"`
}

type azureKms struct {
	providerName string // for logging
	azureKmsConfig
	vaultUri string
	az       *Cloud
}

var registereOnce sync.Once

func (az *Cloud) shouldUseKms() bool {

	// No ClusterKeyVault: older cluster won't have this value
	if "" == az.ClusterKeyVault {
		return false
	}

	// any other value means use key vault as Kms
	return true
}

func (az *Cloud) RegisterKeyVaultAsKmsService() {
	registereOnce.Do(func() {
		registerKms(az)
	})
}

func applyOverrides(kms *azureKms, config io.Reader) (bool, error) {
	override := &azureKms{}
	applied := false
	bytes, err := ioutil.ReadAll(config)
	if err != nil {
		return applied, err
	}

	if err := yaml.Unmarshal(bytes, override); err != nil {
		return applied, err
	}

	if override.VaultName != "" {
		glog.V(2).Infof("azureKms - Provider:%s VaultName:[%s] override with:[%s]", kms.providerName, kms.VaultName, override.VaultName)
		kms.VaultName = override.VaultName
		applied = true
	}

	if override.KeyName != "" {
		glog.V(2).Infof("azureKms - Provider:%s KeyName:[%s] override with:[%s]", kms.providerName, kms.KeyName, override.KeyName)

		kms.KeyName = override.KeyName
		applied = true
	}

	if override.KeyVersion != "" {
		glog.V(2).Infof("azureKms - Provider:%s Version:[%s] override with:[%s]", kms.providerName, kms.KeyVersion, override.KeyVersion)
		kms.KeyVersion = override.KeyVersion
		applied = true
	}

	return applied, nil
}

func registerKms(az *Cloud) {
	if false == az.shouldUseKms() {
		return // KeyVault is not enabled
	}

	encryptionconfig.KMSPluginRegistry.Register(azureKeyVaultKms, func(config io.Reader) (envelope.Service, error) {
		kms := &azureKms{}
		kms.az = az
		kms.KeyName = defaultKeyName
		kms.VaultName = az.ClusterKeyVault

		overrideApplied := false

		// Check if we have overrides from the user
		overrideApplied, err := applyOverrides(kms, config)
		if err != nil {
			return nil, fmt.Errorf("azureKms - Provider:%s failed to load overrides with error:%s", kms.providerName, err)
		}

		// FQDN according to cluster's cloud environment
		kms.vaultUri = fmt.Sprintf("https://%s.%s", kms.VaultName, az.Environment.KeyVaultDNSSuffix)
		/*
			 The only way to test that the key exists is to use it.
			 		 if we running with defaul settings (we assume that we have create permission), then
					 		 we will attempt to create the key if not existent

							 		 *if we are running with overrides then we assume that the user create them (with proper permission)
		*/

		sample := []byte("Hello, World!")
		value := base64.RawURLEncoding.EncodeToString(sample)
		parameters := keyvault.KeyOperationsParameters{
			Algorithm: keyvault.RSA15,
			Value:     &value,
		}

		// Use user provided backoff or
		// use custom values
		backoff := az.resourceRequestBackoff
		if 0 == backoff.Steps {
			backoff.Duration = 1 * time.Second
			backoff.Factor = 1.5
			backoff.Steps = 10
		}

		err = wait.ExponentialBackoff(backoff, func() (bool, error) {
			// If key is not there create it. any other error is returned to user
			result, err := az.KeyVaultClient.Encrypt(kms.vaultUri, kms.KeyName, kms.KeyVersion, parameters)
			// no err, key exists
			if nil == err {
				glog.V(2).Infof("azureKms - key:%s was found in vault:%s", kms.KeyName, kms.vaultUri)
				return true, nil
			}

			// Host is not there
			if strings.Contains(err.Error(), "no such host") {
				glog.V(2).Infof("azureKms - Error:%s attempting to test key:%s on vault:%s. will retry ", err, kms.KeyName, kms.vaultUri)
				return true, fmt.Errorf("azureKms - error testing key vault settings:%s", err)
			}

			// call was made but no response object
			// terminal state, access denied
			if 403 == result.Response.StatusCode {
				glog.V(2).Infof("azureKms - Access denied testing key vault settings:%s", err)
				return true, fmt.Errorf("azureKms - Access denied testing key vault settings:%s", err)
			}

			// key (or vault) does not exist
			if 404 == result.Response.StatusCode {
				// if user overrides then we are hands of
				if overrideApplied {
					glog.V(2).Infof("azureKms - Error testing key vault user-overridden settings:%s ", err)
					return true, fmt.Errorf("Key vault defaults were applied, but testing the setting failed with: %s", err)
				}

				glog.V(2).Infof("azureKms - key:%s on vault %s was not found, will create", kms.KeyName, kms.vaultUri)
				// Creation logic
				var keySize int32 = 4096
				ops := []keyvault.JSONWebKeyOperation{keyvault.Encrypt, keyvault.Decrypt}
				// Key does not exist, lets created it
				keyCreateParameters := keyvault.KeyCreateParameters{
					Kty:     keyvault.RSA,
					KeySize: &keySize,
					KeyOps:  &ops,
				}

				kb, err := az.KeyVaultClient.CreateKey(kms.vaultUri, kms.KeyName, keyCreateParameters)
				if nil == err {
					glog.V(2).Infof("azureKms - key:%s on vault %s created", kms.KeyName, kms.vaultUri)
					return true, nil // no more work needed
				}

				// terminal state, access denied
				if 403 == kb.Response.StatusCode {
					glog.V(2).Infof("azureKms - Access denied creating key:%s on vault %s:%s", err)
					return true, fmt.Errorf("azureKms - Access denied creating key:%s on vault %s:%s", err)
				}

				// an call was made, response we have but its unknown, we retry
				glog.V(2).Infof("azureKms - Error:%s attempting to create key:%s on vault:%s ", err, kms.KeyName, kms.vaultUri)
				return false, nil
			}

			// everything else, we retry
			glog.V(2).Infof("azureKms - Error:%s attempting to test key:%s on vault:%s. will retry ", err, kms.KeyName, kms.vaultUri)
			return false, nil
		})

		if err != nil {
			glog.V(2).Infof("Error attempting to test keyvault configuration %s", err)
			return nil, err // This will return time out but the above log should suffice
		}
		return kms, nil
	})
}

func (kms *azureKms) Encrypt(data []byte) (string, error) {
	az := kms.az
	value := base64.RawURLEncoding.EncodeToString(data)
	parameters := keyvault.KeyOperationsParameters{
		Algorithm: keyvault.RSA15,
		Value:     &value,
	}

	result, err := az.KeyVaultClient.Encrypt(kms.vaultUri, kms.KeyName, kms.KeyVersion, parameters)
	if err != nil {
		return "", err
	}

	return *result.Result, nil
}

func (kms *azureKms) Decrypt(data string) ([]byte, error) {
	az := kms.az
	parameters := keyvault.KeyOperationsParameters{
		Algorithm: keyvault.RSA15,
		Value:     &data,
	}

	result, err := az.KeyVaultClient.Decrypt(kms.vaultUri, kms.KeyName, kms.KeyVersion, parameters)
	if err != nil {
		return nil, err
	}

	bytes, err := base64.RawURLEncoding.DecodeString(*result.Result)
	if err != nil {
		return nil, err
	}
	return bytes, nil
}
