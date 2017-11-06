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

package gce

import (
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"
	cloudkms "google.golang.org/api/cloudkms/v1"
	"google.golang.org/api/googleapi"
	gcfg "gopkg.in/gcfg.v1"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
)

const (
	// KMSServiceName is the name of the cloudkms provider registered by this cloud.
	KMSServiceName = "gcp-cloudkms"

	defaultGKMSKeyRing         = "google-container-engine"
	defaultGKMSKeyRingLocation = "global"
)

// gkmsConfig contains the GCE specific KMS configuration for setting up a KMS connection.
type gkmsConfig struct {
	Global struct {
		// location is the KMS location of the KeyRing to be used for encryption.
		// It can be found by checking the available KeyRings in the IAM UI.
		// This is not the same as the GCP location of the project.
		// +optional
		Location string `gcfg:"kms-location"`
		// keyRing is the keyRing of the hosted key to be used. The default value is "google-kubernetes".
		// +optional
		KeyRing string `gcfg:"kms-keyring"`
		// cryptoKey is the name of the key to be used for encryption of Data-Encryption-Keys.
		CryptoKey string `gcfg:"kms-cryptokey"`
	}
}

// readGCPCloudKMSConfig parses and returns the configuration parameters for Google Cloud KMS.
func readGCPCloudKMSConfig(reader io.Reader) (*gkmsConfig, error) {
	cfg := &gkmsConfig{}
	if err := gcfg.FatalOnly(gcfg.ReadInto(cfg, reader)); err != nil {
		glog.Errorf("Couldn't read Google Cloud KMS config: %v", err)
		return nil, err
	}
	return cfg, nil
}

// gkmsService provides Encrypt and Decrypt methods which allow cryptographic operations
// using Google Cloud KMS service.
type gkmsService struct {
	parentName      string
	parentHash      string
	cloudkmsService *cloudkms.Service

	// latestKeyName is the key name which was used during the last encrypt request.
	latestKeyName string
}

// getGCPCloudKMSService provides a Google Cloud KMS based implementation of envelope.Service.
func (gce *GCECloud) getGCPCloudKMSService(config io.Reader) (envelope.Service, error) {
	kmsConfig, err := readGCPCloudKMSConfig(config)
	if err != nil {
		return nil, err
	}

	// Hosting on GCE/GKE with Google KMS encryption provider
	cloudkmsService := gce.GetKMSService()

	// Set defaults for location and keyRing.
	location := kmsConfig.Global.Location
	if len(location) == 0 {
		location = defaultGKMSKeyRingLocation
	}
	keyRing := kmsConfig.Global.KeyRing
	if len(keyRing) == 0 {
		keyRing = defaultGKMSKeyRing
	}

	cryptoKey := kmsConfig.Global.CryptoKey
	if len(cryptoKey) == 0 {
		return nil, fmt.Errorf("missing cryptoKey for cloudprovided KMS: " + KMSServiceName)
	}

	parentName := fmt.Sprintf("projects/%s/locations/%s", gce.projectID, location)

	// Create the keyRing if it does not exist yet
	_, err = cloudkmsService.Projects.Locations.KeyRings.Create(parentName,
		&cloudkms.KeyRing{}).KeyRingId(keyRing).Do()
	if err != nil && unrecoverableCreationError(err) {
		return nil, err
	}
	parentName = parentName + "/keyRings/" + keyRing

	// Create the cryptoKey if it does not exist yet
	_, err = cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.Create(parentName,
		&cloudkms.CryptoKey{
			Purpose: "ENCRYPT_DECRYPT",
		}).CryptoKeyId(cryptoKey).Do()
	if err != nil && unrecoverableCreationError(err) {
		return nil, err
	}

	parentName = parentName + "/cryptoKeys/" + cryptoKey

	hasher := sha1.New()
	hasher.Write([]byte(parentName))

	service := &gkmsService{
		parentName:      parentName,
		parentHash:      base64.URLEncoding.EncodeToString(hasher.Sum(nil)),
		cloudkmsService: cloudkmsService,
	}

	// Sanity check before startup. For non-GCP clusters, the user's account may not have permissions to create
	// the key. We need to verify the existence of the key before apiserver startup.
	// This also sets the latestKeyVersion known to this transformer.
	_, err = service.Encrypt([]byte("test"))
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt data using Google cloudkms, using key %s. Ensure that the keyRing and cryptoKey exist. Got error: %v", parentName, err)
	}

	return service, nil
}

// Decrypt decrypts a base64 representation of ("<parent-name>:<key-used-to-encrypt>:<encrypted bytes>").
func (t *gkmsService) Decrypt(data string) ([]byte, error) {
	// Extract the ciphertext and key name from the on-disk data. Verify that the data was encrypted
	// by this google-cloudkms key.
	encData, _, err := t.parseDataWithHeader(data)
	if err != nil {
		return nil, err
	}

	resp, err := t.cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Decrypt(t.parentName, &cloudkms.DecryptRequest{
			Ciphertext: encData,
		}).Do()
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(resp.Plaintext)
}

// Encrypt encrypts bytes, and returns base64 representation of ("<parent-name>:<key-used-to-encrypt>:<encrypted bytes>").
func (t *gkmsService) Encrypt(data []byte) (string, error) {
	resp, err := t.cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Encrypt(t.parentName, &cloudkms.EncryptRequest{
			Plaintext: base64.StdEncoding.EncodeToString(data),
		}).Do()
	if err != nil {
		return "", err
	}

	t.latestKeyName = resp.Name
	return t.createDataWithHeader(resp.Name, resp.Ciphertext), nil
}

// createDataWithHeader converts the ciphertext to on-disk representation by storing
// the parent key name's hash with it, along with the full key name used for encryption.
// It is assumed that data and keyName do not contain the ":" character.
func (t *gkmsService) createDataWithHeader(keyVersionName, data string) string {
	return t.parentHash + ":" + keyVersionName + ":" + data
}

// parseDataWithHeader parses the on-disk representation of data. It verifies the parent name
// used to encrypt the data with the parent name of this transformer. It returns the actual
// cipher text, and the key version (full path) used to encrypt the data.
func (t *gkmsService) parseDataWithHeader(data string) (string, string, error) {
	dataChunks := strings.SplitN(data, ":", 3)
	if len(dataChunks) != 3 {
		return "", "", fmt.Errorf("invalid data encountered for decryption: %q. Missing key's parent-hash or key-name", data)
	}

	if t.parentHash != dataChunks[0] {
		key, err := base64.StdEncoding.DecodeString(dataChunks[0])
		if err != nil {
			return "", "", fmt.Errorf("invalid base64 representation of key's parent name encountered with encrypted data: %q", dataChunks[0])
		}
		return "", "", fmt.Errorf("data not encrypted by this google-cloudkms key. Key used: %q", key)
	}
	return dataChunks[2], dataChunks[1], nil
}

// unrecoverableCreationError decides if Kubernetes should ignore the encountered Google KMS
// error. Only to be used for errors seen while creating a KeyRing or CryptoKey.
func unrecoverableCreationError(err error) bool {
	apiError, isAPIError := err.(*googleapi.Error)
	// 409 means the object exists.
	// 403 means we do not have permission to create the object, the user must do it.
	// Else, it is an unrecoverable error.
	if !isAPIError || (apiError.Code != 409 && apiError.Code != 403) {
		return true
	}
	return false
}
