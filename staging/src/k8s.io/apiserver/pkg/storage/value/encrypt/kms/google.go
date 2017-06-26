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

package kms

import (
	"context"
	"encoding/base64"
	"fmt"

	"golang.org/x/oauth2/google"
	cloudkms "google.golang.org/api/cloudkms/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
)

type gkmsTransformer struct {
	parentName      string
	cloudkmsService *cloudkms.Service
}

// NewGoogleKMSService creates a Google KMS connection and returns a KMSService instance which can encrypt and decrypt data.
func NewGoogleKMSService(projectID, location, keyRing, cryptoKey string, cloud *cloudprovider.Interface) (value.KMSService, error) {
	var cloudkmsService *cloudkms.Service
	var err error

	// Safe when cloud is nil too.
	if gke, ok := (*cloud).(*gce.GCECloud); ok {
		// Hosting on GCE/GKE with Google KMS encryption provider
		cloudkmsService = gke.GetKMSService()

		// Project ID is assumed to be the user's project unless there
		// is an override in the configuration file
		if projectID == "" {
			projectID = gke.GetProjectID()
		}

		// Default location for keys
		if location == "" {
			location = "global"
		}
	} else {
		// Outside GCE/GKE. Requires GOOGLE_APPLICATION_CREDENTIALS environment variable.
		ctx := context.Background()
		client, err := google.DefaultClient(ctx, cloudkms.CloudPlatformScope)
		if err != nil {
			return nil, err
		}
		cloudkmsService, err = cloudkms.New(client)
		if err != nil {
			return nil, err
		}
	}

	parentName := fmt.Sprintf("projects/%s/locations/%s", projectID, location)

	// Create the keyRing if it does not exist yet
	_, err = cloudkmsService.Projects.Locations.KeyRings.Create(parentName,
		&cloudkms.KeyRing{}).KeyRingId(keyRing).Do()
	if err != nil {
		apiError, ok := err.(*googleapi.Error)
		// If it was a 409, that means the keyring existed.
		// If it was a 403, we do not have permission to create the keyring, the user must do it.
		// Else, it is an unrecoverable error.
		if !ok || (apiError.Code != 409 && apiError.Code != 403) {
			return nil, err
		}
	}
	parentName = parentName + "/keyRings/" + keyRing

	// Create the cryptoKey if it does not exist yet
	_, err = cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.Create(parentName,
		&cloudkms.CryptoKey{
			Purpose: "ENCRYPT_DECRYPT",
		}).CryptoKeyId(cryptoKey).Do()
	if err != nil {
		apiError, ok := err.(*googleapi.Error)
		// If it was a 409, that means the key existed.
		// If it was a 403, we do not have permission to create the key, the user must do it.
		// Else, it is an unrecoverable error.
		if !ok || (apiError.Code != 409 && apiError.Code != 403) {
			return nil, err
		}
	}
	parentName = parentName + "/cryptoKeys/" + cryptoKey

	return &gkmsTransformer{
		parentName:      parentName,
		cloudkmsService: cloudkmsService,
	}, nil
}

// Decrypt decrypts a base64 representation of encrypted bytes.
func (t *gkmsTransformer) Decrypt(data string) ([]byte, error) {
	resp, err := t.cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Decrypt(t.parentName, &cloudkms.DecryptRequest{
			Ciphertext: data,
		}).Do()
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(resp.Plaintext)
}

// Encrypt encrypts bytes, and returns base64 representation of the ciphertext.
func (t *gkmsTransformer) Encrypt(data []byte) (string, error) {
	resp, err := t.cloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Encrypt(t.parentName, &cloudkms.EncryptRequest{
			Plaintext: base64.StdEncoding.EncodeToString(data),
		}).Do()
	if err != nil {
		return "", err
	}
	return resp.Ciphertext, nil
}
