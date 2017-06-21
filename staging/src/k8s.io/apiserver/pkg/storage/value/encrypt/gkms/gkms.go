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

// Package gms transforms values for storage at rest using Google KMS.
package gkms

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
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
)

type gkmsTransformer struct {
	ParentName      string
	CloudkmsService *cloudkms.Service
}

func NewGoogleKMSTransformer(projectID, location, keyRing, cryptoKey string, cloudProvider *kubeoptions.CloudProviderOptions) (value.Transformer, error) {
	var cloudkmsService *cloudkms.Service
	var err error

	cloud, err := cloudprovider.InitCloudProvider(cloudProvider.CloudProvider, cloudProvider.CloudConfigFile)
	if err != nil {
		return nil, err
	}

	// Safe when cloud is nil too.
	if gke, ok := cloud.(*gce.GCECloud); ok {
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

	return &gkmsTransformer{parentName, cloudkmsService}, nil
}

func (t *gkmsTransformer) TransformFromStorage(data []byte, context value.Context) ([]byte, bool, error) {
	resp, err := t.CloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Decrypt(t.ParentName, &cloudkms.DecryptRequest{
			Ciphertext: base64.StdEncoding.EncodeToString(data),
		}).Do()
	if err != nil {
		return nil, false, err
	}
	result, err := base64.StdEncoding.DecodeString(resp.Plaintext)
	return result, false, err
}

func (t *gkmsTransformer) TransformToStorage(data []byte, context value.Context) ([]byte, error) {
	resp, err := t.CloudkmsService.Projects.Locations.KeyRings.CryptoKeys.
		Encrypt(t.ParentName, &cloudkms.EncryptRequest{
			Plaintext: base64.StdEncoding.EncodeToString(data),
		}).Do()
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.DecodeString(resp.Ciphertext)
}
