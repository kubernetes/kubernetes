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
	"crypto/aes"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"

	"golang.org/x/oauth2/google"
	cloudkms "google.golang.org/api/cloudkms/v1"
	"google.golang.org/api/googleapi"
	randutil "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
)

const (
	keyNameLength    = 5
	primaryKeyPrefix = "-"
)

type gkmsTransformer struct {
	parentName      string
	cloudkmsService *cloudkms.Service

	transformers   map[string]value.Transformer
	primaryKeyName string

	storage value.KmsStorage

	rotateLock  sync.RWMutex
	refreshLock sync.RWMutex
}

func NewGoogleKMSTransformer(projectID, location, keyRing, cryptoKey string, cloud *cloudprovider.Interface, storage value.KmsStorage) (value.KmsService, error) {
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

	// TODO(sakshams): Change the code below to a Get followed by a create.
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

	err = storage.Setup()
	if err != nil {
		return nil, err
	}

	kmsService := &gkmsTransformer{
		parentName:      parentName,
		cloudkmsService: cloudkmsService,
		storage:         storage,
	}
	// If there are no keys, rotate(false) will create one.
	if err = kmsService.Rotate(false); err != nil {
		return nil, err
	}

	return kmsService, nil
}

// TODO(sakshams): Should be moved to kms transformer as this is common for all implementations of kms.
func (t *gkmsTransformer) Rotate(rotateIfNotEmpty bool) error {
	t.rotateLock.Lock()
	defer t.rotateLock.Unlock()

	deks, err := t.storage.GetAllDEKs()
	if err != nil {
		return err
	}

	// If this is during setup, don't rotate if a key already exists.
	if !rotateIfNotEmpty && len(deks) != 0 {
		return t.Refresh()
	}

	newDEKs := map[string]string{}
	for keyname, dek := range deks {
		// Qualify the primary key with a "-" prefix for consistent accesses and updates.
		if strings.HasPrefix(keyname, "-") {
			keyname = keyname[1:]
		}
		newDEKs[keyname] = dek
	}

	keyname := "-" + generateName(newDEKs)
	dekBytes, err := generateKey(32)
	if err != nil {
		return err
	}

	newDEKs[keyname], err = t.Encrypt(dekBytes)
	if err != nil {
		return err
	}
	t.storage.StoreNewDEKs(newDEKs)

	return t.Refresh()
}

// TODO(sakshams): Should be moved to kms transformer as this is common for all implementations of kms.
func (t *gkmsTransformer) Refresh() error {
	t.refreshLock.Lock()
	defer t.refreshLock.Unlock()

	deks, err := t.storage.GetAllDEKs()
	if err != nil {
		return err
	}
	transformers := map[string]value.Transformer{}
	primaryKeyName := ""
	for keyname, encDek := range deks {
		if strings.HasPrefix(keyname, "-") {
			// This is the primary keyname
			keyname = keyname[1:]
			primaryKeyName = keyname
		}
		dekBytes, err := t.Decrypt(encDek)
		if err != nil {
			return err
		}
		block, err := aes.NewCipher(dekBytes)
		if err != nil {
			return err
		}
		// TODO(sakshams): Define a singleton prefix transformer as well
		prefixTransformer := value.PrefixTransformer{
			Prefix:      []byte(keyname + ":"),
			Transformer: aestransformer.NewCBCTransformer(block),
		}
		transformers[keyname] = value.NewPrefixTransformers(nil, prefixTransformer)
	}
	if primaryKeyName == "" {
		return fmt.Errorf("no primary key found for kms transformer")
	}
	// Transformers need to be re-assigned before re-assigning primary key to avoid a race condition.
	// This also means that we can NOT allow deleting the primary key and adding a new one in the same operation.
	t.transformers = transformers
	t.primaryKeyName = primaryKeyName

	return nil
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

func (t *gkmsTransformer) GetReadingTransformer(keyname string) (value.Transformer, error) {
	if transformer, ok := t.transformers[keyname]; ok {
		return transformer, nil
	}
	// TODO(sakshams): Fill this up
	return nil, fmt.Errorf("did not find a transformer for key: %s", keyname)
}

func (t *gkmsTransformer) GetWritingTransformer() (value.Transformer, error) {
	if transformer, ok := t.transformers[t.primaryKeyName]; ok {
		return transformer, nil
	}
	return nil, fmt.Errorf("primary key transformer not found, keyname: %s", t.primaryKeyName)
}

func generateName(existingNames map[string]string) string {
	name := randutil.String(keyNameLength)

	_, ok := existingNames[name]
	for ok {
		name := randutil.String(keyNameLength)
		_, ok = existingNames[name]
	}

	return name
}

func generateKey(length int) ([]byte, error) {
	key := make([]byte, length)
	_, err := rand.Read(key)
	if err != nil {
		return []byte{}, err
	}

	return key, nil
}
