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

package transformhelpers

import (
	"k8s.io/kubernetes/pkg/api"

	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/kms"
)

// KMSFactory provides shared instances of:
// 1. Cloud provider in use
// 2. Google KMS service
// 3. KMS Storage wrapper
// They are used for cases with multiple transformers.
type KMSFactory struct {
	getGCECloud func() (*kms.GCECloud, error)

	gkmsService value.KMSService

	storage value.KMSStorage
}

// NewKMSFactory creates a key store for KMS data, and returns a KMSFactory instance.
func NewKMSFactory(storageFactory serverstorage.StorageFactory, getGCECloud func() (*kms.GCECloud, error)) (*KMSFactory, error) {
	storageConfig, err := storageFactory.NewConfig(api.Resource("configmap"))
	if err != nil {
		return nil, err
	}
	plainKeyStore := NewKeyStore(storageConfig, "kms")
	return &KMSFactory{
		getGCECloud: getGCECloud,
		storage:     plainKeyStore,
	}, nil
}

// NewKMSFactoryWithStorageAndGKMS allows creating a mockup KMSFactory for unit tests.
func NewKMSFactoryWithStorageAndGKMS(storage value.KMSStorage, gkmsService value.KMSService) *KMSFactory {
	return &KMSFactory{
		storage:     storage,
		gkmsService: gkmsService,
	}
}

// GetGoogleKMSTransformer creates a Google KMS service which can Encrypt and Decrypt data.
// Ensures only one instance of the service at any time.
// TODO(sakshams): Does not handle the case where two different requests for a KMS service are for
// different projects, locations etc.
func (kmsFactory *KMSFactory) GetGoogleKMSTransformer(projectID, location, keyRing, cryptoKey string) (value.Transformer, error) {
	if kmsFactory.gkmsService == nil {
		cloud, err := kmsFactory.getGCECloud()
		if err != nil {
			return nil, err
		}
		kmsFactory.gkmsService, err = kms.NewGoogleKMSService(projectID, location, keyRing, cryptoKey, cloud)
		if err != nil {
			return nil, err
		}
	}
	return kms.NewKMSTransformer(kmsFactory.gkmsService, kmsFactory.storage)
}
