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
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"

	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/kms"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
)

type CloudAndStorage struct {
	cloud             *cloudprovider.Interface
	cloudProviderOpts *kubeoptions.CloudProviderOptions

	gkmsService value.KmsService

	storage value.KmsStorage
}

func NewCloudAndStorageGetterSetter(s *options.ServerRunOptions, storageFactory serverstorage.StorageFactory) (*CloudAndStorage, error) {
	storageConfig, err := storageFactory.NewConfig(api.Resource("configmap"))
	if err != nil {
		return nil, err
	}
	plainKeyStore := NewKeyStore(storageConfig, "gkms")
	return &CloudAndStorage{
		cloudProviderOpts: s.CloudProvider,
		storage:           plainKeyStore,
	}, nil
}

func (cs *CloudAndStorage) GetCloud() (*cloudprovider.Interface, error) {
	if cs.cloud != nil {
		return cs.cloud, nil
	}
	cloud, err := cloudprovider.InitCloudProvider(cs.cloudProviderOpts.CloudProvider, cs.cloudProviderOpts.CloudConfigFile)
	if err != nil {
		return nil, err
	}
	cs.cloud = &cloud
	return cs.cloud, nil
}

func (cs *CloudAndStorage) GetGoogleKMSService(projectID, location, keyRing, cryptoKey string) (value.KmsService, error) {
	var err error
	if cs.gkmsService != nil {
		return cs.gkmsService, nil
	}
	cloud, err := cs.GetCloud()
	if err != nil {
		return nil, err
	}
	cs.gkmsService, err = kms.NewGoogleKMSTransformer(projectID, location, keyRing, cryptoKey, cloud, cs.storage)
	return cs.gkmsService, err
}
