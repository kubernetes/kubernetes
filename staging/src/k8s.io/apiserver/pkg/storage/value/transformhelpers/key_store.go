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
	"strings"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	randutil "k8s.io/apimachinery/pkg/util/rand"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/core/configmap"
	configmapregistry "k8s.io/kubernetes/pkg/registry/core/configmap"
	configmapstore "k8s.io/kubernetes/pkg/registry/core/configmap/storage"
)

const (
	// dekMapName is the name of configmap which stores the DEKs.
	dekMapName = "dek-map"
	// keyNameLength is the length of names for DEKs.
	keyNameLength = 5
)

type keyStore struct {
	configmaps configmap.Registry
	lock       sync.RWMutex
}

// NewKeyStore returns a KMSStorage instance which uses etcd to store KMS specific information as a configmap,
// but under a different path on the disk.
func NewKeyStore(config *storagebackend.Config, dekPrefix string) value.KMSStorage {
	// We need to disable any encryption on this keystore
	newConfig := *config
	newConfig.Transformer = nil
	dekOpts := generic.RESTOptions{StorageConfig: &newConfig, Decorator: generic.UndecoratedStorage, ResourcePrefix: dekPrefix}
	return &keyStore{configmaps: configmapregistry.NewRegistry(configmapstore.NewREST(dekOpts))}
}

// Setup creates the empty configmap on disk if it did not already exist.
func (p *keyStore) Setup() error {
	ctx := genericapirequest.NewDefaultContext()

	_, err := p.GetAllDEKs()
	// TODO(sakshams): Should we do this if err is 404, or should we do this as long as err is not nil?
	if err != nil {
		// We need to create the configmap
		cfg := &api.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      dekMapName,
				Namespace: metav1.NamespaceDefault,
			},
			Data: map[string]string{},
		}
		_, err := p.configmaps.CreateConfigMap(ctx, cfg)
		if err != nil {
			return err
		}
	}

	return nil
}

// GetAllDEKs reads and returns all available DEKs from etcd as a map.
func (p *keyStore) GetAllDEKs() (map[string]string, error) {
	ctx := genericapirequest.NewDefaultContext()
	cfg, err := p.configmaps.GetConfigMap(ctx, dekMapName, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return cfg.Data, nil
}

// StoreNewDEKs writes the provided DEKs to disk.
func (p *keyStore) StoreNewDEK(encDEK string) error {
	// This function is invoked only by Rotate() calls, which are already lock protected.
	// TODO(sakshams): Investigate if locks are really needed here.
	p.lock.Lock()
	defer p.lock.Unlock()

	ctx := genericapirequest.NewDefaultContext()
	cfg, err := p.configmaps.GetConfigMap(ctx, dekMapName, &metav1.GetOptions{})
	if err != nil {
		return err
	}

	updater := func(ctx genericapirequest.Context, _ runtime.Object, oldObj runtime.Object) (runtime.Object, error) {
		newDEKs := map[string]string{}
		// Re-create the map because unsafe to modify map while iterating over it
		for dekname, dek := range oldObj.(*api.ConfigMap).Data {
			// Remove the identifying prefix in front of the primary key.
			if strings.HasPrefix(dekname, "-") {
				dekname = dekname[1:]
			}
			newDEKs[dekname] = dek
		}

		// Get a new and unique name
		newDEKname := GenerateName(newDEKs)

		newDEKs["-"+newDEKname] = encDEK
		oldObj.(*api.ConfigMap).Data = newDEKs
		return oldObj, nil
	}

	_, err = p.configmaps.UpdateConfigMap(ctx, cfg, updater)
	return err
}

// GenerateName generates a unique new name for a DEK. Exposed for running encryptionconfig_test.
func GenerateName(existingNames map[string]string) string {
	name := randutil.String(keyNameLength)

	_, ok := existingNames[name]
	for ok {
		name := randutil.String(keyNameLength)
		_, ok = existingNames[name]
	}

	return name
}
