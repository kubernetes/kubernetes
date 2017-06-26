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
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	dekMapName = "dek-map"
)

type keyStore struct {
	configmaps configmap.Registry
	lock       sync.RWMutex
}

// NewKeyStore returns a KMSStorage instance which uses etcd to store KMS specific information as a configmap,
// but under a different path on the disk.
func NewKeyStore(config *storagebackend.Config, dekPrefix string) value.KMSStorage {
	dekOpts := generic.RESTOptions{StorageConfig: config, Decorator: generic.UndecoratedStorage, ResourcePrefix: dekPrefix}
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
func (p *keyStore) StoreNewDEKs(newDEKs map[string]string) error {
	// This function is invoked only by Rotate() calls, which are already lock protected.
	// TODO(sakshams): Investigate if locks are really needed here.
	p.lock.Lock()
	defer p.lock.Unlock()

	ctx := genericapirequest.NewDefaultContext()
	cfg := &api.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dekMapName,
			Namespace: metav1.NamespaceDefault,
		},
		Data: newDEKs,
	}
	_, err := p.configmaps.UpdateConfigMap(ctx, cfg)
	return err
}
