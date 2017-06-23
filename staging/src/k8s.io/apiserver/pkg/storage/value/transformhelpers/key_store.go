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

func NewKeyStore(config *storagebackend.Config, dekPrefix string) *keyStore {
	dekOpts := generic.RESTOptions{StorageConfig: config, Decorator: generic.UndecoratedStorage, ResourcePrefix: dekPrefix}
	return &keyStore{configmaps: configmapregistry.NewRegistry(configmapstore.NewREST(dekOpts))}
}

func (p *keyStore) Setup() error {
	ctx := genericapirequest.NewDefaultContext()

	_, err := p.GetAllDEKs()
	// TODO(sakshams): Only do this if NotFound error
	// if serr, ok := err.(*storage.StorageError); ok && serr.Code == storage.ErrCodeKeyNotFound {
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

func (p *keyStore) GetAllDEKs() (map[string]string, error) {
	ctx := genericapirequest.NewDefaultContext()
	cfg, err := p.configmaps.GetConfigMap(ctx, dekMapName, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return cfg.Data, nil
}

func (p *keyStore) StoreNewDEKs(newDEKs map[string]string) error {
	// This function is invoked only by Rotate() calls, which are already lock protected.
	// TODO(sakshams): Check if this function requires locks.
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
