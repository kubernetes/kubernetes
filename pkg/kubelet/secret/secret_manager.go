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
package secret

import (
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
)

type manager struct {
	// kubeClient for api server
	kubeClient clientset.Interface
	// Store secrets
	secretStore cache.Store
}

type Manager interface {
	// Start secrets watching.
	Start()
	// getSecretsForPod inspects the Pod and retrieves the referenced secrets.
	GetSecretsForPod(pod *api.Pod) ([]api.Secret, error)
}

func NewManager(kubeClient clientset.Interface) Manager {
	secretStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	return &manager{
		kubeClient,
		secretStore,
	}
}

func (m *manager) Start() {
	if m.kubeClient == nil {
		glog.Infof("Kubernetes client is nil, not starting secret manager.")
		return
	}
	secretLW := cache.NewListWatchFromClient(m.kubeClient.(*clientset.Clientset).CoreClient, "secrets", api.NamespaceAll, fields.Everything())
	go cache.NewReflector(secretLW, &api.Secret{}, m.secretStore, 0).Run()
}

func (m *manager) GetSecretsForPod(pod *api.Pod) ([]api.Secret, error) {
	secrets := []api.Secret{}

	for _, secretRef := range pod.Spec.ImagePullSecrets {
		var key string = secretRef.Name
		if len(pod.Namespace) > 0 {
			key = pod.Namespace + "/" + secretRef.Name
		}
		item, exits, err := m.secretStore.GetByKey(key)
		if err != nil || !exits {
			glog.Warningf("Unable to get secret by key %v:%v.", key, err)
			continue
		}
		secret := item.(*api.Secret)
		secrets = append(secrets, *secret)
	}

	return secrets, nil
}
