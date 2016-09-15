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
	"sync"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util/sets"
)

type manager struct {
	// kubeClient for api server
	kubeClient clientset.Interface
	// Store secrets
	secretStore cache.Store
	// Watching will exit when stopCh is closed.
	stopCh chan struct{}
	// old pullSecretNames
	prevPullSecretNames sets.String
	// current pullSecretNames
	pullSecretNames sets.String
	// Mutex
	lock sync.Mutex
	// record if the first watching
	notFirst bool
}

type Manager interface {
	// Watching secrets bounded to pods.
	Watching([]*api.Pod)
	// GetPullSecretsForPod inspects the Pod and retrieves the referenced secrets.
	GetPullSecretsForPod(pod *api.Pod) ([]api.Secret, error)
}

func NewManager(kubeClient clientset.Interface) Manager {
	secretStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	return &manager{
		kubeClient,
		secretStore,
		make(chan struct{}),
		sets.NewString(),
		sets.NewString(),
		sync.Mutex{},
		false,
	}
}

// Watching secrets bouned to pods from apiserver
// as the secrets changed, will start a new watch
func (m *manager) Watching(pods []*api.Pod) {
	if m.kubeClient == nil {
		glog.Infof("Kubernetes client is nil, not starting secret manager.")
		return
	}
	m.lock.Lock()
	defer m.lock.Unlock()
	m.pullSecretNames = sets.NewString()
	for _, pod := range pods {
		for _, secretRef := range pod.Spec.ImagePullSecrets {
			m.pullSecretNames.Insert(secretRef.Name)
		}
	}
	if m.pullSecretNames.Equal(m.prevPullSecretNames) {
		return //no changes
	}
	if m.notFirst {
		close(m.stopCh) //stop previous watching
	}
	m.stopCh = make(chan struct{})
	var value string
	for secretName := range m.pullSecretNames {
		value += "," + secretName
	}
	m.prevPullSecretNames = m.pullSecretNames
	//Now apiserver does not support watching resouses by one field with mult-values
	//could support?
	secretLW := cache.NewListWatchFromClient(m.kubeClient.(*clientset.Clientset).CoreClient, "secrets", api.NamespaceAll, fields.OneTermEqualSelector("metadata.name", value))
	go cache.NewReflector(secretLW, &api.Secret{}, m.secretStore, 0).RunUntil(m.stopCh)
	m.notFirst = true
}

func (m *manager) GetPullSecretsForPod(pod *api.Pod) ([]api.Secret, error) {
	pullSecrets := []api.Secret{}

	for _, secretRef := range pod.Spec.ImagePullSecrets {
		var key string = secretRef.Name
		if len(pod.Namespace) > 0 {
			key = pod.Namespace + "/" + secretRef.Name
		}
		item, exists, err := m.secretStore.GetByKey(key)

		if !exists || err != nil {
			glog.Warningf("Unable to get pull secret by key %v:%v.", key, err)
			continue
		}
		pullSecret := item.(*api.Secret)
		pullSecrets = append(pullSecrets, *pullSecret)
	}

	return pullSecrets, nil
}
