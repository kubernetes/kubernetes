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

package storage

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	apirest "k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/rest"
	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	regproxy "k8s.io/kubernetes/federation/registry/proxy"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

func restClientFunc(kubeClientset kubeclient.Interface) rest.Interface {
	return kubeClientset.Extensions().RESTClient()
}

// REST implements a REST Storage for replicasets
type REST struct {
	*regproxy.Store
}

// NewREST returns a REST storage object that will work against ReplicaSet
// status and scale proxy is not proxied
func NewREST(optsGetter generic.RESTOptionsGetter, fedClient fedclient.Interface, fedStore apirest.StandardStorage) *REST {
	store := &regproxy.Store{
		NewFunc:           func() runtime.Object { return &extensions.ReplicaSet{} },
		NewListFunc:       func() runtime.Object { return &extensions.ReplicaSetList{} },
		RESTClientFunc:    restClientFunc,
		QualifiedResource: api.Resource("replicasets"),
		NamespaceScoped:   true,
		FedClient:         fedClient,
		FedStore:          fedStore,
	}

	return &REST{store}
}
