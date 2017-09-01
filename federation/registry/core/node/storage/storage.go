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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/client-go/rest"
	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	regproxy "k8s.io/kubernetes/federation/registry/proxy"
	"k8s.io/kubernetes/pkg/api"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

func restClientFunc(kubeClientset kubeclient.Interface) rest.Interface {
	return kubeClientset.Core().RESTClient()
}

// REST implements a REST Storage for nodes.
type REST struct {
	*regproxy.Store
}

// NewREST returns REST storage objects that will work against nodes and nodes/status.
func NewREST(optsGetter generic.RESTOptionsGetter, fedClient fedclient.Interface) (*REST, *StatusREST) {
	store := &regproxy.Store{
		NewFunc:           func() runtime.Object { return &api.Node{} },
		NewListFunc:       func() runtime.Object { return &api.NodeList{} },
		RESTClientFunc:    restClientFunc,
		QualifiedResource: api.Resource("nodes"),
		FedClient:         fedClient,
	}

	statusStore := *store
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements a REST Storage for node status.
type StatusREST struct {
	store *regproxy.Store
}

// New implements StandardStorage.New.
func (r *StatusREST) New() runtime.Object {
	return r.store.New()
}

// Get implements StandardStorage.Get.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}
