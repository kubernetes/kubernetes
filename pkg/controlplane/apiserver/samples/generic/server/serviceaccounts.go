/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/informers"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// clientGetter implements ServiceAccountTokenGetter using a factory function
type clientGetter struct {
	secretLister         v1listers.SecretLister
	serviceAccountLister v1listers.ServiceAccountLister
}

// genericTokenGetter returns a ServiceAccountTokenGetter that does not depend
// on pods and nodes.
func genericTokenGetter(factory informers.SharedInformerFactory) serviceaccount.ServiceAccountTokenGetter {
	return clientGetter{secretLister: factory.Core().V1().Secrets().Lister(), serviceAccountLister: factory.Core().V1().ServiceAccounts().Lister()}
}

func (c clientGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	return c.serviceAccountLister.ServiceAccounts(namespace).Get(name)
}

func (c clientGetter) GetPod(namespace, name string) (*v1.Pod, error) {
	return nil, apierrors.NewNotFound(v1.Resource("pods"), name)
}

func (c clientGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.secretLister.Secrets(namespace).Get(name)
}

func (c clientGetter) GetNode(name string) (*v1.Node, error) {
	return nil, apierrors.NewNotFound(v1.Resource("nodes"), name)
}
