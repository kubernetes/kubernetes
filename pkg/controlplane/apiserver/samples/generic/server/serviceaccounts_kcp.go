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
	"github.com/kcp-dev/client-go/informers"
	kcpv1listers "github.com/kcp-dev/client-go/listers/core/v1"
	"github.com/kcp-dev/logicalcluster/v3"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// clientGetter implements ServiceAccountTokenGetter using a factory function
type clientClusterGetter struct {
	secretLister         kcpv1listers.SecretClusterLister
	serviceAccountLister kcpv1listers.ServiceAccountClusterLister
	clusterName          logicalcluster.Name
}

// genericTokenGetter returns a ServiceAccountTokenGetter that does not depend
// on pods and nodes.
func genericTokenClusterGetter(factory informers.SharedInformerFactory) serviceaccount.ServiceAccountTokenClusterGetter {
	return clientClusterGetter{secretLister: factory.Core().V1().Secrets().Lister(), serviceAccountLister: factory.Core().V1().ServiceAccounts().Lister()}
}

func (c clientClusterGetter) Cluster(name logicalcluster.Name) serviceaccount.ServiceAccountTokenGetter {
	c.clusterName = name
	return c
}

func (c clientClusterGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	return c.serviceAccountLister.Cluster(c.clusterName).ServiceAccounts(namespace).Get(name)
}

func (c clientClusterGetter) GetPod(namespace, name string) (*v1.Pod, error) {
	return nil, apierrors.NewNotFound(v1.Resource("pods"), name)
}

func (c clientClusterGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.secretLister.Cluster(c.clusterName).Secrets(namespace).Get(name)
}

func (c clientClusterGetter) GetNode(name string) (*v1.Node, error) {
	return nil, apierrors.NewNotFound(v1.Resource("nodes"), name)
}
