/*
Copyright 2014 The Kubernetes Authors.

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

// +kcp-code-generator:skip

package serviceaccount

import (
	"context"

	kcpkubernetesclientset "github.com/kcp-dev/client-go/kubernetes"
	kcpcorev1listers "github.com/kcp-dev/client-go/listers/core/v1"
	"github.com/kcp-dev/logicalcluster/v3"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

func NewClusterGetterFromClient(client kcpkubernetesclientset.ClusterInterface, secretLister kcpcorev1listers.SecretClusterLister, serviceAccountLister kcpcorev1listers.ServiceAccountClusterLister /*podLister kcpcorev1listers.PodClusterLister*/) serviceaccount.ServiceAccountTokenClusterGetter {
	return &serviceAccountTokenClusterGetter{
		client:               client,
		secretLister:         secretLister,
		serviceAccountLister: serviceAccountLister,
	}
}

type serviceAccountTokenClusterGetter struct {
	client               kcpkubernetesclientset.ClusterInterface
	secretLister         kcpcorev1listers.SecretClusterLister
	serviceAccountLister kcpcorev1listers.ServiceAccountClusterLister
	podLister            kcpcorev1listers.PodClusterLister
}

func (s *serviceAccountTokenClusterGetter) Cluster(name logicalcluster.Name) serviceaccount.ServiceAccountTokenGetter {
	return NewGetterFromClient(
		s.client.Cluster(name.Path()),
		s.secretLister.Cluster(name),
		s.serviceAccountLister.Cluster(name),
	)
}

// clientGetter implements ServiceAccountTokenGetter using a clientset.Interface
type clientGetter struct {
	client               clientset.Interface
	secretLister         v1listers.SecretLister
	serviceAccountLister v1listers.ServiceAccountLister
	podLister            v1listers.PodLister
	nodeLister           v1listers.NodeLister
}

// NewGetterFromClient returns a ServiceAccountTokenGetter that
// uses the specified client to retrieve service accounts, pods, secrets and nodes.
// The client should NOT authenticate using a service account token
// the returned getter will be used to retrieve, or recursion will result.
func NewGetterFromClient(c clientset.Interface, secretLister v1listers.SecretLister, serviceAccountLister v1listers.ServiceAccountLister) serviceaccount.ServiceAccountTokenGetter {
	return clientGetter{
		client:               c,
		secretLister:         secretLister,
		serviceAccountLister: serviceAccountLister,
	}
}

func (c clientGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	if serviceAccount, err := c.serviceAccountLister.ServiceAccounts(namespace).Get(name); err == nil {
		return serviceAccount, nil
	}
	return c.client.CoreV1().ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
}

func (c clientGetter) GetPod(namespace, name string) (*v1.Pod, error) {
	if c.podLister == nil {
		return nil, apierrors.NewNotFound(v1.Resource("pods"), name)
	}
	return c.client.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{})
}

func (c clientGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	if secret, err := c.secretLister.Secrets(namespace).Get(name); err == nil {
		return secret, nil
	}
	return c.client.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
}

func (c clientGetter) GetNode(name string) (*v1.Node, error) {
	// handle the case where the node lister isn't set due to feature being disabled
	if c.nodeLister == nil {
		return nil, apierrors.NewNotFound(v1.Resource("nodes"), name)
	}
	if node, err := c.nodeLister.Get(name); err == nil {
		return node, nil
	}
	return c.client.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
}
