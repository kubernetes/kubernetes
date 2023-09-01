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

package services

import (
	"context"
	"time"

	"github.com/kcp-dev/logicalcluster/v3"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// ncName is the name of namespace controller
	ncName = "namespace-controller"
	// ncResyncPeriod is resync period of the namespace controller
	ncResyncPeriod = 5 * time.Minute
	// ncConcurrency is concurrency of the namespace controller
	ncConcurrency = 2
)

// NamespaceController is a server which manages namespace controller.
type NamespaceController struct {
	host   string
	stopCh chan struct{}
}

// NewNamespaceController creates a new namespace controller.
func NewNamespaceController(host string) *NamespaceController {
	return &NamespaceController{host: host, stopCh: make(chan struct{})}
}

// Start starts the namespace controller.
func (n *NamespaceController) Start(ctx context.Context) error {
	config := restclient.AddUserAgent(&restclient.Config{
		Host:        n.host,
		BearerToken: framework.TestContext.BearerToken,
		TLSClientConfig: restclient.TLSClientConfig{
			Insecure: true,
		},
	}, ncName)

	// the namespace cleanup controller is very chatty.  It makes lots of discovery calls and then it makes lots of delete calls.
	config.QPS = 50
	config.Burst = 200

	client, err := clientset.NewForConfig(config)
	if err != nil {
		return err
	}
	metadataClient, err := metadata.NewForConfig(config)
	if err != nil {
		return err
	}
	discoverResourcesFn := func(clusterName logicalcluster.Name) ([]*metav1.APIResourceList, error) {
		return client.Discovery().ServerPreferredNamespacedResources()
	}
	informerFactory := informers.NewSharedInformerFactory(client, ncResyncPeriod)

	nc := namespacecontroller.NewNamespaceController(
		ctx,
		client,
		metadataClient,
		discoverResourcesFn,
		informerFactory.Core().V1().Namespaces(),
		ncResyncPeriod, v1.FinalizerKubernetes,
	)
	informerFactory.Start(n.stopCh)
	go nc.Run(ctx, ncConcurrency)
	return nil
}

// Stop stops the namespace controller.
func (n *NamespaceController) Stop() error {
	close(n.stopCh)
	return nil
}

// Name returns the name of namespace controller.
func (n *NamespaceController) Name() string {
	return ncName
}
