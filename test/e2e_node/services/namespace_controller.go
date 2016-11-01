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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
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
	stopCh chan struct{}
}

// NewNamespaceController creates a new namespace controller.
func NewNamespaceController() *NamespaceController {
	return &NamespaceController{stopCh: make(chan struct{})}
}

// Start starts the namespace controller.
func (n *NamespaceController) Start() error {
	// Use the default QPS
	config := restclient.AddUserAgent(&restclient.Config{Host: framework.TestContext.Host}, ncName)
	client, err := clientset.NewForConfig(config)
	if err != nil {
		return err
	}
	clientPool := dynamic.NewClientPool(config, registered.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	gvrFn := client.Discovery().ServerPreferredNamespacedResources
	nc := namespacecontroller.NewNamespaceController(client, clientPool, gvrFn, ncResyncPeriod, api.FinalizerKubernetes)
	go nc.Run(ncConcurrency, n.stopCh)
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
