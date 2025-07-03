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

package app

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/features"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgofeaturegate "k8s.io/client-go/features"
	svm "k8s.io/kubernetes/pkg/controller/storageversionmigrator"
)

func newStorageVersionMigratorControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:    names.StorageVersionMigratorController,
		aliases: []string{"svm"},
		initFunc: func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Runnable, error) {
			if !utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) ||
				!clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InformerResourceVersion) {
				return nil, nil
			}
			return newSVMController(ctx, controllerContext, controllerName)
		},
	}
}

type svmController struct {
	*svm.SVMController
	controllerContext ControllerContext
	controllerName    string
	config            *rest.Config
	client            kubernetes.Interface
}

func newSVMController(ctx context.Context, controllerContext ControllerContext, controllerName string) (*svmController, error) {
	c := &svmController{
		controllerContext: controllerContext,
		controllerName:    controllerName,
	}

	if !controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, fmt.Errorf("storage version migrator requires garbage collector")
	}

	// svm controller can make a lot of requests during migration, keep it fast
	config := controllerContext.ClientBuilder.ConfigOrDie(c.controllerName)
	config.QPS *= 20
	config.Burst *= 100

	client := controllerContext.ClientBuilder.ClientOrDie(controllerName)
	informer := controllerContext.InformerFactory.Storagemigration().V1alpha1().StorageVersionMigrations()

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	c.SVMController = svm.NewSVMController(
		ctx,
		client,
		dynamicClient,
		informer,
		controllerName,
		controllerContext.RESTMapper,
		controllerContext.GraphBuilder,
	)
	c.config = config
	c.client = client
	return c, nil
}

func (c *svmController) Start(ctx context.Context) error {
	informer := c.controllerContext.InformerFactory.Storagemigration().V1alpha1().StorageVersionMigrations()

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(c.config)
	if err != nil {
		return err
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		svm.NewResourceVersionController(
			ctx,
			c.client,
			discoveryClient,
			metadata.NewForConfigOrDie(c.config),
			informer,
			c.controllerContext.RESTMapper,
		).Run(ctx)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		c.SVMController.Run(ctx)
	}()

	wg.Wait()
	return nil
}
