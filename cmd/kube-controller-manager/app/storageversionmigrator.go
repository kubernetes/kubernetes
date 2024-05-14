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

	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/metadata"
	"k8s.io/controller-manager/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/features"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgofeaturegate "k8s.io/client-go/features"
	svm "k8s.io/kubernetes/pkg/controller/storageversionmigrator"
)

func newStorageVersionMigratorControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.StorageVersionMigratorController,
		aliases:  []string{"svm"},
		initFunc: startSVMController,
	}
}

func startSVMController(
	ctx context.Context,
	controllerContext ControllerContext,
	controllerName string,
) (controller.Interface, bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) ||
		!clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InformerResourceVersion) {
		return nil, false, nil
	}

	if !controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, true, fmt.Errorf("storage version migrator requires garbage collector")
	}

	config := controllerContext.ClientBuilder.ConfigOrDie(controllerName)
	client := controllerContext.ClientBuilder.ClientOrDie(controllerName)
	informer := controllerContext.InformerFactory.Storagemigration().V1alpha1().StorageVersionMigrations()

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, false, err
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, false, err
	}

	go svm.NewResourceVersionController(
		ctx,
		client,
		discoveryClient,
		metadata.NewForConfigOrDie(config),
		informer,
		controllerContext.RESTMapper,
	).Run(ctx)

	svmController := svm.NewSVMController(
		ctx,
		client,
		dynamicClient,
		informer,
		controllerName,
		controllerContext.RESTMapper,
		controllerContext.GraphBuilder,
	)
	go svmController.Run(ctx)

	return svmController, true, nil
}
