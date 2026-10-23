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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgofeaturegate "k8s.io/client-go/features"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	svm "k8s.io/kubernetes/pkg/controller/storageversionmigrator"
	"k8s.io/kubernetes/pkg/features"
)

func newStorageVersionMigratorControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.StorageVersionMigratorController,
		aliases:     []string{"svm"},
		constructor: newSVMController,
	}
}

func newSVMController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) ||
		!clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InformerResourceVersion) {
		return nil, nil
	}

	if !controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, fmt.Errorf("storage version migrator requires garbage collector")
	}

	if !clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.InOrderInformers) {
		err := fmt.Errorf("storage version migrator requires the InOrderInformers feature gate to be enabled")
		return nil, err
	}

	client, err := controllerContext.NewClient(controllerName)
	if err != nil {
		return nil, err
	}

	informer := controllerContext.InformerFactory.Storagemigration().V1().StorageVersionMigrations()

	// svm controller can make a lot of requests during migration, keep it fast
	clientConnection := controllerContext.ComponentConfig.Generic.ClientConnection
	svmClientBuilder := controllerContext.ClientBuilder.WithRateLimits(clientConnection.QPS*20, int(clientConnection.Burst)*100)

	crdClientset, err := controllerContext.NewApiextensionsClient(controllerName, svmClientBuilder)
	if err != nil {
		return nil, err
	}

	dynamicClient, err := svmClientBuilder.DynamicClient(controllerName)
	if err != nil {
		return nil, err
	}

	discoveryClient, err := svmClientBuilder.DiscoveryClient(controllerName)
	if err != nil {
		return nil, err
	}

	metaClient, err := svmClientBuilder.MetadataClient(controllerName)
	if err != nil {
		return nil, fmt.Errorf("failed to create metadata client for %s: %w", controllerName, err)
	}

	return newControllerLoop(concurrentRun(
		svm.NewSVMController(
			ctx,
			client,
			dynamicClient,
			informer,
			controllerName,
			controllerContext.RESTMapper,
			controllerContext.GraphBuilder,
		).Run,
		svm.NewResourceVersionController(
			ctx,
			client,
			discoveryClient,
			metaClient,
			informer,
			controllerContext.RESTMapper,
		).Run,
		svm.NewCustomResourceController(
			ctx,
			client,
			informer,
			crdClientset.ApiextensionsV1().CustomResourceDefinitions(),
		).Run,
	), controllerName), nil
}
