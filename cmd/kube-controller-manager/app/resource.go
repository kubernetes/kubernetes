/*
Copyright The Kubernetes Authors.

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
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction"
	"k8s.io/kubernetes/pkg/controller/resourceclaim"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest"
	"k8s.io/kubernetes/pkg/features"
)

func newDeviceTaintEvictionControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.DeviceTaintEvictionController,
		constructor: newDeviceTaintEvictionController,
		requiredFeatureGates: []featuregate.Feature{
			// TODO update app.TestFeatureGatedControllersShouldNotDefineAliases when removing these feature gates.
			features.DynamicResourceAllocation,
			features.DRADeviceTaints,
		},
	}
}

func newDeviceTaintEvictionController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient(names.DeviceTaintEvictionController)
	if err != nil {
		return nil, err
	}

	deviceTaintEvictionController := devicetainteviction.New(
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Resource().V1().ResourceClaims(),
		controllerContext.InformerFactory.Resource().V1().ResourceSlices(),
		controllerContext.InformerFactory.Resource().V1beta2().DeviceTaintRules(),
		controllerContext.InformerFactory.Resource().V1().DeviceClasses(),
		controllerName,
	)
	return newControllerLoop(func(ctx context.Context) {
		if err := deviceTaintEvictionController.Run(ctx, int(controllerContext.ComponentConfig.DeviceTaintEvictionController.ConcurrentSyncs)); err != nil {
			klog.FromContext(ctx).Error(err, "Device taint processing leading to Pod eviction failed and is now paused")
		}
		<-ctx.Done()
	}, controllerName), nil
}

func newResourceClaimControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.ResourceClaimController,
		aliases:     []string{"resource-claim-controller"},
		constructor: newResourceClaimController,
		requiredFeatureGates: []featuregate.Feature{
			features.DynamicResourceAllocation, // TODO update app.TestFeatureGatedControllersShouldNotDefineAliases when removing this feature
		},
	}
}

func newResourceClaimController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("resource-claim-controller")
	if err != nil {
		return nil, err
	}

	ephemeralController, err := resourceclaim.NewController(
		klog.FromContext(ctx),
		resourceclaim.Features{
			AdminAccess:     utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess),
			PrioritizedList: utilfeature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList),
		},
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Resource().V1().ResourceClaims(),
		controllerContext.InformerFactory.Resource().V1().ResourceClaimTemplates())
	if err != nil {
		return nil, fmt.Errorf("failed to init resource claim controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		ephemeralController.Run(ctx, int(controllerContext.ComponentConfig.ResourceClaimController.ConcurrentSyncs))
	}, controllerName), nil
}

func newResourcePoolStatusRequestControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.ResourcePoolStatusRequestController,
		constructor: newResourcePoolStatusRequestController,
		requiredFeatureGates: []featuregate.Feature{
			features.DRAResourcePoolStatus,
		},
	}
}

func newResourcePoolStatusRequestController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("resourcepoolstatusrequest-controller")
	if err != nil {
		return nil, err
	}

	controller, err := resourcepoolstatusrequest.NewController(
		ctx,
		client,
		controllerContext.InformerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		controllerContext.InformerFactory.Resource().V1().ResourceSlices(),
		controllerContext.InformerFactory.Resource().V1().ResourceClaims(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init resourcepoolstatusrequest controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		controller.Run(ctx, 1) // Single worker is sufficient for this controller
	}, controllerName), nil
}
