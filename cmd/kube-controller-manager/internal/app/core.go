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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
package app

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	cpnames "k8s.io/cloud-provider/names"
	"k8s.io/component-base/featuregate"
	kcontroller "k8s.io/controller-manager/controller"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2"
	controller2 "k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	pkgcontroller "k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodeipamcontroller "k8s.io/kubernetes/pkg/controller/nodeipam"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	lifecyclecontroller "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/controller/podgc"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/controller/resourceclaim"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/controller/storageversiongc"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	ttlcontroller "k8s.io/kubernetes/pkg/controller/ttl"
	"k8s.io/kubernetes/pkg/controller/ttlafterfinished"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	"k8s.io/kubernetes/pkg/controller/volume/ephemeral"
	"k8s.io/kubernetes/pkg/controller/volume/expand"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/controller/volume/pvcprotection"
	"k8s.io/kubernetes/pkg/controller/volume/pvprotection"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning"
	"k8s.io/kubernetes/pkg/controller/volume/vacprotection"
	"k8s.io/kubernetes/pkg/features"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/utils/clock"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
)

const (
	// defaultNodeMaskCIDRIPv4 is default mask size for IPv4 node cidr
	defaultNodeMaskCIDRIPv4 = 24
	// defaultNodeMaskCIDRIPv6 is default mask size for IPv6 node cidr
	defaultNodeMaskCIDRIPv6 = 64
)

func newServiceLBControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:    cpnames.ServiceLBController,
		Aliases: []string{"service"},
		Constructor: func(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
			logger := klog.FromContext(ctx)
			logger.Info("Warning: service-controller is set, but no cloud provider functionality is available in kube-controller-manger (KEP-2395). Will not configure service controller.")
			return nil, nil
		},
		IsCloudProviderController: true,
	}
}

func newNodeIpamControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.NodeIpamController,
		Aliases:     []string{"nodeipam"},
		Constructor: newNodeIpamController,
	}
}

func newNodeIpamController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	if !controllerContext.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs {
		return nil, nil
	}
	if controllerContext.ComponentConfig.KubeCloudShared.CIDRAllocatorType == string(ipam.CloudAllocatorType) {
		// Cannot run cloud ipam controller if cloud provider is nil (--cloud-provider not set or set to 'external')
		return nil, errors.New("--cidr-allocator-type is set to 'CloudAllocator' but cloud provider is not configured")
	}

	clusterCIDRs, err := validateCIDRs(controllerContext.ComponentConfig.KubeCloudShared.ClusterCIDR)
	if err != nil {
		return nil, err
	}

	// service cidr processing
	var serviceCIDR *net.IPNet
	var secondaryServiceCIDR *net.IPNet
	logger := klog.FromContext(ctx)
	if len(strings.TrimSpace(controllerContext.ComponentConfig.NodeIPAMController.ServiceCIDR)) != 0 {
		_, serviceCIDR, err = netutils.ParseCIDRSloppy(controllerContext.ComponentConfig.NodeIPAMController.ServiceCIDR)
		if err != nil {
			logger.Info("Warning: unsuccessful parsing of service CIDR", "CIDR", controllerContext.ComponentConfig.NodeIPAMController.ServiceCIDR, "err", err)
		}
	}

	if len(strings.TrimSpace(controllerContext.ComponentConfig.NodeIPAMController.SecondaryServiceCIDR)) != 0 {
		_, secondaryServiceCIDR, err = netutils.ParseCIDRSloppy(controllerContext.ComponentConfig.NodeIPAMController.SecondaryServiceCIDR)
		if err != nil {
			logger.Info("Warning: unsuccessful parsing of service CIDR", "CIDR", controllerContext.ComponentConfig.NodeIPAMController.SecondaryServiceCIDR, "err", err)
		}
	}

	// the following checks are triggered if both serviceCIDR and secondaryServiceCIDR are provided
	if serviceCIDR != nil && secondaryServiceCIDR != nil {
		// should be dual stack (from different IPFamilies)
		dualstackServiceCIDR, err := netutils.IsDualStackCIDRs([]*net.IPNet{serviceCIDR, secondaryServiceCIDR})
		if err != nil {
			return nil, fmt.Errorf("failed to perform dualstack check on serviceCIDR and secondaryServiceCIDR error: %w", err)
		}
		if !dualstackServiceCIDR {
			return nil, fmt.Errorf("serviceCIDR and secondaryServiceCIDR are not dualstack (from different IPfamiles)")
		}
	}

	// only --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 supported with dual stack clusters.
	// --node-cidr-mask-size flag is incompatible with dual stack clusters.
	nodeCIDRMaskSizes, err := setNodeCIDRMaskSizes(controllerContext.ComponentConfig.NodeIPAMController, clusterCIDRs)
	if err != nil {
		return nil, err
	}

	client, err := controllerContext.NewClient("node-controller")
	if err != nil {
		return nil, err
	}

	nodeIpamController, err := nodeipamcontroller.NewNodeIpamController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Nodes(),
		nil, // no cloud provider on kube-controller-manager since v1.31 (KEP-2395)
		client,
		clusterCIDRs,
		serviceCIDR,
		secondaryServiceCIDR,
		nodeCIDRMaskSizes,
		ipam.CIDRAllocatorType(controllerContext.ComponentConfig.KubeCloudShared.CIDRAllocatorType),
	)
	if err != nil {
		return nil, err
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		nodeIpamController.RunWithMetrics(ctx, controllerContext.ControllerManagerMetrics)
	}, controllerName), nil
}

func newNodeLifecycleControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.NodeLifecycleController,
		Aliases:     []string{"nodelifecycle"},
		Constructor: newNodeLifecycleController,
	}
}

func newNodeLifecycleController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("node-controller")
	if err != nil {
		return nil, err
	}

	nlc, err := lifecyclecontroller.NewNodeLifecycleController(
		ctx,
		controllerContext.InformerFactory.Coordination().V1().Leases(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerContext.InformerFactory.Apps().V1().DaemonSets(),
		// node lifecycle controller uses existing cluster role from node-controller
		client,
		controllerContext.ComponentConfig.KubeCloudShared.NodeMonitorPeriod.Duration,
		controllerContext.ComponentConfig.NodeLifecycleController.NodeStartupGracePeriod.Duration,
		controllerContext.ComponentConfig.NodeLifecycleController.NodeMonitorGracePeriod.Duration,
		controllerContext.ComponentConfig.NodeLifecycleController.NodeEvictionRate,
		controllerContext.ComponentConfig.NodeLifecycleController.SecondaryNodeEvictionRate,
		controllerContext.ComponentConfig.NodeLifecycleController.LargeClusterSizeThreshold,
		controllerContext.ComponentConfig.NodeLifecycleController.UnhealthyZoneThreshold,
	)
	if err != nil {
		return nil, err
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		nlc.Run(ctx)
	}, controllerName), nil
}

func newTaintEvictionControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.TaintEvictionController,
		Constructor: newTaintEvictionController,
		RequiredFeatureGates: []featuregate.Feature{
			features.SeparateTaintEvictionController,
		},
	}
}

func newTaintEvictionController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	// taint-manager uses existing cluster role from node-controller
	client, err := controllerContext.NewClient("node-controller")
	if err != nil {
		return nil, err
	}

	tec, err := tainteviction.New(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerName,
	)
	if err != nil {
		return nil, err
	}

	return run.NewControllerLoop(tec.Run, controllerName), nil
}

func newDeviceTaintEvictionControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.DeviceTaintEvictionController,
		Constructor: newDeviceTaintEvictionController,
		RequiredFeatureGates: []featuregate.Feature{
			// TODO update app.TestFeatureGatedControllersShouldNotDefineAliases when removing these feature gates.
			features.DynamicResourceAllocation,
			features.DRADeviceTaints,
		},
	}
}

func newDeviceTaintEvictionController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient(names.DeviceTaintEvictionController)
	if err != nil {
		return nil, err
	}

	deviceTaintEvictionController := devicetainteviction.New(
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Resource().V1().ResourceClaims(),
		controllerContext.InformerFactory.Resource().V1().ResourceSlices(),
		controllerContext.InformerFactory.Resource().V1alpha3().DeviceTaintRules(),
		controllerContext.InformerFactory.Resource().V1().DeviceClasses(),
		controllerName,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		if err := deviceTaintEvictionController.Run(ctx); err != nil {
			klog.FromContext(ctx).Error(err, "Device taint processing leading to Pod eviction failed and is now paused")
		}
		<-ctx.Done()
	}, controllerName), nil
}

func newCloudNodeLifecycleControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:    cpnames.CloudNodeLifecycleController,
		Aliases: []string{"cloud-node-lifecycle"},
		Constructor: func(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
			logger := klog.FromContext(ctx)
			logger.Info("Warning: node-controller is set, but no cloud provider functionality is available in kube-controller-manger (KEP-2395). Will not configure node lifecyle controller.")
			return nil, nil
		},
		IsCloudProviderController: true,
	}
}

func newNodeRouteControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:    cpnames.NodeRouteController,
		Aliases: []string{"route"},
		Constructor: func(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
			logger := klog.FromContext(ctx)
			logger.Info("Warning: configure-cloud-routes is set, but no cloud provider functionality is available in kube-controller-manger (KEP-2395). Will not configure cloud provider routes.")
			return nil, nil
		},
		IsCloudProviderController: true,
	}
}

func newPersistentVolumeBinderControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PersistentVolumeBinderController,
		Aliases:     []string{"persistentvolume-binder"},
		Constructor: newPersistentVolumeBinderController,
	}
}

func newPersistentVolumeBinderController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	logger := klog.FromContext(ctx)
	plugins, err := ProbeProvisionableRecyclableVolumePlugins(logger, controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration)
	if err != nil {
		return nil, fmt.Errorf("failed to probe volume plugins when starting persistentvolume controller: %w", err)
	}

	client, err := controllerContext.NewClient("persistent-volume-binder")
	if err != nil {
		return nil, err
	}

	params := persistentvolumecontroller.ControllerParameters{
		KubeClient:                client,
		SyncPeriod:                controllerContext.ComponentConfig.PersistentVolumeBinderController.PVClaimBinderSyncPeriod.Duration,
		VolumePlugins:             plugins,
		VolumeInformer:            controllerContext.InformerFactory.Core().V1().PersistentVolumes(),
		ClaimInformer:             controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             controllerContext.InformerFactory.Storage().V1().StorageClasses(),
		PodInformer:               controllerContext.InformerFactory.Core().V1().Pods(),
		NodeInformer:              controllerContext.InformerFactory.Core().V1().Nodes(),
		EnableDynamicProvisioning: controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration.EnableDynamicProvisioning,
	}
	volumeController, err := persistentvolumecontroller.NewController(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to construct persistentvolume controller: %w", err)
	}

	return run.NewControllerLoop(volumeController.Run, controllerName), nil
}

func newPersistentVolumeAttachDetachControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PersistentVolumeAttachDetachController,
		Aliases:     []string{"attachdetach"},
		Constructor: newPersistentVolumeAttachDetachController,
	}
}

func newPersistentVolumeAttachDetachController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	logger := klog.FromContext(ctx)
	csiNodeInformer := controllerContext.InformerFactory.Storage().V1().CSINodes()
	csiDriverInformer := controllerContext.InformerFactory.Storage().V1().CSIDrivers()

	plugins, err := ProbeAttachableVolumePlugins(logger, controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration)
	if err != nil {
		return nil, fmt.Errorf("failed to probe volume plugins when starting attach/detach controller: %w", err)
	}

	client, err := controllerContext.NewClient("attachdetach-controller")
	if err != nil {
		return nil, err
	}

	ctx = klog.NewContext(ctx, logger)
	attachDetachController, err := attachdetach.NewAttachDetachController(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumes(),
		csiNodeInformer,
		csiDriverInformer,
		controllerContext.InformerFactory.Storage().V1().VolumeAttachments(),
		plugins,
		GetDynamicPluginProber(controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration),
		controllerContext.ComponentConfig.AttachDetachController.DisableAttachDetachReconcilerSync,
		controllerContext.ComponentConfig.AttachDetachController.ReconcilerSyncLoopPeriod.Duration,
		controllerContext.ComponentConfig.AttachDetachController.DisableForceDetachOnTimeout,
		attachdetach.DefaultTimerConfig,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to start attach/detach controller: %w", err)
	}

	return run.NewControllerLoop(attachDetachController.Run, controllerName), nil
}

func newPersistentVolumeExpanderControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PersistentVolumeExpanderController,
		Aliases:     []string{"persistentvolume-expander"},
		Constructor: newPersistentVolumeExpanderController,
	}
}

func newPersistentVolumeExpanderController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	logger := klog.FromContext(ctx)
	plugins, err := ProbeExpandableVolumePlugins(logger, controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration)
	if err != nil {
		return nil, fmt.Errorf("failed to probe volume plugins when starting volume expand controller: %w", err)
	}
	csiTranslator := csitrans.New()

	client, err := controllerContext.NewClient("expand-controller")
	if err != nil {
		return nil, err
	}

	expandController, err := expand.NewExpandController(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		plugins,
		csiTranslator,
		csimigration.NewPluginManager(csiTranslator, utilfeature.DefaultFeatureGate),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init volume expand controller: %w", err)
	}

	return run.NewControllerLoop(expandController.Run, controllerName), nil
}

func newEphemeralVolumeControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.EphemeralVolumeController,
		Aliases:     []string{"ephemeral-volume"},
		Constructor: newEphemeralVolumeController,
	}
}

func newEphemeralVolumeController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("ephemeral-volume-controller")
	if err != nil {
		return nil, err
	}

	ephemeralController, err := ephemeral.NewController(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims())
	if err != nil {
		return nil, fmt.Errorf("failed to init ephemeral volume controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		ephemeralController.Run(ctx, int(controllerContext.ComponentConfig.EphemeralVolumeController.ConcurrentEphemeralVolumeSyncs))
	}, controllerName), nil
}

const defaultResourceClaimControllerWorkers = 50

func newResourceClaimControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.ResourceClaimController,
		Aliases:     []string{"resource-claim-controller"},
		Constructor: newResourceClaimController,
		RequiredFeatureGates: []featuregate.Feature{
			features.DynamicResourceAllocation, // TODO update app.TestFeatureGatedControllersShouldNotDefineAliases when removing this feature
		},
	}
}

func newResourceClaimController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
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

	return run.NewControllerLoop(func(ctx context.Context) {
		ephemeralController.Run(ctx, defaultResourceClaimControllerWorkers)
	}, controllerName), nil
}

func newEndpointsControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.EndpointsController,
		Aliases:     []string{"endpoint"},
		Constructor: newEndpointsController,
	}
}

func newEndpointsController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("endpoint-controller")
	if err != nil {
		return nil, err
	}

	ec := endpointcontroller.NewEndpointController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Services(),
		controllerContext.InformerFactory.Core().V1().Endpoints(),
		client,
		controllerContext.ComponentConfig.EndpointController.EndpointUpdatesBatchPeriod.Duration,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		ec.Run(ctx, int(controllerContext.ComponentConfig.EndpointController.ConcurrentEndpointSyncs))
	}, controllerName), nil
}

func newReplicationControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.ReplicationControllerController,
		Aliases:     []string{"replicationcontroller"},
		Constructor: newReplicationController,
	}
}

func newReplicationController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("replication-controller")
	if err != nil {
		return nil, err
	}

	rc := replicationcontroller.NewReplicationManager(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().ReplicationControllers(),
		client,
		replicationcontroller.BurstReplicas,
	)

	return run.NewControllerLoop(func(ctx context.Context) {
		rc.Run(ctx, int(controllerContext.ComponentConfig.ReplicationController.ConcurrentRCSyncs))
	}, controllerName), nil
}

func newPodGarbageCollectorControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PodGarbageCollectorController,
		Aliases:     []string{"podgc"},
		Constructor: newPodGarbageCollectorController,
	}
}

func newPodGarbageCollectorController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("pod-garbage-collector")
	if err != nil {
		return nil, err
	}

	pgcc := podgc.NewPodGC(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		int(controllerContext.ComponentConfig.PodGCController.TerminatedPodGCThreshold),
	)
	return run.NewControllerLoop(pgcc.Run, controllerName), nil
}

func newResourceQuotaControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.ResourceQuotaController,
		Aliases:     []string{"resourcequota"},
		Constructor: newResourceQuotaController,
	}
}

func newResourceQuotaController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	resourceQuotaControllerClient, err := controllerContext.NewClient("resourcequota-controller")
	if err != nil {
		return nil, err
	}

	resourceQuotaControllerDiscoveryClient, err := controllerContext.ClientBuilder.DiscoveryClient("resourcequota-controller")
	if err != nil {
		return nil, fmt.Errorf("failed to create the discovery client: %w", err)
	}

	discoveryFunc := resourceQuotaControllerDiscoveryClient.ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(controllerContext.InformerFactory.ForResource)
	quotaConfiguration := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)

	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               resourceQuotaControllerClient.CoreV1(),
		ResourceQuotaInformer:     controllerContext.InformerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              pkgcontroller.StaticResyncPeriodFunc(controllerContext.ComponentConfig.ResourceQuotaController.ResourceQuotaSyncPeriod.Duration),
		InformerFactory:           controllerContext.ObjectOrMetadataInformerFactory,
		ReplenishmentResyncPeriod: controllerContext.ResyncPeriod,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		InformersStarted:          controllerContext.InformersStarted,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		UpdateFilter:              quotainstall.DefaultUpdateFilter(),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(ctx, resourceQuotaControllerOptions)
	if err != nil {
		return nil, err
	}

	return run.NewControllerLoop(run.Concurrent(
		func(ctx context.Context) {
			resourceQuotaController.Run(ctx, int(controllerContext.ComponentConfig.ResourceQuotaController.ConcurrentResourceQuotaSyncs))
		},
		func(ctx context.Context) {
			resourceQuotaController.Sync(ctx, discoveryFunc, 30*time.Second)
		},
	), controllerName), nil
}

func newNamespaceControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.NamespaceController,
		Aliases:     []string{"namespace"},
		Constructor: newNamespaceController,
	}
}

func newNamespaceController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	// the namespace cleanup controller is very chatty.  It makes lots of discovery calls and then it makes lots of delete calls
	// the ratelimiter negatively affects its speed.  Deleting 100 total items in a namespace (that's only a few of each resource
	// including events), takes ~10 seconds by default.
	nsKubeconfig, err := controllerContext.NewClientConfig("namespace-controller")
	if err != nil {
		return nil, err
	}

	nsKubeconfig.QPS *= 20
	nsKubeconfig.Burst *= 100

	namespaceKubeClient, err := clientset.NewForConfig(nsKubeconfig)
	if err != nil {
		return nil, err
	}

	return newModifiedNamespaceController(ctx, controllerContext, controllerName, namespaceKubeClient, nsKubeconfig)
}

func newModifiedNamespaceController(
	ctx context.Context, controllerContext controller2.Context, controllerName string,
	namespaceKubeClient clientset.Interface, nsKubeconfig *restclient.Config,
) (controller2.Controller, error) {
	metadataClient, err := metadata.NewForConfig(nsKubeconfig)
	if err != nil {
		return nil, err
	}

	discoverResourcesFn := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources

	namespaceController := namespacecontroller.NewNamespaceController(
		ctx,
		namespaceKubeClient,
		metadataClient,
		discoverResourcesFn,
		controllerContext.InformerFactory.Core().V1().Namespaces(),
		controllerContext.ComponentConfig.NamespaceController.NamespaceSyncPeriod.Duration,
		v1.FinalizerKubernetes,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		namespaceController.Run(ctx, int(controllerContext.ComponentConfig.NamespaceController.ConcurrentNamespaceSyncs))
	}, controllerName), nil
}

func newServiceAccountControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.ServiceAccountController,
		Aliases:     []string{"serviceaccount"},
		Constructor: newServiceAccountController,
	}
}

func newServiceAccountController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("service-account-controller")
	if err != nil {
		return nil, err
	}

	sac, err := serviceaccountcontroller.NewServiceAccountsController(
		controllerContext.InformerFactory.Core().V1().ServiceAccounts(),
		controllerContext.InformerFactory.Core().V1().Namespaces(),
		client,
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating ServiceAccount controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		sac.Run(ctx, 1)
	}, controllerName), nil
}

func newTTLControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.TTLController,
		Aliases:     []string{"ttl"},
		Constructor: newTTLController,
	}
}

func newTTLController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("ttl-controller")
	if err != nil {
		return nil, err
	}

	ttlc := ttlcontroller.NewTTLController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Nodes(),
		client,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		ttlc.Run(ctx, 5)
	}, controllerName), nil
}

func newGarbageCollectorControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.GarbageCollectorController,
		Aliases:     []string{"garbagecollector"},
		Constructor: newGarbageCollectorController,
	}
}

type garbageCollectorController struct {
	*garbagecollector.GarbageCollector
	controllerContext controller2.Context
	controllerName    string
	discoveryClient   discovery.DiscoveryInterface
}

// Make sure we are propagating properly.
var _ kcontroller.Debuggable = (*garbageCollectorController)(nil)

func newGarbageCollectorController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	if !controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, nil
	}

	client, err := controllerContext.NewClient("generic-garbage-collector")
	if err != nil {
		return nil, err
	}

	discoveryClient, err := controllerContext.ClientBuilder.DiscoveryClient("generic-garbage-collector")
	if err != nil {
		return nil, fmt.Errorf("failed to create the discovery client: %w", err)
	}

	config, err := controllerContext.NewClientConfig("generic-garbage-collector")
	if err != nil {
		return nil, err
	}

	// Increase garbage collector controller's throughput: each object deletion takes two API calls,
	// so to get |config.QPS| deletion rate we need to allow 2x more requests for this controller.
	config.QPS *= 2
	metadataClient, err := metadata.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	garbageCollector, err := garbagecollector.NewComposedGarbageCollector(
		ctx,
		client,
		metadataClient,
		controllerContext.RESTMapper,
		controllerContext.GraphBuilder,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init the generic garbage collector: %w", err)
	}

	return &garbageCollectorController{
		GarbageCollector:  garbageCollector,
		controllerName:    controllerName,
		controllerContext: controllerContext,
		discoveryClient:   discoveryClient,
	}, nil
}

// Name must be implemented explicitly as it collides with the embedded controller.
func (c *garbageCollectorController) Name() string {
	return c.controllerName
}

func (c *garbageCollectorController) Run(ctx context.Context) {
	workers := int(c.controllerContext.ComponentConfig.GarbageCollectorController.ConcurrentGCSyncs)
	const syncPeriod = 30 * time.Second

	run.Concurrent(
		func(ctx context.Context) {
			c.GarbageCollector.Run(ctx, workers, syncPeriod)
		},
		func(ctx context.Context) {
			// Periodically refresh the RESTMapper with new discovery information and sync the garbage collector.
			c.Sync(ctx, c.discoveryClient, syncPeriod)
		},
	)(ctx)
}

func newPersistentVolumeClaimProtectionControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PersistentVolumeClaimProtectionController,
		Aliases:     []string{"pvc-protection"},
		Constructor: newPersistentVolumeClaimProtectionController,
	}
}

func newPersistentVolumeClaimProtectionController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("pvc-protection-controller")
	if err != nil {
		return nil, err
	}

	pvcProtectionController, err := pvcprotection.NewPVCProtectionController(
		klog.FromContext(ctx),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		client,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init the pvc protection controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		pvcProtectionController.Run(ctx, 1)
	}, controllerName), nil
}

func newPersistentVolumeProtectionControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.PersistentVolumeProtectionController,
		Aliases:     []string{"pv-protection"},
		Constructor: newPersistentVolumeProtectionController,
	}
}

func newPersistentVolumeProtectionController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("pv-protection-controller")
	if err != nil {
		return nil, err
	}

	pvpc := pvprotection.NewPVProtectionController(
		klog.FromContext(ctx),
		controllerContext.InformerFactory.Core().V1().PersistentVolumes(),
		client,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		pvpc.Run(ctx, 1)
	}, controllerName), nil
}

func newVolumeAttributesClassProtectionControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.VolumeAttributesClassProtectionController,
		Constructor: newVolumeAttributesClassProtectionController,
		RequiredFeatureGates: []featuregate.Feature{
			features.VolumeAttributesClass,
		},
	}
}

func newVolumeAttributesClassProtectionController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("volumeattributesclass-protection-controller")
	if err != nil {
		return nil, err
	}

	vacProtectionController, err := vacprotection.NewVACProtectionController(
		klog.FromContext(ctx),
		client,
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumes(),
		controllerContext.InformerFactory.Storage().V1().VolumeAttributesClasses(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init the vac protection controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		vacProtectionController.Run(ctx, 1)
	}, controllerName), nil
}

func newTTLAfterFinishedControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.TTLAfterFinishedController,
		Aliases:     []string{"ttl-after-finished"},
		Constructor: newTTLAfterFinishedController,
	}
}

func newTTLAfterFinishedController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("ttl-after-finished-controller")
	if err != nil {
		return nil, err
	}

	ttlc := ttlafterfinished.New(
		ctx,
		controllerContext.InformerFactory.Batch().V1().Jobs(),
		client,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		ttlc.Run(ctx, int(controllerContext.ComponentConfig.TTLAfterFinishedController.ConcurrentTTLSyncs))
	}, controllerName), nil
}

func newLegacyServiceAccountTokenCleanerControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.LegacyServiceAccountTokenCleanerController,
		Aliases:     []string{"legacy-service-account-token-cleaner"},
		Constructor: newLegacyServiceAccountTokenCleanerController,
	}
}

func newLegacyServiceAccountTokenCleanerController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("legacy-service-account-token-cleaner")
	if err != nil {
		return nil, err
	}

	cleanUpPeriod := controllerContext.ComponentConfig.LegacySATokenCleaner.CleanUpPeriod.Duration
	legacySATokenCleaner, err := serviceaccountcontroller.NewLegacySATokenCleaner(
		controllerContext.InformerFactory.Core().V1().ServiceAccounts(),
		controllerContext.InformerFactory.Core().V1().Secrets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		client,
		clock.RealClock{},
		serviceaccountcontroller.LegacySATokenCleanerOptions{
			CleanUpPeriod: cleanUpPeriod,
			SyncInterval:  serviceaccountcontroller.DefaultCleanerSyncInterval,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init the legacy service account token cleaner: %w", err)
	}

	return run.NewControllerLoop(legacySATokenCleaner.Run, controllerName), nil
}

// processCIDRs is a helper function that works on a comma separated cidrs and returns
// a list of typed cidrs
// error if failed to parse any of the cidrs or invalid length of cidrs
func validateCIDRs(cidrsList string) ([]*net.IPNet, error) {
	// failure: bad cidrs in config
	clusterCIDRs, dualStack, err := processCIDRs(cidrsList)
	if err != nil {
		return nil, err
	}

	// failure: more than one cidr but they are not configured as dual stack
	if len(clusterCIDRs) > 1 && !dualStack {
		return nil, fmt.Errorf("len of ClusterCIDRs==%v and they are not configured as dual stack (at least one from each IPFamily", len(clusterCIDRs))
	}

	// failure: more than cidrs is not allowed even with dual stack
	if len(clusterCIDRs) > 2 {
		return nil, fmt.Errorf("length of clusterCIDRs is:%v more than max allowed of 2", len(clusterCIDRs))
	}

	return clusterCIDRs, nil
}

// processCIDRs is a helper function that works on a comma separated cidrs and returns
// a list of typed cidrs
// a flag if cidrs represents a dual stack
// error if failed to parse any of the cidrs
func processCIDRs(cidrsList string) ([]*net.IPNet, bool, error) {
	cidrsSplit := strings.Split(strings.TrimSpace(cidrsList), ",")

	cidrs, err := netutils.ParseCIDRs(cidrsSplit)
	if err != nil {
		return nil, false, err
	}

	// if cidrs has an error then the previous call will fail
	// safe to ignore error checking on next call
	dualstack, _ := netutils.IsDualStackCIDRs(cidrs)

	return cidrs, dualstack, nil
}

// setNodeCIDRMaskSizes returns the IPv4 and IPv6 node cidr mask sizes to the value provided
// for --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 respectively. If value not provided,
// then it will return default IPv4 and IPv6 cidr mask sizes.
func setNodeCIDRMaskSizes(cfg nodeipamconfig.NodeIPAMControllerConfiguration, clusterCIDRs []*net.IPNet) ([]int, error) {

	sortedSizes := func(maskSizeIPv4, maskSizeIPv6 int) []int {
		nodeMaskCIDRs := make([]int, len(clusterCIDRs))

		for idx, clusterCIDR := range clusterCIDRs {
			if netutils.IsIPv6CIDR(clusterCIDR) {
				nodeMaskCIDRs[idx] = maskSizeIPv6
			} else {
				nodeMaskCIDRs[idx] = maskSizeIPv4
			}
		}
		return nodeMaskCIDRs
	}

	// --node-cidr-mask-size flag is incompatible with dual stack clusters.
	ipv4Mask, ipv6Mask := defaultNodeMaskCIDRIPv4, defaultNodeMaskCIDRIPv6
	isDualstack := len(clusterCIDRs) > 1

	// case one: cluster is dualstack (i.e, more than one cidr)
	if isDualstack {
		// if --node-cidr-mask-size then fail, user must configure the correct dual-stack mask sizes (or use default)
		if cfg.NodeCIDRMaskSize != 0 {
			return nil, errors.New("usage of --node-cidr-mask-size is not allowed with dual-stack clusters")

		}

		if cfg.NodeCIDRMaskSizeIPv4 != 0 {
			ipv4Mask = int(cfg.NodeCIDRMaskSizeIPv4)
		}
		if cfg.NodeCIDRMaskSizeIPv6 != 0 {
			ipv6Mask = int(cfg.NodeCIDRMaskSizeIPv6)
		}
		return sortedSizes(ipv4Mask, ipv6Mask), nil
	}

	maskConfigured := cfg.NodeCIDRMaskSize != 0
	maskV4Configured := cfg.NodeCIDRMaskSizeIPv4 != 0
	maskV6Configured := cfg.NodeCIDRMaskSizeIPv6 != 0
	isSingleStackIPv6 := netutils.IsIPv6CIDR(clusterCIDRs[0])

	// original flag is set
	if maskConfigured {
		// original mask flag is still the main reference.
		if maskV4Configured || maskV6Configured {
			return nil, errors.New("usage of --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 is not allowed if --node-cidr-mask-size is set. For dual-stack clusters please unset it and use IPFamily specific flags")
		}

		mask := int(cfg.NodeCIDRMaskSize)
		return sortedSizes(mask, mask), nil
	}

	if maskV4Configured {
		if isSingleStackIPv6 {
			return nil, errors.New("usage of --node-cidr-mask-size-ipv4 is not allowed for a single-stack IPv6 cluster")
		}

		ipv4Mask = int(cfg.NodeCIDRMaskSizeIPv4)
	}

	// !maskV4Configured && !maskConfigured && maskV6Configured
	if maskV6Configured {
		if !isSingleStackIPv6 {
			return nil, errors.New("usage of --node-cidr-mask-size-ipv6 is not allowed for a single-stack IPv4 cluster")
		}

		ipv6Mask = int(cfg.NodeCIDRMaskSizeIPv6)
	}
	return sortedSizes(ipv4Mask, ipv6Mask), nil
}

func newStorageVersionGarbageCollectorControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:        names.StorageVersionGarbageCollectorController,
		Aliases:     []string{"storage-version-gc"},
		Constructor: newStorageVersionGarbageCollectorController,
		RequiredFeatureGates: []featuregate.Feature{
			genericfeatures.APIServerIdentity,
			genericfeatures.StorageVersionAPI,
		},
	}
}

func newStorageVersionGarbageCollectorController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient("storage-version-garbage-collector")
	if err != nil {
		return nil, err
	}

	svgcc := storageversiongc.NewStorageVersionGC(
		ctx,
		client,
		controllerContext.InformerFactory.Coordination().V1().Leases(),
		controllerContext.InformerFactory.Internal().V1alpha1().StorageVersions(),
	)
	return run.NewControllerLoop(svgcc.Run, controllerName), nil
}

func newSELinuxWarningControllerDescriptor() *controller2.Descriptor {
	return &controller2.Descriptor{
		Name:                names.SELinuxWarningController,
		Constructor:         newSELinuxWarningController,
		IsDisabledByDefault: true,
		RequiredFeatureGates: []featuregate.Feature{
			features.SELinuxChangePolicy,
		},
	}
}

func newSELinuxWarningController(ctx context.Context, controllerContext controller2.Context, controllerName string) (controller2.Controller, error) {
	client, err := controllerContext.NewClient(controllerName)
	if err != nil {
		return nil, err
	}

	logger := klog.FromContext(ctx)
	csiDriverInformer := controllerContext.InformerFactory.Storage().V1().CSIDrivers()
	plugins, err := ProbePersistentVolumePlugins(logger, controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration)
	if err != nil {
		return nil, fmt.Errorf("failed to probe volume plugins when starting SELinux warning controller: %w", err)
	}

	seLinuxController, err := selinuxwarning.NewController(
		ctx,
		client,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumes(),
		csiDriverInformer,
		plugins,
		GetDynamicPluginProber(controllerContext.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to start SELinux warning controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		seLinuxController.Run(ctx, 1)
	}, controllerName), nil
}
