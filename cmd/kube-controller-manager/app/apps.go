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
	"fmt"
	"time"

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1beta1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/pkg/controller/statefulset"
	"k8s.io/kubernetes/pkg/features"
)

func newDaemonSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.DaemonSetController,
		aliases:     []string{"daemonset"},
		constructor: newDaemonSetController,
	}
}

func newDaemonSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("daemon-set-controller")
	if err != nil {
		return nil, err
	}

	dsc, err := daemon.NewDaemonSetsController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().DaemonSets(),
		controllerContext.InformerFactory.Apps().V1().ControllerRevisions(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		client,
		flowcontrol.NewBackOff(1*time.Second, 15*time.Minute),
	)
	if err != nil {
		return nil, fmt.Errorf("error creating DaemonSets controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		dsc.Run(ctx, int(controllerContext.ComponentConfig.DaemonSetController.ConcurrentDaemonSetSyncs))
	}, controllerName), nil
}

func newStatefulSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.StatefulSetController,
		aliases:     []string{"statefulset"},
		constructor: newStatefulSetController,
	}
}

func newStatefulSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("statefulset-controller")
	if err != nil {
		return nil, err
	}

	ssc := statefulset.NewStatefulSetController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Apps().V1().StatefulSets(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Apps().V1().ControllerRevisions(),
		client,
	)
	return newControllerLoop(func(ctx context.Context) {
		ssc.Run(ctx, int(controllerContext.ComponentConfig.StatefulSetController.ConcurrentStatefulSetSyncs))
	}, controllerName), nil
}

func newReplicaSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.ReplicaSetController,
		aliases:     []string{"replicaset"},
		constructor: newReplicaSetController,
	}
}

func newReplicaSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("replicaset-controller")
	if err != nil {
		return nil, err
	}

	var workloadInformer schedulinginformers.WorkloadInformer
	var podGroupInformer schedulinginformers.PodGroupInformer
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) {
		served, err := schedulingWorkloadAPIServed(client)
		if err != nil {
			return nil, fmt.Errorf("discovery for %s: %w", schedulingv1beta1.SchemeGroupVersion, err)
		}
		if served {
			workloadInformer = controllerContext.InformerFactory.Scheduling().V1beta1().Workloads()
			podGroupInformer = controllerContext.InformerFactory.Scheduling().V1beta1().PodGroups()
		} else {
			klog.FromContext(ctx).Info("WorkloadWithDeployment is enabled but scheduling.k8s.io/v1beta1 is not served, running ReplicaSet controller without workload-aware scheduling")
		}
	}

	rsc, err := replicaset.NewReplicaSetController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		workloadInformer,
		podGroupInformer,
		client,
		replicaset.BurstReplicas,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating ReplicaSet controller: %w", err)
	}
	return newControllerLoop(func(ctx context.Context) {
		rsc.Run(ctx, int(controllerContext.ComponentConfig.ReplicaSetController.ConcurrentRSSyncs))
	}, controllerName), nil
}

func newDeploymentControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.DeploymentController,
		aliases:     []string{"deployment"},
		constructor: newDeploymentController,
	}
}

func newDeploymentController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("deployment-controller")
	if err != nil {
		return nil, err
	}

	var workloadInformer schedulinginformers.WorkloadInformer
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) {
		served, err := schedulingWorkloadAPIServed(client)
		if err != nil {
			return nil, fmt.Errorf("discovery for %s: %w", schedulingv1beta1.SchemeGroupVersion, err)
		}
		if served {
			workloadInformer = controllerContext.InformerFactory.Scheduling().V1beta1().Workloads()
		} else {
			klog.FromContext(ctx).Info("WorkloadWithDeployment is enabled but scheduling.k8s.io/v1beta1 is not served, running Deployment controller without workload-aware scheduling")
		}
	}

	dc, err := deployment.NewDeploymentController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().Deployments(),
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		workloadInformer,
		client,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating Deployment controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		dc.Run(ctx, int(controllerContext.ComponentConfig.DeploymentController.ConcurrentDeploymentSyncs))
	}, controllerName), nil
}

// schedulingWorkloadAPIServed reports whether the apiserver serves the
// scheduling.k8s.io/v1beta1 Workload and PodGroup resources.
func schedulingWorkloadAPIServed(client kubernetes.Interface) (bool, error) {
	resList, err := client.Discovery().ServerResourcesForGroupVersion(schedulingv1beta1.SchemeGroupVersion.String())
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if resList != nil {
		var hasWorkloads, hasPodGroups bool
		for _, r := range resList.APIResources {
			switch r.Name {
			case "workloads":
				hasWorkloads = true
			case "podgroups":
				hasPodGroups = true
			}
		}
		return hasWorkloads && hasPodGroups, nil
	}
	return false, err
}
