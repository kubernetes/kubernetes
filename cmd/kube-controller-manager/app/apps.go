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

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/pkg/controller/statefulset"
)

func newDaemonSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.DaemonSetController,
		aliases:  []string{"daemonset"},
		initFunc: startDaemonSetController,
	}
}
func startDaemonSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	dsc, err := daemon.NewDaemonSetsController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().DaemonSets(),
		controllerContext.InformerFactory.Apps().V1().ControllerRevisions(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "daemon-set-controller"),
		flowcontrol.NewBackOff(1*time.Second, 15*time.Minute),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating DaemonSets controller: %v", err)
	}
	go dsc.Run(ctx, int(controllerContext.ComponentConfig.DaemonSetController.ConcurrentDaemonSetSyncs))
	return nil, true, nil
}

func newStatefulSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.StatefulSetController,
		aliases:  []string{"statefulset"},
		initFunc: startStatefulSetController,
	}
}
func startStatefulSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	go statefulset.NewStatefulSetController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Apps().V1().StatefulSets(),
		controllerContext.InformerFactory.Core().V1().PersistentVolumeClaims(),
		controllerContext.InformerFactory.Apps().V1().ControllerRevisions(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "statefulset-controller"),
	).Run(ctx, int(controllerContext.ComponentConfig.StatefulSetController.ConcurrentStatefulSetSyncs))
	return nil, true, nil
}

func newReplicaSetControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.ReplicaSetController,
		aliases:  []string{"replicaset"},
		initFunc: startReplicaSetController,
	}
}

func startReplicaSetController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	go replicaset.NewReplicaSetController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "replicaset-controller"),
		replicaset.BurstReplicas,
	).Run(ctx, int(controllerContext.ComponentConfig.ReplicaSetController.ConcurrentRSSyncs))
	return nil, true, nil
}

func newDeploymentControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.DeploymentController,
		aliases:  []string{"deployment"},
		initFunc: startDeploymentController,
	}
}

func startDeploymentController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	dc, err := deployment.NewDeploymentController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().Deployments(),
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "deployment-controller"),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating Deployment controller: %v", err)
	}
	go dc.Run(ctx, int(controllerContext.ComponentConfig.DeploymentController.ConcurrentDeploymentSyncs))
	return nil, true, nil
}
