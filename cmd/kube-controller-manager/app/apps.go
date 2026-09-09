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
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/controller/deployment"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/pkg/controller/statefulset"
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

	rsc := replicaset.NewReplicaSetController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		client,
		replicaset.BurstReplicas,
	)
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

	dc, err := deployment.NewDeploymentController(
		ctx,
		controllerContext.InformerFactory.Apps().V1().Deployments(),
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		client,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating Deployment controller: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		dc.Run(ctx, int(controllerContext.ComponentConfig.DeploymentController.ConcurrentDeploymentSyncs))
	}, controllerName), nil
}
