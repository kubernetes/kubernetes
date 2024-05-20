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

	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/cronjob"
	"k8s.io/kubernetes/pkg/controller/job"
)

func newJobControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.JobController,
		aliases:  []string{"job"},
		initFunc: startJobController,
	}
}

func startJobController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	jobController, err := job.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Batch().V1().Jobs(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "job-controller"),
	)
	if err != nil {
		return nil, true, fmt.Errorf("creating Job controller: %v", err)
	}
	go jobController.Run(ctx, int(controllerContext.ComponentConfig.JobController.ConcurrentJobSyncs))
	return nil, true, nil
}

func newCronJobControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.CronJobController,
		aliases:  []string{"cronjob"},
		initFunc: startCronJobController,
	}
}

func startCronJobController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	cj2c, err := cronjob.NewControllerV2(ctx, controllerContext.InformerFactory.Batch().V1().Jobs(),
		controllerContext.InformerFactory.Batch().V1().CronJobs(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "cronjob-controller"),
	)
	if err != nil {
		return nil, true, fmt.Errorf("creating CronJob controller V2: %v", err)
	}

	go cj2c.Run(ctx, int(controllerContext.ComponentConfig.CronJobController.ConcurrentCronJobSyncs))
	return nil, true, nil
}
