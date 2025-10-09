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

	"k8s.io/kubernetes/pkg/controller/cronjob"
	"k8s.io/kubernetes/pkg/controller/job"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
)

func newJobControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:        names.JobController,
		Aliases:     []string{"job"},
		Constructor: newJobController,
	}
}

func newJobController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("job-controller")
	if err != nil {
		return nil, err
	}

	jc, err := job.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Batch().V1().Jobs(),
		client,
	)
	if err != nil {
		return nil, fmt.Errorf("creating Job controller: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		jc.Run(ctx, int(controllerContext.ComponentConfig.JobController.ConcurrentJobSyncs))
	}, controllerName), nil
}

func newCronJobControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:        names.CronJobController,
		Aliases:     []string{"cronjob"},
		Constructor: newCronJobController,
	}
}

func newCronJobController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("cronjob-controller")
	if err != nil {
		return nil, err
	}

	cj2c, err := cronjob.NewControllerV2(
		ctx,
		controllerContext.InformerFactory.Batch().V1().Jobs(),
		controllerContext.InformerFactory.Batch().V1().CronJobs(),
		client,
	)
	if err != nil {
		return nil, fmt.Errorf("creating CronJob controller V2: %w", err)
	}

	return run.NewControllerLoop(func(ctx context.Context) {
		cj2c.Run(ctx, int(controllerContext.ComponentConfig.CronJobController.ConcurrentCronJobSyncs))
	}, controllerName), nil
}
