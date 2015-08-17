/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package autoscalercontroller

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

type HorizontalPodAutoscalerController struct {
	kubeClient unversioned.ExperimentalInterface
}

func New(kubeClient unversioned.ExperimentalInterface) *HorizontalPodAutoscalerController {
	return &HorizontalPodAutoscalerController{
		kubeClient: kubeClient,
	}
}

func (a *HorizontalPodAutoscalerController) Run(syncPeriod time.Duration) {
	go util.Forever(func() {
		if err := a.reconcileAutoscalers(); err != nil {
			glog.Errorf("Couldn't reconcile horizontal pod autoscalers: %v", err)
		}
	}, syncPeriod)
}

func (a *HorizontalPodAutoscalerController) reconcileAutoscalers() error {
	ns := api.NamespaceAll
	list, err := a.kubeClient.HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	// TODO: implement!
	glog.Info("autoscalers: %v", list)
	return nil
}
