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
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

const (
	heapsterNamespace = "kube-system"
	heapsterService   = "monitoring-heapster"
)

var resourceToMetric = map[api.ResourceName]string{
	api.ResourceCPU: "cpu-usage",
}
var heapsterQueryStart, _ = time.ParseDuration("-20m")

type HorizontalPodAutoscalerController struct {
	client    *client.Client
	expClient client.ExperimentalInterface
}

func New(client *client.Client, expClient client.ExperimentalInterface) *HorizontalPodAutoscalerController {
	//TODO: switch to client.Interface
	return &HorizontalPodAutoscalerController{
		client:    client,
		expClient: expClient,
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
	list, err := a.expClient.HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	for _, hpa := range list.Items {
		reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Namespace, hpa.Spec.ScaleRef.Name)

		scale, err := a.expClient.Scales(hpa.Spec.ScaleRef.Namespace).Get(hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Name)
		if err != nil {
			glog.Warningf("Failed to query scale subresource for %s: %v", reference, err)
			continue
		}
		podList, err := a.client.Pods(hpa.Spec.ScaleRef.Namespace).
			List(labels.SelectorFromSet(labels.Set(scale.Status.Selector)), fields.Everything())

		if err != nil {
			glog.Warningf("Failed to get pod list for %s: %v", reference, err)
			continue
		}
		podNames := []string{}
		for _, pod := range podList.Items {
			podNames = append(podNames, pod.Name)
		}

		metric, metricDefined := resourceToMetric[hpa.Spec.Target.Resource]
		if !metricDefined {
			glog.Warningf("Heapster metric not defined for %s %v", reference, hpa.Spec.Target.Resource)
			continue
		}
		startTime := time.Now().Add(heapsterQueryStart)
		metricPath := fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s/metrics/%s",
			hpa.Spec.ScaleRef.Namespace,
			strings.Join(podNames, ","),
			metric)

		resultRaw, err := a.client.
			Get().
			Prefix("proxy").
			Resource("services").
			Namespace(heapsterNamespace).
			Name(heapsterService).
			Suffix(metricPath).
			Param("start", startTime.Format(time.RFC3339)).
			Do().
			Raw()

		if err != nil {
			glog.Warningf("Failed to get pods metrics for %s: %v", reference, err)
			continue
		}

		glog.Infof("Metrics available for %s: %s", reference, string(resultRaw))
	}
	return nil
}
