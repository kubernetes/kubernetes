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
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/autoscaler/metrics"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

type HorizontalPodAutoscalerController struct {
	client        client.Interface
	expClient     client.ExperimentalInterface
	metricsClient metrics.MetricsClient
}

var downscaleForbiddenWindow, _ = time.ParseDuration("20m")
var upscaleForbiddenWindow, _ = time.ParseDuration("3m")

func New(client client.Interface, expClient client.ExperimentalInterface, metricsClient metrics.MetricsClient) *HorizontalPodAutoscalerController {
	return &HorizontalPodAutoscalerController{
		client:        client,
		expClient:     expClient,
		metricsClient: metricsClient,
	}
}

func (a *HorizontalPodAutoscalerController) Run(syncPeriod time.Duration) {
	go util.Until(func() {
		if err := a.reconcileAutoscalers(); err != nil {
			glog.Errorf("Couldn't reconcile horizontal pod autoscalers: %v", err)
		}
	}, syncPeriod, util.NeverStop)
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
		currentReplicas := scale.Status.Replicas
		currentConsumption, err := a.metricsClient.ResourceConsumption(hpa.Spec.ScaleRef.Namespace).Get(hpa.Spec.Target.Resource,
			scale.Status.Selector)

		// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
		if err != nil {
			glog.Warningf("Error while getting metrics for %s: %v", reference, err)
			continue
		}

		// if the ratio is 1.2 we want to have 2 replicas
		desiredReplicas := 1 + int((currentConsumption.Quantity.MilliValue()*int64(currentReplicas))/hpa.Spec.Target.Quantity.MilliValue())

		if desiredReplicas < hpa.Spec.MinCount {
			desiredReplicas = hpa.Spec.MinCount
		}
		if desiredReplicas > hpa.Spec.MaxCount {
			desiredReplicas = hpa.Spec.MaxCount
		}
		now := time.Now()
		rescale := false
		if desiredReplicas != currentReplicas {
			// Going down
			if desiredReplicas < currentReplicas && (hpa.Status == nil || hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(downscaleForbiddenWindow).Before(now)) {
				rescale = true
			}

			// Going up
			if desiredReplicas > currentReplicas && (hpa.Status == nil || hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(upscaleForbiddenWindow).Before(now)) {
				rescale = true
			}

			if rescale {
				scale.Spec.Replicas = desiredReplicas
				_, err = a.expClient.Scales(hpa.Namespace).Update(hpa.Spec.ScaleRef.Kind, scale)
				if err != nil {
					glog.Warningf("Failed to rescale %s: %v", reference, err)
					continue
				}
			}
		}

		status := expapi.HorizontalPodAutoscalerStatus{
			CurrentReplicas:    currentReplicas,
			DesiredReplicas:    desiredReplicas,
			CurrentConsumption: currentConsumption,
		}
		hpa.Status = &status
		if rescale {
			now := util.NewTime(now)
			hpa.Status.LastScaleTimestamp = &now
		}

		_, err = a.expClient.HorizontalPodAutoscalers(hpa.Namespace).Update(&hpa)
		if err != nil {
			glog.Warningf("Failed to update HorizontalPodAutoscaler %s: %v", hpa.Name, err)
			continue
		}
	}
	return nil
}
