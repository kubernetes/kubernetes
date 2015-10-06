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

package podautoscaler

import (
	"fmt"
	"math"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

const (
	// Usage shoud exceed the tolerance before we start downscale or upscale the pods.
	// TODO: make it a flag or HPA spec element.
	tolerance = 0.1
)

type HorizontalController struct {
	client        client.Interface
	metricsClient metrics.MetricsClient
	eventRecorder record.EventRecorder
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func NewHorizontalController(client client.Interface, metricsClient metrics.MetricsClient) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(client.Events(""))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "horizontal-pod-autoscaler"})

	return &HorizontalController{
		client:        client,
		metricsClient: metricsClient,
		eventRecorder: recorder,
	}
}

func (a *HorizontalController) Run(syncPeriod time.Duration) {
	go util.Until(func() {
		if err := a.reconcileAutoscalers(); err != nil {
			glog.Errorf("Couldn't reconcile horizontal pod autoscalers: %v", err)
		}
	}, syncPeriod, util.NeverStop)
}

func (a *HorizontalController) reconcileAutoscaler(hpa experimental.HorizontalPodAutoscaler) error {
	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Namespace, hpa.Spec.ScaleRef.Name)

	scale, err := a.client.Experimental().Scales(hpa.Spec.ScaleRef.Namespace).Get(hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Name)
	if err != nil {
		a.eventRecorder.Event(&hpa, "FailedGetScale", err.Error())
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas
	currentConsumption, err := a.metricsClient.
		ResourceConsumption(hpa.Spec.ScaleRef.Namespace).
		Get(hpa.Spec.Target.Resource, scale.Status.Selector)

	// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
	if err != nil {
		a.eventRecorder.Event(&hpa, "FailedGetMetrics", err.Error())
		return fmt.Errorf("failed to get metrics for %s: %v", reference, err)
	}

	usageRatio := float64(currentConsumption.Quantity.MilliValue()) / float64(hpa.Spec.Target.Quantity.MilliValue())
	desiredReplicas := int(math.Ceil(usageRatio * float64(currentReplicas)))

	if desiredReplicas < hpa.Spec.MinReplicas {
		desiredReplicas = hpa.Spec.MinReplicas
	}

	// TODO: remove when pod ideling is done.
	if desiredReplicas == 0 {
		desiredReplicas = 1
	}

	if desiredReplicas > hpa.Spec.MaxReplicas {
		desiredReplicas = hpa.Spec.MaxReplicas
	}
	now := time.Now()
	rescale := false

	if desiredReplicas != currentReplicas {
		// Going down only if the usageRatio dropped significantly below the target
		// and there was no rescaling in the last downscaleForbiddenWindow.
		if desiredReplicas < currentReplicas && usageRatio < (1-tolerance) &&
			(hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(downscaleForbiddenWindow).Before(now)) {
			rescale = true
		}

		// Going up only if the usage ratio increased significantly above the target
		// and there was no rescaling in the last upscaleForbiddenWindow.
		if desiredReplicas > currentReplicas && usageRatio > (1+tolerance) &&
			(hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(upscaleForbiddenWindow).Before(now)) {
			rescale = true
		}
	}

	if rescale {
		scale.Spec.Replicas = desiredReplicas
		_, err = a.client.Experimental().Scales(hpa.Namespace).Update(hpa.Spec.ScaleRef.Kind, scale)
		if err != nil {
			a.eventRecorder.Eventf(&hpa, "FailedRescale", "New size: %d; error: %v", desiredReplicas, err.Error())
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		a.eventRecorder.Eventf(&hpa, "SuccessfulRescale", "New size: %d", desiredReplicas)
		glog.Infof("Successfull rescale of %s, old size: %d, new size: %d, usage ratio: %f",
			hpa.Name, currentReplicas, desiredReplicas, usageRatio)
	} else {
		desiredReplicas = currentReplicas
	}

	hpa.Status = experimental.HorizontalPodAutoscalerStatus{
		CurrentReplicas:    currentReplicas,
		DesiredReplicas:    desiredReplicas,
		CurrentConsumption: currentConsumption,
	}
	if rescale {
		now := unversioned.NewTime(now)
		hpa.Status.LastScaleTimestamp = &now
	}

	_, err = a.client.Experimental().HorizontalPodAutoscalers(hpa.Namespace).Update(&hpa)
	if err != nil {
		a.eventRecorder.Event(&hpa, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	return nil
}

func (a *HorizontalController) reconcileAutoscalers() error {
	ns := api.NamespaceAll
	list, err := a.client.Experimental().HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	for _, hpa := range list.Items {
		err := a.reconcileAutoscaler(hpa)
		if err != nil {
			glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
		}
	}
	return nil
}
