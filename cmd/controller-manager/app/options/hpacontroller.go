/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// HPAControllerOptions holds the HPAController options.
type HPAControllerOptions struct {
	HorizontalPodAutoscalerUseRESTClients           bool
	HorizontalPodAutoscalerTolerance                float64
	HorizontalPodAutoscalerDownscaleForbiddenWindow metav1.Duration
	HorizontalPodAutoscalerUpscaleForbiddenWindow   metav1.Duration
	HorizontalPodAutoscalerSyncPeriod               metav1.Duration
}

// AddFlags adds flags related to HPAController for controller manager to the specified FlagSet.
func (o *HPAControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.HorizontalPodAutoscalerSyncPeriod.Duration, "horizontal-pod-autoscaler-sync-period", o.HorizontalPodAutoscalerSyncPeriod.Duration, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&o.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-upscale-delay", o.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "The period since last upscale, before another upscale can be performed in horizontal pod autoscaler.")
	fs.DurationVar(&o.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-downscale-delay", o.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "The period since last downscale, before another downscale can be performed in horizontal pod autoscaler.")
	fs.Float64Var(&o.HorizontalPodAutoscalerTolerance, "horizontal-pod-autoscaler-tolerance", o.HorizontalPodAutoscalerTolerance, "The minimum change (from 1.0) in the desired-to-actual metrics ratio for the horizontal pod autoscaler to consider scaling.")
	fs.BoolVar(&o.HorizontalPodAutoscalerUseRESTClients, "horizontal-pod-autoscaler-use-rest-clients", o.HorizontalPodAutoscalerUseRESTClients, "If set to true, causes the horizontal pod autoscaler controller to use REST clients through the kube-aggregator, instead of using the legacy metrics client through the API server proxy.  This is required for custom metrics support in the horizontal pod autoscaler.")
}

// ApplyTo fills up HPAController config with options.
func (o *HPAControllerOptions) ApplyTo(cfg *componentconfig.HPAControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.HorizontalPodAutoscalerSyncPeriod = o.HorizontalPodAutoscalerSyncPeriod
	cfg.HorizontalPodAutoscalerUpscaleForbiddenWindow = o.HorizontalPodAutoscalerUpscaleForbiddenWindow
	cfg.HorizontalPodAutoscalerDownscaleForbiddenWindow = o.HorizontalPodAutoscalerDownscaleForbiddenWindow
	cfg.HorizontalPodAutoscalerTolerance = o.HorizontalPodAutoscalerTolerance
	cfg.HorizontalPodAutoscalerUseRESTClients = o.HorizontalPodAutoscalerUseRESTClients

	return nil
}

// Validate checks validation of HPAControllerOptions.
func (o *HPAControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
