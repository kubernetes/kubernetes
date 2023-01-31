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
	"fmt"

	"github.com/spf13/pflag"

	poautosclerconfig "k8s.io/kubernetes/pkg/controller/podautoscaler/config"
)

// HPAControllerOptions holds the HPAController options.
type HPAControllerOptions struct {
	*poautosclerconfig.HPAControllerConfiguration
}

// AddFlags adds flags related to HPAController for controller manager to the specified FlagSet.
func (o *HPAControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentHorizontalPodAutoscalerSyncs, "concurrent-horizontal-pod-autoscaler-syncs", o.ConcurrentHorizontalPodAutoscalerSyncs, "The number of horizontal pod autoscaler objects that are allowed to sync concurrently. Larger number = more responsive horizontal pod autoscaler objects processing, but more CPU (and network) load.")
	fs.DurationVar(&o.HorizontalPodAutoscalerSyncPeriod.Duration, "horizontal-pod-autoscaler-sync-period", o.HorizontalPodAutoscalerSyncPeriod.Duration, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&o.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-upscale-delay", o.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "The period since last upscale, before another upscale can be performed in horizontal pod autoscaler.")
	fs.MarkDeprecated("horizontal-pod-autoscaler-upscale-delay", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&o.HorizontalPodAutoscalerDownscaleStabilizationWindow.Duration, "horizontal-pod-autoscaler-downscale-stabilization", o.HorizontalPodAutoscalerDownscaleStabilizationWindow.Duration, "The period for which autoscaler will look backwards and not scale down below any recommendation it made during that period.")
	fs.DurationVar(&o.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-downscale-delay", o.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "The period since last downscale, before another downscale can be performed in horizontal pod autoscaler.")
	fs.MarkDeprecated("horizontal-pod-autoscaler-downscale-delay", "This flag is currently no-op and will be deleted.")
	fs.Float64Var(&o.HorizontalPodAutoscalerTolerance, "horizontal-pod-autoscaler-tolerance", o.HorizontalPodAutoscalerTolerance, "The minimum change (from 1.0) in the desired-to-actual metrics ratio for the horizontal pod autoscaler to consider scaling.")
	fs.DurationVar(&o.HorizontalPodAutoscalerCPUInitializationPeriod.Duration, "horizontal-pod-autoscaler-cpu-initialization-period", o.HorizontalPodAutoscalerCPUInitializationPeriod.Duration, "The period after pod start when CPU samples might be skipped.")
	fs.DurationVar(&o.HorizontalPodAutoscalerInitialReadinessDelay.Duration, "horizontal-pod-autoscaler-initial-readiness-delay", o.HorizontalPodAutoscalerInitialReadinessDelay.Duration, "The period after pod start during which readiness changes will be treated as initial readiness.")
}

// ApplyTo fills up HPAController config with options.
func (o *HPAControllerOptions) ApplyTo(cfg *poautosclerconfig.HPAControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentHorizontalPodAutoscalerSyncs = o.ConcurrentHorizontalPodAutoscalerSyncs
	cfg.HorizontalPodAutoscalerSyncPeriod = o.HorizontalPodAutoscalerSyncPeriod
	cfg.HorizontalPodAutoscalerDownscaleStabilizationWindow = o.HorizontalPodAutoscalerDownscaleStabilizationWindow
	cfg.HorizontalPodAutoscalerTolerance = o.HorizontalPodAutoscalerTolerance
	cfg.HorizontalPodAutoscalerCPUInitializationPeriod = o.HorizontalPodAutoscalerCPUInitializationPeriod
	cfg.HorizontalPodAutoscalerInitialReadinessDelay = o.HorizontalPodAutoscalerInitialReadinessDelay
	cfg.HorizontalPodAutoscalerUpscaleForbiddenWindow = o.HorizontalPodAutoscalerUpscaleForbiddenWindow
	cfg.HorizontalPodAutoscalerDownscaleForbiddenWindow = o.HorizontalPodAutoscalerDownscaleForbiddenWindow

	return nil
}

// Validate checks validation of HPAControllerOptions.
func (o *HPAControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if o.ConcurrentHorizontalPodAutoscalerSyncs < 1 {
		errs = append(errs, fmt.Errorf("concurrent-horizontal-pod-autoscaler-syncs must be greater than 0, but got %d", o.ConcurrentHorizontalPodAutoscalerSyncs))
	}
	return errs
}
