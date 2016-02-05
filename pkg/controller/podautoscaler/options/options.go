/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"time"

	"github.com/spf13/pflag"
)

type PodAutoscalerOptions struct {
	HorizontalPodAutoscalerSyncPeriod time.Duration
}

func NewPodAutoscalerOptions() PodAutoscalerOptions {
	return PodAutoscalerOptions{
		HorizontalPodAutoscalerSyncPeriod: 30 * time.Second,
	}
}

func (o *PodAutoscalerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.DurationVar(&o.HorizontalPodAutoscalerSyncPeriod, "horizontal-pod-autoscaler-sync-period", o.HorizontalPodAutoscalerSyncPeriod,
		"The period for syncing the number of pods in horizontal pod autoscaler.",
	)
}
