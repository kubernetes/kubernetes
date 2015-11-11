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

package prober

import (
	"time"

	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	httprobe "k8s.io/kubernetes/pkg/probe/http"
)

func NewTestManagerWithHttpProbe(
	defaultProbePeriod time.Duration,
	statusManager status.Manager,
	readinessManager results.Manager,
	livenessManager results.Manager,
	runner kubecontainer.ContainerCommandRunner,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder,
	httpProber httprobe.HTTPProber) Manager {

	prober := &prober{
		exec:       nil,
		http:       httpProber,
		tcp:        nil,
		runner:     runner,
		refManager: refManager,
		recorder:   recorder,
	}
	return &manager{
		defaultProbePeriod: defaultProbePeriod,
		statusManager:      statusManager,
		prober:             prober,
		readinessManager:   readinessManager,
		livenessManager:    livenessManager,
		workers:            make(map[probeKey]*worker),
	}

}
