/*
Copyright 2023 The Kubernetes Authors.

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
	"context"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/probe"
	httpprobe "k8s.io/kubernetes/pkg/probe/http"
	"net/http"
	"time"
)

// httpProbeRunner holds state and logic for running HTTP probes.
type httpProbeRunner struct {
	host        string
	timeout     time.Duration
	httpRequest *http.Request
}

// newHTTPProbeRunner returns httpProbeRunner which implements probeRunner.
func newHTTPProbeRunner() *httpProbeRunner {
	return &httpProbeRunner{}
}

func (hp *httpProbeRunner) sync(container v1.Container, status v1.PodStatus, probeType probeType) error {
	if hp.host == status.PodIP {
		// no sync required, podIP hasn't changed.
		return nil
	}

	// no need to handle error here, factory will handle this case.
	probeSpec, _ := getProbeSpecFromContainer(container, probeType)

	timeout := time.Duration(probeSpec.TimeoutSeconds) * time.Second
	req, err := httpprobe.NewRequestForHTTPGetAction(probeSpec.HTTPGet, &container, status.PodIP, "probe")
	if err != nil {
		return err
	}

	// persist
	hp.host = status.PodIP
	hp.timeout = timeout
	hp.httpRequest = req
	return err
}

func (hp *httpProbeRunner) run(_ context.Context, container v1.Container, status v1.PodStatus, probeType probeType, prober *prober) (probe.Result, string, error) {
	err := hp.sync(container, status, probeType)
	if err != nil {
		return probe.Unknown, "", err
	}

	klog.V(4).InfoS("HTTP-Probe", "scheme", hp.httpRequest.URL.Scheme, "host", hp.host,
		"port", hp.httpRequest.URL.Port(), "path", hp.httpRequest.URL.Path, "timeout", hp.timeout,
		"headers", hp.httpRequest.Header)
	return prober.http.Probe(hp.httpRequest, hp.timeout)
}
