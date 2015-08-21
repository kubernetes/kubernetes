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
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/component"
	"k8s.io/kubernetes/pkg/probe"
	httprobe "k8s.io/kubernetes/pkg/probe/http"
	tcprobe "k8s.io/kubernetes/pkg/probe/tcp"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

const maxProbeRetries = 3

// Prober checks the healthiness of a component.
type Prober interface {
	Probe(component *api.Component) (probe.Result, error)
}

// Prober helps to check the liveness/readiness of a component.
type prober struct {
	http httprobe.HTTPProber
	tcp  tcprobe.TCPProber

	readinessManager *component.ReadinessManager
}

// New creates a component Prober
func New(
	readinessManager *component.ReadinessManager,
) Prober {
	return &prober{
		http: httprobe.New(),
		tcp:  tcprobe.New(),

		readinessManager: readinessManager,
	}
}

// Probe checks the liveness/readiness of the given component.
func (pb *prober) Probe(component *api.Component) (probe.Result, error) {
	pb.probeReadiness(component)
	return pb.probeLiveness(component)
}

// probeLiveness probes the liveness of a component.
// If the initalDelay since component creation on liveness probe has not passed the probe will return probe.Success.
func (pb *prober) probeLiveness(component *api.Component) (probe.Result, error) {
	var live probe.Result
	var output string
	var err error
	p := component.Spec.LivenessProbe
	if p == nil {
		return probe.Success, nil
	}
	if time.Now().Unix()-component.CreationTimestamp.Unix() < p.InitialDelaySeconds {
		return probe.Success, nil
	} else {
		live, output, err = pb.runProbeWithRetries(p, component, maxProbeRetries)
	}
	if err != nil || live != probe.Success {
		// Liveness failed in one way or another.
		if err != nil {
			glog.V(1).Infof("Liveness probe for %q errored: %v", component.Name, err)
			return probe.Unknown, err
		} else { // live != probe.Success
			glog.V(1).Infof("Liveness probe for %q failed (%v): %s", component.Name, live, output)
			return live, nil
		}
	}
	glog.V(3).Infof("Liveness probe for %q succeeded", component.Name)
	return probe.Success, nil
}

// probeReadiness probes and sets the readiness of a component.
// If the initial delay on the readiness probe has not passed, we set readiness to false.
func (pb *prober) probeReadiness(component *api.Component) {
	var ready probe.Result
	var output string
	var err error
	p := component.Spec.ReadinessProbe
	if p == nil {
		ready = probe.Success
	} else if time.Now().Unix()-component.CreationTimestamp.Unix() < p.InitialDelaySeconds {
		ready = probe.Failure
	} else {
		ready, output, err = pb.runProbeWithRetries(p, component, maxProbeRetries)
	}
	if err != nil || ready == probe.Failure {
		// Readiness failed in one way or another.
		if err != nil {
			glog.V(1).Infof("readiness probe for %q errored: %v", component.Name, err)
			return
		} else { // ready != probe.Success
			glog.V(1).Infof("Readiness probe for %q failed (%v): %s", component.Name, ready, output)
			return
		}
	}
	if err != nil {
		glog.V(1).Infof("readiness probe for %q errored: %v", component.Name, err)
		return
	}
	if ready == probe.Failure {
		glog.V(1).Infof("Readiness probe for %q failed (%v): %s", component.Name, ready, output)
		return
	}
	if ready == probe.Success {
		pb.readinessManager.SetReadiness(component.Name, true)
	}

	glog.V(3).Infof("Readiness probe for %q succeeded", component.Name)
}

// runProbeWithRetries tries to probe the component in a finite loop, it returns the last result
// if it never succeeds.
func (pb *prober) runProbeWithRetries(p *api.Probe, component *api.Component, retries int) (probe.Result, string, error) {
	var err error
	var result probe.Result
	var output string
	for i := 0; i < retries; i++ {
		result, output, err = pb.runProbe(p, component)
		if result == probe.Success {
			return probe.Success, output, nil
		}
	}
	return result, output, err
}

func (pb *prober) runProbe(p *api.Probe, component *api.Component) (probe.Result, string, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		return probe.Unknown, "", fmt.Errorf("Unsupported probe type (exec) for component: %v", component.Name)
	}
	if p.HTTPGet != nil {
		scheme := strings.ToLower(string(p.HTTPGet.Scheme))
		host := p.HTTPGet.Host
		if host == "" {
			return probe.Unknown, "", fmt.Errorf("Missing http probe host for component: %v", component.Name)
		}
		port, err := extractPort(p.HTTPGet.Port, component)
		if err != nil {
			return probe.Unknown, "", err
		}
		path := p.HTTPGet.Path
		glog.V(4).Infof("HTTP-Probe - Component: %q, Host: %v://%v, Port: %v, Path: %v", component.Name, scheme, host, port, path)
		url := formatURL(scheme, host, port, path)
		return pb.http.Probe(url, timeout)
	}
	if p.TCPSocket != nil {
		port, err := extractPort(p.TCPSocket.Port, component)
		if err != nil {
			return probe.Unknown, "", err
		}
		host := p.TCPSocket.Host
		if host == "" {
			return probe.Unknown, "", fmt.Errorf("Missing tcp probe host for component: %v", component.Name)
		}
		glog.V(4).Infof("TCP-Probe - Component: %q, Host: tcp://%v, Port: %v, Timeout: %v", component.Name, host, port, timeout)
		return pb.tcp.Probe(host, port, timeout)
	}
	glog.Warningf("Failed to find probe builder for component: %v", component)
	return probe.Unknown, "", nil
}

func extractPort(param util.IntOrString, component *api.Component) (int, error) {
	port := -1
	switch param.Kind {
	case util.IntstrInt:
		port = param.IntVal
	case util.IntstrString:
		//TODO: support named ports
		return port, fmt.Errorf("Unsuppoerted named port %q for component %q", param.StrVal, component.Name)
	default:
		return port, fmt.Errorf("IntOrString had no kind: %+v", param)
	}
	if port > 0 && port < 65536 {
		return port, nil
	}
	return port, fmt.Errorf("invalid port number: %v", port)
}

// formatURL formats a URL from args.  For testability.
func formatURL(scheme string, host string, port int, path string) *url.URL {
	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, strconv.Itoa(port)),
		Path:   path,
	}
}
