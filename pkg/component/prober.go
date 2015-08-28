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

package component

import (
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/probe"
	httprobe "k8s.io/kubernetes/pkg/probe/http"
	tcprobe "k8s.io/kubernetes/pkg/probe/tcp"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// Prober checks the status of a component.
type Prober interface {
	Probe(*api.Component, *api.Probe) (probe.Result, error)
}

// Prober helps to check the status of a component.
type prober struct {
	http httprobe.HTTPProber
	tcp  tcprobe.TCPProber
}

// New creates a component Prober
func NewProber() Prober {
	return &prober{
		http: httprobe.New(),
		tcp:  tcprobe.New(),
	}
}

// Probe checks the liveness/readiness of the given component.
func (pb *prober) Probe(c *api.Component, p *api.Probe) (probe.Result, error) {
	if p == nil {
		result := probe.Unknown
		glog.V(1).Infof("Probe (%s) for component %q: not configured", result, c.Name)
		return result, nil
	}

	result, output, err := pb.runProbe(p, c)
	if err != nil {
		glog.V(1).Infof("Probe (error) for component %q: %v", c.Name, err)
		return probe.Unknown, err
	}

	// don't expect success until after InitialDelaySeconds
	if result != probe.Success && time.Now().Unix()-c.CreationTimestamp.Unix() < p.InitialDelaySeconds {
		glog.V(3).Infof(
			"Probe (%s) for component %q: within initial delay (%s)",
			result,
			c.Name,
			time.Duration(p.InitialDelaySeconds)*time.Second,
		)
		return probe.Unknown, nil
	}

	glog.V(1).Infof("Probe (%s) for component %q: %s", result, c.Name, output)
	return result, nil
}

func (pb *prober) runProbe(p *api.Probe, c *api.Component) (probe.Result, string, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		return probe.Unknown, "", fmt.Errorf("Unsupported probe type (exec) for component %q", c.Name)
	}
	if p.HTTPGet != nil {
		scheme := strings.ToLower(string(p.HTTPGet.Scheme))
		host := p.HTTPGet.Host
		if host == "" {
			return probe.Unknown, "", fmt.Errorf("Missing http probe host for component %q", c.Name)
		}
		port, err := extractPort(p.HTTPGet.Port, c)
		if err != nil {
			return probe.Unknown, "", err
		}
		path := p.HTTPGet.Path
		glog.V(4).Infof("HTTP-Probe - Component: %q, Host: %v://%v, Port: %v, Path: %v, Timeout: %v", c.Name, scheme, host, port, path, timeout)
		url := formatURL(scheme, host, port, path)
		return pb.http.Probe(url, timeout)
	}
	if p.TCPSocket != nil {
		port, err := extractPort(p.TCPSocket.Port, c)
		if err != nil {
			return probe.Unknown, "", err
		}
		host := p.TCPSocket.Host
		if host == "" {
			return probe.Unknown, "", fmt.Errorf("Missing tcp probe host for component %q", c.Name)
		}
		glog.V(4).Infof("TCP-Probe - Component: %q, Host: tcp://%v, Port: %v, Timeout: %v", c.Name, host, port, timeout)
		return pb.tcp.Probe(host, port, timeout)
	}
	glog.Warningf("Failed to find probe builder for component: %v", c)
	return probe.Unknown, "", nil
}

func extractPort(param util.IntOrString, c *api.Component) (int, error) {
	port := -1
	switch param.Kind {
	case util.IntstrInt:
		port = param.IntVal
	case util.IntstrString:
		//TODO: support named ports
		return port, fmt.Errorf("Unsuppoerted named port %q for component %q", param.StrVal, c.Name)
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
