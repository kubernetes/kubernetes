/*
Copyright 2014 The Kubernetes Authors.

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

package componentstatus

import (
	"context"
	"crypto/tls"
	"fmt"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/kubernetes/pkg/probe"
	httpprober "k8s.io/kubernetes/pkg/probe/http"
)

const (
	probeTimeOut = 20 * time.Second
)

type ValidatorFn func([]byte) error

type Server interface {
	DoServerCheck() (probe.Result, string, error)
}

type HttpServer struct {
	Addr        string
	Port        int
	Path        string
	EnableHTTPS bool
	TLSConfig   *tls.Config
	Validate    ValidatorFn
	Prober      httpprober.Prober
	Once        sync.Once
}

type ServerStatus struct {
	// +optional
	Component string `json:"component,omitempty"`
	// +optional
	Health string `json:"health,omitempty"`
	// +optional
	HealthCode probe.Result `json:"healthCode,omitempty"`
	// +optional
	Msg string `json:"msg,omitempty"`
	// +optional
	Err string `json:"err,omitempty"`
}

func (server *HttpServer) DoServerCheck() (probe.Result, string, error) {
	// setup the prober
	server.Once.Do(func() {
		if server.Prober != nil {
			return
		}
		const followNonLocalRedirects = true
		server.Prober = httpprober.NewWithTLSConfig(server.TLSConfig, followNonLocalRedirects)
	})

	scheme := "http"
	if server.EnableHTTPS {
		scheme = "https"
	}
	url := utilnet.FormatURL(scheme, server.Addr, server.Port, server.Path)

	req, err := httpprober.NewProbeRequest(url, nil)
	if err != nil {
		return probe.Unknown, "", fmt.Errorf("failed to construct probe request: %w", err)
	}
	result, data, err := server.Prober.Probe(req, probeTimeOut)

	if err != nil {
		return probe.Unknown, "", err
	}
	if result == probe.Failure {
		return probe.Failure, data, err
	}
	if server.Validate != nil {
		if err := server.Validate([]byte(data)); err != nil {
			return probe.Failure, data, err
		}
	}
	return result, data, nil
}

type EtcdServer struct {
	storagebackend.Config
}

func (server *EtcdServer) DoServerCheck() (probe.Result, string, error) {
	prober, err := factory.CreateProber(server.Config)
	if err != nil {
		return probe.Failure, "", err
	}
	defer prober.Close()

	ctx, cancel := context.WithTimeout(context.Background(), probeTimeOut)
	defer cancel()
	err = prober.Probe(ctx)
	if err != nil {
		return probe.Failure, "", err
	}
	return probe.Success, "ok", err
}
