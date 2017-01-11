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
	"net/http"
	"sync"
	"time"

	"github.com/coreos/etcd/pkg/tlsutil"
	"github.com/golang/glog"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/pkg/probe"
	httpprober "k8s.io/kubernetes/pkg/probe/http"
)

const (
	probeTimeOut = 20 * time.Second
)

// TODO: this basic interface is duplicated in N places.  consolidate?
type httpGet interface {
	Get(url string) (*http.Response, error)
}

type ValidatorFn func([]byte) error

type Server struct {
	Addr        string
	Port        int
	Path        string
	EnableHTTPS bool
	CertFile    string
	KeyFile     string
	Validate    ValidatorFn
	Prober      httpprober.HTTPProber
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

func (server *Server) DoServerCheck() (probe.Result, string, error) {
	// setup the prober
	if server.Prober == nil {
		server.setupProber()
	}

	scheme := "http"
	if server.EnableHTTPS {
		scheme = "https"
	}
	url := utilnet.FormatURL(scheme, server.Addr, server.Port, server.Path)

	result, data, err := server.Prober.Probe(url, nil, probeTimeOut)

	if err != nil {
		return probe.Unknown, "", err
	}
	if result == probe.Failure {
		return probe.Failure, string(data), err
	}
	if server.Validate != nil {
		if err := server.Validate([]byte(data)); err != nil {
			return probe.Failure, string(data), err
		}
	}
	return result, string(data), nil
}

// setupProber will set up an appropriate httpprober in a thread-safety way.
func (server *Server) setupProber() {
	f := func() {
		if len(server.CertFile) > 0 && len(server.KeyFile) > 0 {
			if cert, err := tlsutil.NewCert(server.CertFile, server.KeyFile, nil); err != nil {
				glog.Errorf("failed to parse %q and %q: %s", server.CertFile, server.KeyFile, err)
			} else {
				server.Prober = httpprober.NewWithClientCert(cert)
				return
			}
		}
		server.Prober = httpprober.New()
	}
	server.Once.Do(f)
}
