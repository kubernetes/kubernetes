/*
Copyright 2017 The Kubernetes Authors.

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

package plugins

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"

	"github.com/golang/glog"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/kubectl"
)

// Proxier encapsulates a proxy server to the REST API that can be explicitly started/stopped.
type Proxier interface {
	Start(started chan bool) error
	Stop() error
}

type bearerProxier struct {
	clientConfig *restclient.Config
	listener     net.Listener
	ip           string
	port         int
	started      bool
	bearer       string
}

// NewBearerProxier creates a proxier that requires a bearer token to access the API. The
// token is not the same used in the underlying authentication, it's specific for the proxy
// and the token value is exposed in the Env() method.
func NewBearerProxier(clientConfig *restclient.Config) *bearerProxier {
	return &bearerProxier{
		clientConfig: clientConfig,
		ip:           "127.0.0.1",
	}
}

func (p *bearerProxier) Start(started chan bool) error {
	token, err := generateBearer()
	if err != nil {
		return err
	}
	p.bearer = token

	filter := &kubectl.FilterServer{
		AcceptPaths:              kubectl.MakeRegexpArrayOrDie(kubectl.DefaultPathAcceptRE),
		RejectPaths:              kubectl.MakeRegexpArrayOrDie(kubectl.DefaultPathRejectRE),
		AcceptHosts:              kubectl.MakeRegexpArrayOrDie(kubectl.DefaultHostAcceptRE),
		RejectMethods:            kubectl.MakeRegexpArrayOrDie(kubectl.DefaultMethodRejectRE),
		AcceptProxyAuthorization: kubectl.MakeRegexpArrayOrDie(fmt.Sprintf("^Bearer %s$", p.bearer)),
	}

	server, err := kubectl.NewProxyServer("", "/", "", filter, p.clientConfig)
	if err != nil {
		return err
	}

	p.listener, err = server.Listen(p.ip, p.port)
	if err != nil {
		return err
	}
	p.started = true

	glog.V(8).Infof("Starting to serve API proxy for plugin on: %s", p.listener.Addr())
	go func() {
		started <- true
		err := server.ServeOnListenerState(p.listener, func(conn net.Conn, state http.ConnState) {
			glog.V(9).Infof("API proxy is: %s", state)
		})
		if err != nil {
			glog.Fatal(fmt.Errorf("Unable to start API proxy: %v", err))
		}
	}()

	return nil
}

func (p *bearerProxier) Stop() error {
	glog.V(8).Infof("Closing API proxy for plugin: %s", p.listener.Addr())
	err := p.listener.Close()
	p.started = false
	return err
}

func (p *bearerProxier) Env() (EnvList, error) {
	if !p.started {
		return EnvList{}, nil
	}

	return EnvList{
		{"KUBECTL_PLUGINS_API_PROXY_ADDR", p.listener.Addr().String()},
		{"KUBECTL_PLUGINS_API_PROXY_AUTH_TOKEN", p.bearer},
		{"KUBECTL_PLUGINS_API_PROXY_AUTH_HEADER", fmt.Sprintf("Proxy-Authorization: Bearer %s", p.bearer)},
	}, nil
}

func generateBearer() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
