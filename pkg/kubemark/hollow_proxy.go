/*
Copyright 2015 The Kubernetes Authors.

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

package kubemark

import (
	"time"

	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/types"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"

	"github.com/golang/glog"
)

type HollowProxy struct {
	ProxyServer *proxyapp.ProxyServer
}

type FakeProxyHandler struct{}

func (*FakeProxyHandler) OnServiceUpdate(services []api.Service)      {}
func (*FakeProxyHandler) OnEndpointsUpdate(endpoints []api.Endpoints) {}

type FakeProxier struct{}

func (*FakeProxier) OnServiceUpdate(services []api.Service) {}
func (*FakeProxier) Sync()                                  {}
func (*FakeProxier) SyncLoop() {
	select {}
}

func NewHollowProxyOrDie(
	nodeName string,
	client *client.Client,
	endpointsConfig *proxyconfig.EndpointsConfig,
	serviceConfig *proxyconfig.ServiceConfig,
	iptInterface utiliptables.Interface,
	broadcaster record.EventBroadcaster,
	recorder record.EventRecorder,
) *HollowProxy {
	// Create and start Hollow Proxy
	//config := options.NewProxyConfig()
	/* defaults are:
		- client
		  - content type
		  - qps
		  - burst
		- config sync period
		- bind address 0s
		- healthz localhost 10249
		- oom -999
	  - resource container /kube-proxy
	  - iptables sync period 30s
	  - upd idle timeout 250ms
	  - iptables masq bit 14

	*/
	// AG: they were setting this to 0 to override the -999 default, but it's only
	// ever applied by NewDefaultProxyServer, so this is effectively unnecessary
	//config.OOMScoreAdj = util.Int32Ptr(0)

	// reset to ""
	//config.ResourceContainer = ""

	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}

	proxyconfig.NewSourceAPI(
		client,
		30*time.Second,
		serviceConfig.Channel("api"),
		endpointsConfig.Channel("api"),
	)

	return &HollowProxy{
		ProxyServer: &proxyapp.ProxyServer{
			Client:       client,
			IptInterface: iptInterface,
			Proxier:      &FakeProxier{},
			Broadcaster:  broadcaster,
			Recorder:     recorder,
			ProxyMode:    "fake",
			NodeRef:      nodeRef,
		},
	}
}

func (hp *HollowProxy) Run() {
	if err := hp.ProxyServer.Run(); err != nil {
		glog.Fatalf("Error while running proxy: %v\n", err)
	}
}
