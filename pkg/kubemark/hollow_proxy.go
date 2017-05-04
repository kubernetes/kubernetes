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

	"k8s.io/apimachinery/pkg/types"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/record"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/util"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"

	"github.com/golang/glog"
)

type HollowProxy struct {
	ProxyServer *proxyapp.ProxyServer
}

type FakeProxyHandler struct{}

func (*FakeProxyHandler) OnServiceAdd(service *api.Service)                        {}
func (*FakeProxyHandler) OnServiceUpdate(oldService, service *api.Service)         {}
func (*FakeProxyHandler) OnServiceDelete(service *api.Service)                     {}
func (*FakeProxyHandler) OnServiceSynced()                                         {}
func (*FakeProxyHandler) OnEndpointsAdd(endpoints *api.Endpoints)                  {}
func (*FakeProxyHandler) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {}
func (*FakeProxyHandler) OnEndpointsDelete(endpoints *api.Endpoints)               {}
func (*FakeProxyHandler) OnEndpointsSynced()                                       {}

type FakeProxier struct{}

func (*FakeProxier) Sync() {}
func (*FakeProxier) SyncLoop() {
	select {}
}

func NewHollowProxyOrDie(
	nodeName string,
	client clientset.Interface,
	eventClient v1core.EventsGetter,
	iptInterface utiliptables.Interface,
	broadcaster record.EventBroadcaster,
	recorder record.EventRecorder,
) *HollowProxy {
	// Create and start Hollow Proxy
	nodeRef := &clientv1.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}

	return &HollowProxy{
		ProxyServer: &proxyapp.ProxyServer{
			Client:                client,
			EventClient:           eventClient,
			IptInterface:          iptInterface,
			Proxier:               &FakeProxier{},
			Broadcaster:           broadcaster,
			Recorder:              recorder,
			ProxyMode:             "fake",
			NodeRef:               nodeRef,
			OOMScoreAdj:           util.Int32Ptr(0),
			ResourceContainer:     "",
			ConfigSyncPeriod:      30 * time.Second,
			ServiceEventHandler:   &FakeProxyHandler{},
			EndpointsEventHandler: &FakeProxyHandler{},
		},
	}
}

func (hp *HollowProxy) Run() {
	if err := hp.ProxyServer.Run(); err != nil {
		glog.Fatalf("Error while running proxy: %v\n", err)
	}
}
