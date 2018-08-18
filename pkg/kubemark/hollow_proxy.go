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
	"fmt"
	"net"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/pkg/proxy"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
	utilpointer "k8s.io/utils/pointer"

	"github.com/golang/glog"
)

type HollowProxy struct {
	ProxyServer *proxyapp.ProxyServer
}

type FakeProxier struct{}

func (*FakeProxier) Sync() {}
func (*FakeProxier) SyncLoop() {
	select {}
}
func (*FakeProxier) OnServiceAdd(service *v1.Service)                        {}
func (*FakeProxier) OnServiceUpdate(oldService, service *v1.Service)         {}
func (*FakeProxier) OnServiceDelete(service *v1.Service)                     {}
func (*FakeProxier) OnServiceSynced()                                        {}
func (*FakeProxier) OnEndpointsAdd(endpoints *v1.Endpoints)                  {}
func (*FakeProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {}
func (*FakeProxier) OnEndpointsDelete(endpoints *v1.Endpoints)               {}
func (*FakeProxier) OnEndpointsSynced()                                      {}

func NewHollowProxyOrDie(
	nodeName string,
	client clientset.Interface,
	eventClient v1core.EventsGetter,
	iptInterface utiliptables.Interface,
	sysctl utilsysctl.Interface,
	execer utilexec.Interface,
	broadcaster record.EventBroadcaster,
	recorder record.EventRecorder,
	useRealProxier bool,
	proxierSyncPeriod time.Duration,
	proxierMinSyncPeriod time.Duration,
) (*HollowProxy, error) {
	// Create proxier and service/endpoint handlers.
	var proxier proxy.ProxyProvider
	var serviceHandler proxyconfig.ServiceHandler
	var endpointsHandler proxyconfig.EndpointsHandler

	if useRealProxier {
		// Real proxier with fake iptables, sysctl, etc underneath it.
		//var err error
		proxierIPTables, err := iptables.NewProxier(
			iptInterface,
			sysctl,
			execer,
			proxierSyncPeriod,
			proxierMinSyncPeriod,
			false,
			0,
			"10.0.0.0/8",
			nodeName,
			getNodeIP(client, nodeName),
			recorder,
			nil,
			[]string{},
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create proxier: %v", err)
		}
		proxier = proxierIPTables
		serviceHandler = proxierIPTables
		endpointsHandler = proxierIPTables
	} else {
		proxier = &FakeProxier{}
		serviceHandler = &FakeProxier{}
		endpointsHandler = &FakeProxier{}
	}

	// Create a Hollow Proxy instance.
	nodeRef := &v1.ObjectReference{
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
			Proxier:               proxier,
			Broadcaster:           broadcaster,
			Recorder:              recorder,
			ProxyMode:             "fake",
			NodeRef:               nodeRef,
			OOMScoreAdj:           utilpointer.Int32Ptr(0),
			ResourceContainer:     "",
			ConfigSyncPeriod:      30 * time.Second,
			ServiceEventHandler:   serviceHandler,
			EndpointsEventHandler: endpointsHandler,
		},
	}, nil
}

func (hp *HollowProxy) Run() {
	if err := hp.ProxyServer.Run(); err != nil {
		glog.Fatalf("Error while running proxy: %v\n", err)
	}
}

func getNodeIP(client clientset.Interface, hostname string) net.IP {
	var nodeIP net.IP
	node, err := client.CoreV1().Nodes().Get(hostname, metav1.GetOptions{})
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = utilnode.GetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
