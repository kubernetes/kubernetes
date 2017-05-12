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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/record"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/util"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"

	"github.com/golang/glog"
)

type HollowProxy struct {
	ProxyServer *proxyapp.ProxyServer
}

func NewHollowProxyOrDie(
	nodeName string,
	client clientset.Interface,
	eventClient v1core.EventsGetter,
	iptInterface utiliptables.Interface,
	sysctl utilsysctl.Interface,
	execer utilexec.Interface,
	broadcaster record.EventBroadcaster,
	recorder record.EventRecorder,
) (*HollowProxy, error) {
	// Create a proxier with fake iptables underneath it.
	proxier, err := iptables.NewProxier(
		iptInterface,
		sysctl,
		execer,
		30*time.Second,
		5*time.Second,
		false,
		0,
		"10.0.0.0/8",
		nodeName,
		getNodeIP(client, nodeName),
		recorder,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create proxier: %v", err)
	}

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
			Proxier:               proxier,
			Broadcaster:           broadcaster,
			Recorder:              recorder,
			ProxyMode:             "fake",
			NodeRef:               nodeRef,
			OOMScoreAdj:           util.Int32Ptr(0),
			ResourceContainer:     "",
			ConfigSyncPeriod:      30 * time.Second,
			ServiceEventHandler:   proxier,
			EndpointsEventHandler: proxier,
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
	node, err := client.Core().Nodes().Get(hostname, metav1.GetOptions{})
	if err != nil {
		glog.Warningf("Failed to retrieve node info: %v", err)
		return nil
	}
	nodeIP, err = nodeutil.InternalGetNodeHostIP(node)
	if err != nil {
		glog.Warningf("Failed to retrieve node IP: %v", err)
		return nil
	}
	return nodeIP
}
