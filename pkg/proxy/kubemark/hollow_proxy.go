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
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/events"
	proxyapp "k8s.io/kubernetes/cmd/kube-proxy/app"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/utils/ptr"
)

type HollowProxy struct {
	ProxyServer *proxyapp.ProxyServer
}

type FakeProxier struct {
	proxyconfig.NoopNodeHandler
}

func (*FakeProxier) Sync() {}
func (*FakeProxier) SyncLoop() {
	select {}
}
func (*FakeProxier) OnServiceAdd(service *v1.Service)                                 {}
func (*FakeProxier) OnServiceUpdate(oldService, service *v1.Service)                  {}
func (*FakeProxier) OnServiceDelete(service *v1.Service)                              {}
func (*FakeProxier) OnServiceSynced()                                                 {}
func (*FakeProxier) OnEndpointSliceAdd(slice *discoveryv1.EndpointSlice)              {}
func (*FakeProxier) OnEndpointSliceUpdate(oldSlice, slice *discoveryv1.EndpointSlice) {}
func (*FakeProxier) OnEndpointSliceDelete(slice *discoveryv1.EndpointSlice)           {}
func (*FakeProxier) OnEndpointSlicesSynced()                                          {}
func (*FakeProxier) OnServiceCIDRsChanged(_ []string)                                 {}

func NewHollowProxy(
	nodeName string,
	client clientset.Interface,
	eventClient v1core.EventsGetter,
	broadcaster events.EventBroadcaster,
	recorder events.EventRecorder,
) *HollowProxy {
	return &HollowProxy{
		ProxyServer: &proxyapp.ProxyServer{
			Config: &proxyconfigapi.KubeProxyConfiguration{
				Mode:             proxyconfigapi.ProxyMode("fake"),
				ConfigSyncPeriod: metav1.Duration{Duration: 30 * time.Second},
				OOMScoreAdj:      ptr.To[int32](0),
			},

			Client:      client,
			Proxier:     &FakeProxier{},
			Broadcaster: broadcaster,
			Recorder:    recorder,
			NodeRef: &v1.ObjectReference{
				Kind:      "Node",
				Name:      nodeName,
				UID:       types.UID(nodeName),
				Namespace: "",
			},
		},
	}
}

func (hp *HollowProxy) Run() error {

	if err := hp.ProxyServer.Run(context.TODO()); err != nil {
		return fmt.Errorf("Error while running proxy: %w", err)
	}
	return nil
}
