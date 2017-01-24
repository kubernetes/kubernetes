/*
Copyright 2016 The Kubernetes Authors.

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

package localkube

import (
	kubeproxy "k8s.io/kubernetes/cmd/kube-proxy/app"
	"k8s.io/kubernetes/cmd/kube-proxy/app/options"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

var (
	MasqueradeBit = int32(14)
	OOMScoreAdj   = int32(qos.KubeProxyOOMScoreAdj)
)

func (lk LocalkubeServer) NewProxyServer() Server {
	return NewSimpleServer("proxy", serverInterval, StartProxyServer(lk))
}

func StartProxyServer(lk LocalkubeServer) func() error {
	config := options.NewProxyConfig()

	// master details
	config.Master = lk.GetAPIServerInsecureURL()

	config.Mode = componentconfig.ProxyModeIPTables

	// defaults
	config.OOMScoreAdj = &OOMScoreAdj
	config.IPTablesMasqueradeBit = &MasqueradeBit

	lk.SetExtraConfigForComponent("proxy", &config)

	server, err := kubeproxy.NewProxyServerDefault(config)
	if err != nil {
		panic(err)
	}

	return func() error {
		return server.Run()
	}
}
