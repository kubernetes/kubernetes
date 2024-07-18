/*
Copyright 2018 The Kubernetes Authors.

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

package fuzzer

import (
	"fmt"
	"time"

	"github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/ptr"
)

func getRandomDualStackIPPair(c fuzz.Continue) []string {
	ipv4 := fmt.Sprintf("%d.%d.%d.%d", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256))

	byts := make([]byte, 16)
	c.Read(byts)
	ipv6 := fmt.Sprintf("%x:%x:%x:%x:%x:%x:%x:%x",
		byts[0:2], byts[2:4], byts[4:6], byts[6:8],
		byts[8:10], byts[10:12], byts[12:14], byts[14:16])
	return []string{ipv4, ipv6}
}

// Funcs returns the fuzzer functions for the kube-proxy apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *kubeproxyconfig.KubeProxyConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.NodeIPOverride = getRandomDualStackIPPair(c)[:1+c.Intn(2)]
			obj.ClientConnection.ContentType = c.RandString()
			obj.Linux.Conntrack.MaxPerCore = ptr.To(c.Int31())
			obj.Linux.Conntrack.Min = ptr.To(c.Int31())
			obj.Linux.Conntrack.TCPCloseWaitTimeout = &metav1.Duration{Duration: time.Duration(c.Int63()) * time.Hour}
			obj.Linux.Conntrack.TCPEstablishedTimeout = &metav1.Duration{Duration: time.Duration(c.Int63()) * time.Hour}
			obj.FeatureGates = map[string]bool{c.RandString(): true}
			obj.HealthzBindAddresses = []string{fmt.Sprintf("%d.%d.%d.%d/32", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256))}
			obj.HealthzBindPort = 1000 + c.Int31n(65536)
			obj.IPTables.MasqueradeBit = ptr.To(c.Int31())
			obj.IPTables.LocalhostNodePorts = ptr.To(c.RandBool())
			obj.NFTables.MasqueradeBit = ptr.To(c.Int31())
			obj.IPVS.MasqueradeBit = obj.IPTables.MasqueradeBit
			obj.MetricsBindAddresses = []string{fmt.Sprintf("%d.%d.%d.%d/32", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256))}
			obj.MetricsBindPort = 1000 + c.Int31n(65536)
			obj.Linux.OOMScoreAdj = ptr.To(c.Int31())
			obj.ClientConnection.ContentType = "bar"
			obj.NodePortAddresses = []string{"1.2.3.0/24"}
			if obj.Logging.Format == "" {
				obj.Logging.Format = "text"
			}
		},
	}
}
