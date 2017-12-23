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

package fuzzer

import (
	"github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

// Funcs returns the fuzzer functions for the kubeletconfig apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		// provide non-empty values for fields with defaults, so the defaulter doesn't change values during round-trip
		func(obj *kubeproxyconfig.KubeProxyConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.FeatureGates = "foo"
			obj.BindAddress = "foo"
			obj.HealthzBindAddress = "foo:10256"
			obj.MetricsBindAddress = "foo:"
			obj.EnableProfiling = true
			obj.ClusterCIDR = "foo"
			obj.HostnameOverride = "foo"
			obj.ClientConnection = kubeproxyconfig.ClientConnectionConfiguration{
				KubeConfigFile:     "foo",
				AcceptContentTypes: "foo",
				ContentType:        "foo",
				QPS:                float32(5),
				Burst:              10,
			}
			obj.Mode = kubeproxyconfig.ProxyModeIPTables
			obj.IPVS = kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod: metav1.Duration{Duration: 1},
			}
			obj.IPTables = kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: utilpointer.Int32Ptr(0),
				SyncPeriod:    metav1.Duration{Duration: 1},
			}
			obj.OOMScoreAdj = utilpointer.Int32Ptr(0)
			obj.ResourceContainer = "foo"
			obj.UDPIdleTimeout = metav1.Duration{Duration: 1}
			obj.Conntrack = kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: utilpointer.Int32Ptr(2),
				Min:        utilpointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5},
			}
			obj.ConfigSyncPeriod = metav1.Duration{Duration: 1}
		},
	}
}
