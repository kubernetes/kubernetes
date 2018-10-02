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
	utilpointer "k8s.io/utils/pointer"
)

// Funcs returns the fuzzer functions for the kube-proxy apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *kubeproxyconfig.KubeProxyConfiguration, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			obj.BindAddress = fmt.Sprintf("%d.%d.%d.%d", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256))
			obj.ClientConnection.ContentType = c.RandString()
			obj.Conntrack.MaxPerCore = utilpointer.Int32Ptr(c.Int31())
			obj.Conntrack.Min = utilpointer.Int32Ptr(c.Int31())
			obj.Conntrack.TCPCloseWaitTimeout = &metav1.Duration{Duration: time.Duration(c.Int63()) * time.Hour}
			obj.Conntrack.TCPEstablishedTimeout = &metav1.Duration{Duration: time.Duration(c.Int63()) * time.Hour}
			obj.FeatureGates = map[string]bool{c.RandString(): true}
			obj.HealthzBindAddress = fmt.Sprintf("%d.%d.%d.%d:%d", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(65536))
			obj.IPTables.MasqueradeBit = utilpointer.Int32Ptr(c.Int31())
			obj.MetricsBindAddress = fmt.Sprintf("%d.%d.%d.%d:%d", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(65536))
			obj.OOMScoreAdj = utilpointer.Int32Ptr(c.Int31())
			obj.ResourceContainer = "foo"
			obj.ClientConnection.ContentType = "bar"
			obj.NodePortAddresses = []string{"1.2.3.0/24"}
		},
	}
}
