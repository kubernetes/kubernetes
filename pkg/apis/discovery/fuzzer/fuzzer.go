/*
Copyright 2019 The Kubernetes Authors.

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
	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery"
)

// Funcs returns the fuzzer functions for the discovery api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *discovery.EndpointSlice, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again

			addressTypes := []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6, discovery.AddressTypeFQDN}
			obj.AddressType = addressTypes[c.Rand.Intn(len(addressTypes))]

			for i, endpointPort := range obj.Ports {
				if endpointPort.Name == nil {
					emptyStr := ""
					obj.Ports[i].Name = &emptyStr
				}

				if endpointPort.Protocol == nil {
					protos := []api.Protocol{api.ProtocolTCP, api.ProtocolUDP, api.ProtocolSCTP}
					obj.Ports[i].Protocol = &protos[c.Rand.Intn(len(protos))]
				}
			}
		},
	}
}
