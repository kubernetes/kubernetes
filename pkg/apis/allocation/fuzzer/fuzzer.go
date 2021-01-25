/*
Copyright 2021 The Kubernetes Authors.

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
	"math/big"
	"net"

	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/allocation"
)

// Funcs returns the fuzzer functions for the allocation api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *allocation.IPRange, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again

			// length in bytes of the IP Family: IPv4: 4 bytes IPv6: 16 bytes
			ipLen := []int{4, 16}

			ipBytes := ipLen[c.Rand.Intn(2)]
			ip := generateRandomIP(ipBytes, c)
			maskLen := ipBytes * 8
			mask := net.CIDRMask(c.Rand.Intn(maskLen), maskLen)
			cidr := &net.IPNet{IP: ip, Mask: mask}
			obj.Spec.Range = cidr.String()
		},
		func(obj *allocation.IPAddress, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again

			// length in bytes of the IP Family: IPv4: 4 bytes IPv6: 16 bytes
			ipLen := []int{4, 16}

			ipBytes := ipLen[c.Rand.Intn(2)]
			ip := generateRandomIP(ipBytes, c)
			obj.Name = big.NewInt(0).SetBytes(ip.To16()).String()
			obj.Spec.Address = ip.String()
		},
	}
}

func generateRandomIP(n int, c fuzz.Continue) net.IP {
	var ip net.IP
	for i := 0; i < n; i++ {
		number := uint8(c.Rand.Intn(255))
		ip = append(ip, number)
	}
	return ip
}
