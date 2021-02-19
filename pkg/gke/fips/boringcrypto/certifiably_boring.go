// +build boringcrypto,linux,amd64  // The `boringcrypto` build tag is automatically satisfied when using the go-boringcrypto toolchain as of release `b6`

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

package boringcrypto

import (
	"crypto/boring" // Only present in Go+Boringcrypto releases since `b6`
	"os"
)

// "This GKE-distributed binary expects to be utilizing boringcrypto" as raw ASCII bytes.
//
// Importing a Go+BoringCrypto-exclusive library and using an ASCII byte array allows for us to, in theory, easily `grep` compiled binaries to verify that a binary file was compiled against BoringCrypto.
var sentinelBoringCryptoMessageBytes = [...]byte{0x54, 0x68, 0x69, 0x73, 0x20, 0x47, 0x4b, 0x45, 0x2d, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x20, 0x62, 0x69, 0x6e, 0x61, 0x72, 0x79, 0x20, 0x65, 0x78, 0x70, 0x65, 0x63, 0x74, 0x73, 0x20, 0x74, 0x6f, 0x20, 0x62, 0x65, 0x20, 0x75, 0x74, 0x69, 0x6c, 0x69, 0x7a, 0x69, 0x6e, 0x67, 0x20, 0x62, 0x6f, 0x72, 0x69, 0x6e, 0x67, 0x63, 0x72, 0x79, 0x70, 0x74, 0x6f}

func init() {
	// A runtime check is necessary to guarantee that the compiler won't omit the sentinel message at compile time.
	explodeAnyway := os.Getenv("GKE_EXPLODE_ON_INIT_IF_BORINGCRYPTO_IS_ENABLED") != ""
	if !boring.Enabled() || explodeAnyway {
		panic(string(sentinelBoringCryptoMessageBytes[:]))
	}
}
