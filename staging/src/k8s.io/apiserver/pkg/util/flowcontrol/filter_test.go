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

package flowcontrol

import (
	"encoding/binary"
	"testing"
)

func BenchmarkCRC(b *testing.B) {
	var buf [8]byte
	var ans uint64
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(buf[:], uint64(i)+ans)
		ans = crcFlowID("benchmarking", string(buf[:]))
	}
}

func BenchmarkSHA(b *testing.B) {
	var buf [8]byte
	var ans uint64
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(buf[:], uint64(i)+ans)
		ans = shaFlowID("benchmarking", string(buf[:]))
	}
}
