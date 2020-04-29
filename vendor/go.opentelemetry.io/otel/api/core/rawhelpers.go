// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package core

import (
	"math"
	"unsafe"
)

func boolToRaw(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

func rawToBool(r uint64) bool {
	return r != 0
}

func int64ToRaw(i int64) uint64 {
	return uint64(i)
}

func rawToInt64(r uint64) int64 {
	return int64(r)
}

func uint64ToRaw(u uint64) uint64 {
	return u
}

func rawToUint64(r uint64) uint64 {
	return r
}

func float64ToRaw(f float64) uint64 {
	return math.Float64bits(f)
}

func rawToFloat64(r uint64) float64 {
	return math.Float64frombits(r)
}

func int32ToRaw(i int32) uint64 {
	return uint64(i)
}

func rawToInt32(r uint64) int32 {
	return int32(r)
}

func uint32ToRaw(u uint32) uint64 {
	return uint64(u)
}

func rawToUint32(r uint64) uint32 {
	return uint32(r)
}

func float32ToRaw(f float32) uint64 {
	return uint32ToRaw(math.Float32bits(f))
}

func rawToFloat32(r uint64) float32 {
	return math.Float32frombits(rawToUint32(r))
}

func rawPtrToFloat64Ptr(r *uint64) *float64 {
	return (*float64)(unsafe.Pointer(r))
}

func rawPtrToInt64Ptr(r *uint64) *int64 {
	return (*int64)(unsafe.Pointer(r))
}

func rawPtrToUint64Ptr(r *uint64) *uint64 {
	return r
}
