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

package internal

import (
	"math"
	"unsafe"
)

func BoolToRaw(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

func RawToBool(r uint64) bool {
	return r != 0
}

func Int64ToRaw(i int64) uint64 {
	return uint64(i)
}

func RawToInt64(r uint64) int64 {
	return int64(r)
}

func Float64ToRaw(f float64) uint64 {
	return math.Float64bits(f)
}

func RawToFloat64(r uint64) float64 {
	return math.Float64frombits(r)
}

func RawPtrToFloat64Ptr(r *uint64) *float64 {
	return (*float64)(unsafe.Pointer(r))
}

func RawPtrToInt64Ptr(r *uint64) *int64 {
	return (*int64)(unsafe.Pointer(r))
}
