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

package math

// MaxInt return the larger of a or b
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinInt return the smaller of a or b
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// BoundedInt bound the value in range [lower, upper]
func BoundedInt(value, lower, upper int) int {
	return MinInt(MaxInt(value, lower), upper)
}

// MaxInt8 return the larger of a or b
func MaxUint8(a, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}

// MinInt8 return the smaller of a or b
func MinUint8(a, b uint8) uint8 {
	if a < b {
		return a
	}
	return b
}

// BoundedInt8 bound the value in range [lower, upper]
func BoundedUint8(value, lower, upper uint8) uint8 {
	return MinUint8(MaxUint8(value, lower), upper)
}

// MaxInt32 return the larger of a or b
func MaxInt32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

// MinInt32 return the smaller of a or b
func MinInt32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// BoundedInt32 bound the value in range [lower, upper]
func BoundedInt32(value, lower, upper int32) int32 {
	return MinInt32(MaxInt32(value, lower), upper)
}

// MaxInt64 return the larger of a or b
func MaxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// MinInt64 return the smaller of a or b
func MinInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// BoundedInt64 bound the value in range [lower, upper]
func BoundedInt64(value, lower, upper int64) int64 {
	return MinInt64(MaxInt64(value, lower), upper)
}
