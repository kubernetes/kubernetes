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

package value

// Compare compares floats. The result will be 0 if lhs==rhs, -1 if f <
// rhs, and +1 if f > rhs.
func FloatCompare(lhs, rhs float64) int {
	if lhs > rhs {
		return 1
	} else if lhs < rhs {
		return -1
	}
	return 0
}

// IntCompare compares integers. The result will be 0 if i==rhs, -1 if i <
// rhs, and +1 if i > rhs.
func IntCompare(lhs, rhs int64) int {
	if lhs > rhs {
		return 1
	} else if lhs < rhs {
		return -1
	}
	return 0
}

// Compare compares booleans. The result will be 0 if b==rhs, -1 if b <
// rhs, and +1 if b > rhs.
func BoolCompare(lhs, rhs bool) int {
	if lhs == rhs {
		return 0
	} else if lhs == false {
		return -1
	}
	return 1
}
