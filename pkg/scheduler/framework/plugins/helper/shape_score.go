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

package helper

// FunctionShape represents a collection of FunctionShapePoint.
type FunctionShape []FunctionShapePoint

// FunctionShapePoint represents a shape point.
type FunctionShapePoint struct {
	// Utilization is function argument.
	Utilization int64
	// Score is function value.
	Score int64
}

// BuildBrokenLinearFunction creates a function which is built using linear segments. Segments are defined via shape array.
// Shape[i].Utilization slice represents points on "Utilization" axis where different segments meet.
// Shape[i].Score represents function values at meeting points.
//
// function f(p) is defined as:
//   shape[0].Score for p < f[0].Utilization
//   shape[i].Score for p == shape[i].Utilization
//   shape[n-1].Score for p > shape[n-1].Utilization
// and linear between points (p < shape[i].Utilization)
func BuildBrokenLinearFunction(shape FunctionShape) func(int64) int64 {
	return func(p int64) int64 {
		for i := 0; i < len(shape); i++ {
			if p <= int64(shape[i].Utilization) {
				if i == 0 {
					return shape[0].Score
				}
				return shape[i-1].Score + (shape[i].Score-shape[i-1].Score)*(p-shape[i-1].Utilization)/(shape[i].Utilization-shape[i-1].Utilization)
			}
		}
		return shape[len(shape)-1].Score
	}
}
