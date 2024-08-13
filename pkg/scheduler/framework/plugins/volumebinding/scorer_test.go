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

package volumebinding

import (
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
)

const (
	classHDD = "hdd"
	classSSD = "ssd"
)

func TestScore(t *testing.T) {
	defaultShape := make(helper.FunctionShape, 0, len(defaultShapePoint))
	for _, point := range defaultShapePoint {
		defaultShape = append(defaultShape, helper.FunctionShapePoint{
			Utilization: int64(point.Utilization),
			Score:       int64(point.Score) * (framework.MaxNodeScore / config.MaxCustomPriorityScore),
		})
	}
	type scoreCase struct {
		classResources classResourceMap
		score          int64
	}
	tests := []struct {
		name  string
		shape helper.FunctionShape
		cases []scoreCase
	}{
		{
			name:  "default shape, single class",
			shape: defaultShape,
			cases: []scoreCase{
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
					},
					0,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
					},
					30,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 50,
							Capacity:  100,
						},
					},
					50,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
					},
					100,
				},
			},
		},
		{
			name:  "default shape, multiple classes",
			shape: defaultShape,
			cases: []scoreCase{
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
					},
					0,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
					},
					15,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
					},
					30,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 60,
							Capacity:  100,
						},
					},
					45,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 50,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 50,
							Capacity:  100,
						},
					},
					50,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 50,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
					},
					75,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
					},
					100,
				},
			},
		},
		{
			name: "custom shape, multiple classes",
			shape: helper.FunctionShape{
				{
					Utilization: 50,
					Score:       0,
				},
				{
					Utilization: 80,
					Score:       30,
				},
				{
					Utilization: 100,
					Score:       50,
				},
			},
			cases: []scoreCase{
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
					},
					0,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 0,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
					},
					0,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
					},
					0,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 30,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 60,
							Capacity:  100,
						},
					},
					5,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 50,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
					},
					25,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 90,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 90,
							Capacity:  100,
						},
					},
					40,
				},
				{
					classResourceMap{
						classHDD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
						classSSD: &StorageResource{
							Requested: 100,
							Capacity:  100,
						},
					},
					50,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := buildScorerFunction(tt.shape)
			for _, c := range tt.cases {
				gotScore := f(c.classResources)
				if gotScore != c.score {
					t.Errorf("Expect %d, but got %d", c.score, gotScore)
				}
			}
		})
	}
}
