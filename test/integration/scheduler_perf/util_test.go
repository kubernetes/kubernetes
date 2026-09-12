/*
Copyright 2015 The Kubernetes Authors.

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

package benchmark

import (
	"reflect"
	"testing"
)

func Test_uniqueLVCombos(t *testing.T) {
	type args struct {
		lvs []*labelValues
	}
	tests := []struct {
		name string
		args args
		want []map[string]string
	}{
		{
			name: "empty input",
			args: args{
				lvs: []*labelValues{},
			},
			want: []map[string]string{{}},
		},
		{
			name: "single label, multiple values",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1", "a2"}},
				},
			},
			want: []map[string]string{
				{"A": "a1"},
				{"A": "a2"},
			},
		},
		{
			name: "multiple labels, single value each",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1"}},
					{"B", []string{"b1"}},
				},
			},
			want: []map[string]string{
				{"A": "a1", "B": "b1"},
			},
		},
		{
			name: "multiple labels, multiple values",
			args: args{
				lvs: []*labelValues{
					{"A", []string{"a1", "a2"}},
					{"B", []string{"b1", "b2"}},
				},
			},
			want: []map[string]string{
				{"A": "a1", "B": "b1"},
				{"A": "a1", "B": "b2"},
				{"A": "a2", "B": "b1"},
				{"A": "a2", "B": "b2"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := uniqueLVCombos(tt.args.lvs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("uniqueLVCombos() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test_separateClientConnections verifies that kubescheduler under test and test framework use
// two separate client connections (and configurations) to the API server, and that connection
// parameters set on workload or testcase level are correctly interpreted.
func Test_separateClientConnections(t *testing.T) {
	var (
		tcQPS   float32 = 10.0
		tcBurst         = 20
		wQPS    float32 = 123.45
		wBurst          = 678
	)

	tests := []struct {
		name          string
		testCaseQPS   *float32
		testCaseBurst *int
		workloadQPS   *float32
		workloadBurst *int
		expectedQPS   float32
		expectedBurst int
	}{
		{
			name:          "workload override takes precedence",
			testCaseQPS:   &tcQPS,
			testCaseBurst: &tcBurst,
			workloadQPS:   &wQPS,
			workloadBurst: &wBurst,
			expectedQPS:   wQPS,
			expectedBurst: wBurst,
		},
		{
			name:          "fallback to testCase baseline when workload limits are nil",
			testCaseQPS:   &tcQPS,
			testCaseBurst: &tcBurst,
			workloadQPS:   nil,
			workloadBurst: nil,
			expectedQPS:   tcQPS,
			expectedBurst: tcBurst,
		},
		{
			name:          "global default fallback when both are nil",
			testCaseQPS:   nil,
			testCaseBurst: nil,
			workloadQPS:   nil,
			workloadBurst: nil,
			expectedQPS:   5000.0,
			expectedBurst: 5000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := &testCase{
				Name:                    "TestTC",
				SchedulerAPIClientQPS:   tt.testCaseQPS,
				SchedulerAPIClientBurst: tt.testCaseBurst,
			}
			w := &Workload{
				Name:                    "TestWorkload",
				SchedulerAPIClientQPS:   tt.workloadQPS,
				SchedulerAPIClientBurst: tt.workloadBurst,
			}

			scheduler, _, done, frameworkTCtx := setupTestCase(t, tc, nil, w, &schedulerPerfOptions{})

			frameworkTCtx.Cleanup(func() {
				frameworkTCtx.Cancel("test is done")
				<-done
			})

			// Verify framework QPS and Burst are at their default values (5000)
			frameworkCfg := frameworkTCtx.RESTConfig()
			if frameworkCfg == nil {
				t.Fatal("Expected non-nil framework REST config")
			}
			if frameworkCfg.QPS != 5000.0 {
				t.Errorf("Expected framework QPS to be 5000.0, got %f", frameworkCfg.QPS)
			}
			if frameworkCfg.Burst != 5000 {
				t.Errorf("Expected framework Burst to be 5000, got %d", frameworkCfg.Burst)
			}

			// Verify scheduler QPS and Burst match the expected values
			if len(scheduler.Profiles) == 0 {
				t.Fatal("Expected at least one scheduler profile")
			}
			for _, profile := range scheduler.Profiles {
				schedulerCfg := profile.KubeConfig()
				if schedulerCfg == nil {
					t.Fatal("Expected non-nil scheduler REST config")
				}
				if schedulerCfg.QPS != tt.expectedQPS {
					t.Errorf("Expected scheduler QPS to be %f, got %f", tt.expectedQPS, schedulerCfg.QPS)
				}
				if schedulerCfg.Burst != tt.expectedBurst {
					t.Errorf("Expected scheduler Burst to be %d, got %d", tt.expectedBurst, schedulerCfg.Burst)
				}
			}
		})
	}
}
