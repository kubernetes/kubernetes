// Copyright 2017 The etcd Authors
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

package report

import "testing"

func TestPercentiles(t *testing.T) {
	nums := make([]float64, 100)
	nums[99] = 1 // 99-percentile (1 out of 100)
	data := percentiles(nums)
	if data[len(pctls)-2] != 1 {
		t.Fatalf("99-percentile expected 1, got %f", data[len(pctls)-2])
	}

	nums = make([]float64, 1000)
	nums[999] = 1 // 99.9-percentile (1 out of 1000)
	data = percentiles(nums)
	if data[len(pctls)-1] != 1 {
		t.Fatalf("99.9-percentile expected 1, got %f", data[len(pctls)-1])
	}
}
