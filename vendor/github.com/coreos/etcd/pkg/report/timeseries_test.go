// Copyright 2016 The etcd Authors
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

import (
	"testing"
	"time"
)

func TestGetTimeseries(t *testing.T) {
	sp := newSecondPoints()
	now := time.Now()
	sp.Add(now, time.Second)
	sp.Add(now.Add(5*time.Second), time.Second)
	n := sp.getTimeSeries().Len()
	if n < 3 {
		t.Fatalf("expected at 6 points of time series, got %s", sp.getTimeSeries())
	}
}
