// Copyright 2015 Google Inc. All Rights Reserved.
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

package collector

import (
	"time"

	v1 "github.com/google/cadvisor/info/v1"
)

type FakeCollectorManager struct {
}

func (fkm *FakeCollectorManager) RegisterCollector(collector Collector) error {
	return nil
}

func (fkm *FakeCollectorManager) GetSpec() ([]v1.MetricSpec, error) {
	return []v1.MetricSpec{}, nil
}

func (fkm *FakeCollectorManager) Collect(metric map[string][]v1.MetricVal) (time.Time, map[string][]v1.MetricVal, error) {
	var zero time.Time
	return zero, metric, nil
}
