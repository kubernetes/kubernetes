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

package base

import (
	"time"

	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/info/v2"
)

type Collector struct {
}

var _ collector.Collector = &Collector{}

func (c *Collector) Collect() (time.Time, []v2.Metric, error) {
	return time.Now(), []v2.Metrics{}, nil
}

func (c *Collector) Name() string {
	return "default"
}
