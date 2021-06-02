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

package metrics

import (
	"time"
)

// LatencyMetric is a struct for dashboard metrics.
type LatencyMetric struct {
	Perc50  time.Duration `json:"Perc50"`
	Perc90  time.Duration `json:"Perc90"`
	Perc99  time.Duration `json:"Perc99"`
	Perc100 time.Duration `json:"Perc100"`
}
