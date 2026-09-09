// Copyright 2020 The Prometheus Authors
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

package promlint

import dto "github.com/prometheus/client_model/go"

// A Problem is an issue detected by a linter.
type Problem struct {
	// The name of the metric indicated by this Problem.
	Metric string

	// A description of the issue for this Problem.
	Text string
}

// newProblem is helper function to create a Problem.
func newProblem(mf *dto.MetricFamily, text string) Problem {
	return Problem{
		Metric: mf.GetName(),
		Text:   text,
	}
}
