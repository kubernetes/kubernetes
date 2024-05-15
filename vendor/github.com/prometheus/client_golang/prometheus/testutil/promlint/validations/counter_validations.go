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

package validations

import (
	"errors"
	"strings"

	dto "github.com/prometheus/client_model/go"
)

// LintCounter detects issues specific to counters, as well as patterns that should
// only be used with counters.
func LintCounter(mf *dto.MetricFamily) []error {
	var problems []error

	isCounter := mf.GetType() == dto.MetricType_COUNTER
	isUntyped := mf.GetType() == dto.MetricType_UNTYPED
	hasTotalSuffix := strings.HasSuffix(mf.GetName(), "_total")

	switch {
	case isCounter && !hasTotalSuffix:
		problems = append(problems, errors.New(`counter metrics should have "_total" suffix`))
	case !isUntyped && !isCounter && hasTotalSuffix:
		problems = append(problems, errors.New(`non-counter metrics should not have "_total" suffix`))
	}

	return problems
}
