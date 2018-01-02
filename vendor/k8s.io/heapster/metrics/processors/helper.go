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

package processors

import (
	"fmt"

	"k8s.io/heapster/metrics/core"
)

func aggregate(src, dst *core.MetricSet, metricsToAggregate []string) error {
	for _, metricName := range metricsToAggregate {
		metricValue, found := src.MetricValues[metricName]
		if !found {
			continue
		}
		aggregatedValue, found := dst.MetricValues[metricName]
		if found {
			if aggregatedValue.ValueType != metricValue.ValueType {
				return fmt.Errorf("NamespaceAggregator: type not supported in %s", metricName)
			}

			if aggregatedValue.ValueType == core.ValueInt64 {
				aggregatedValue.IntValue += metricValue.IntValue
			} else if aggregatedValue.ValueType == core.ValueFloat {
				aggregatedValue.FloatValue += metricValue.FloatValue
			} else {
				return fmt.Errorf("NamespaceAggregator: type not supported in %s", metricName)
			}
		} else {
			aggregatedValue = metricValue
		}
		dst.MetricValues[metricName] = aggregatedValue
	}
	return nil
}
