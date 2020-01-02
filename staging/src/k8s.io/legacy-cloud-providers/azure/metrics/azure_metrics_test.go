// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAzureMetricLabelCardinality(t *testing.T) {
	mc := NewMetricContext("test", "create", "resource_group", "subscription_id", "source")
	assert.Len(t, mc.attributes, len(metricLabels), "cardinalities of labels and values must match")
}

func TestAzureMetricLabelPrefix(t *testing.T) {
	mc := NewMetricContext("prefix", "request", "resource_group", "subscription_id", "source")
	found := false
	for _, attribute := range mc.attributes {
		if attribute == "prefix_request" {
			found = true
		}
	}
	assert.True(t, found, "request label must be prefixed")
}
