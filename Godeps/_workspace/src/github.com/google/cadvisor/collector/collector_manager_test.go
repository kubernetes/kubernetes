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
	"testing"
	"time"

	"github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

type fakeCollector struct {
	nextCollectionTime time.Time
	err                error
	collectedFrom      int
}

func (fc *fakeCollector) Collect(metric map[string]v1.MetricVal) (time.Time, map[string]v1.MetricVal, error) {
	fc.collectedFrom++
	return fc.nextCollectionTime, metric, fc.err
}

func (fc *fakeCollector) Name() string {
	return "fake-collector"
}

func (fc *fakeCollector) GetSpec() []v1.MetricSpec {
	return []v1.MetricSpec{}
}

func TestCollect(t *testing.T) {
	cm := &GenericCollectorManager{}

	firstTime := time.Now().Add(-time.Hour)
	secondTime := time.Now().Add(time.Hour)
	f1 := &fakeCollector{
		nextCollectionTime: firstTime,
	}
	f2 := &fakeCollector{
		nextCollectionTime: secondTime,
	}

	assert := assert.New(t)
	assert.NoError(cm.RegisterCollector(f1))
	assert.NoError(cm.RegisterCollector(f2))

	// First collection, everyone gets collected from.
	nextTime, _, err := cm.Collect()
	assert.Equal(firstTime, nextTime)
	assert.NoError(err)
	assert.Equal(1, f1.collectedFrom)
	assert.Equal(1, f2.collectedFrom)

	f1.nextCollectionTime = time.Now().Add(2 * time.Hour)

	// Second collection, only the one that is ready gets collected from.
	nextTime, _, err = cm.Collect()
	assert.Equal(secondTime, nextTime)
	assert.NoError(err)
	assert.Equal(2, f1.collectedFrom)
	assert.Equal(1, f2.collectedFrom)
}
