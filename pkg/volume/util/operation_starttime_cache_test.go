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

package util

import (
	"testing"
	"time"
)

func TestNewOperationStartTimeCache(t *testing.T) {
	c := NewOperationStartTimeCache()
	plugin, operation, startTime, ok := c.Load("NotExistedKey")
	if ok {
		t.Errorf("c.Load('NotExistedKey') = %s, %s, %s, %t; wanted FALSE, got TRUE",
			plugin, operation, startTime.String(), ok)
	}
}

func TestOperationStartTimeCacheLoad(t *testing.T) {
	cKey := "unique-identifier"
	op := "some-volume-operation"
	p := "kubernetes.io/volume-plugin-name"
	c := NewOperationStartTimeCache()
	plugin, operation, startTime, ok := c.Load(cKey)
	if plugin != "" || operation != "" || !time.Time.IsZero(startTime) || ok {
		t.Errorf("c.Load('%s') = %s, %s, %s, %t; expected: \"\", \"\", %s, false",
			cKey, plugin, operation, startTime.String(), ok, time.Time{}.String())
	}
	// add an entry and load it out
	c.AddIfNotExist(cKey, p, op)
	plugin, operation, startTime, ok = c.Load(cKey)
	if !ok || p != plugin || op != operation || time.Time.IsZero(startTime) {
		t.Errorf("c.Load('%s') = %s, %s, %s, %t; expected: %s, %s, %s, true",
			cKey, plugin, operation, startTime.String(), ok, p, op, time.Time{}.String())
	}
}

func TestOperationStartTimeCacheAddIfNotExist(t *testing.T) {
	cKey := "unique-identifier"
	op := "some-volume-operation"
	p := "kubernetes.io/volume-plugin-name"
	c := NewOperationStartTimeCache()
	_, _, _, ok := c.Load(cKey)
	if ok {
		t.Errorf("c.Load('%s') = _, _, _, %t; got TRUE, want FALSE", cKey, ok)
	}
	// add an entry
	c.AddIfNotExist(cKey, p, op)
	plugin, operation, startTime, ok := c.Load(cKey)
	if !ok {
		t.Errorf("c.Load('%s') = _, _, _ , %t; got FALSE, wanted TRUE", cKey, ok)
	}
	// add one more time with the same key should NOT update the cache
	c.AddIfNotExist(cKey, "another-plugin-name", "another-operation-name")
	pluginNew, operationNew, startTimeNew, ok := c.Load(cKey)

	if !ok || pluginNew != plugin || operationNew != operation || startTimeNew != startTime {
		t.Errorf("c.Load('%s') = %s, %s, %s, %t; expected: %s, %s, %s, true",
			cKey, pluginNew, operationNew, startTimeNew.String(), ok,
			plugin, operation, startTime.String())
	}
}

func TestOperationStartTimeCacheHas(t *testing.T) {
	cKey := "unique-identifier"
	op := "some-volume-operation"
	p := "kubernetes.io/volume-plugin-name"
	c := NewOperationStartTimeCache()
	existed := c.Has(cKey)
	if existed {
		t.Errorf("c.Has('%s') = %t; got TRUE, want FALSE", cKey, existed)
	}
	// add an entry
	c.AddIfNotExist(cKey, p, op)
	existed = c.Has(cKey)
	if !existed {
		t.Errorf("c.Has('%s') = %t; got FALSE, wanted TRUE", cKey, existed)
	}
}

func TestOperationStartTimeCacheDelete(t *testing.T) {
	cKey := "unique-identifier"
	op := "some-volume-operation"
	p := "kubernetes.io/volume-plugin-name"
	c := NewOperationStartTimeCache()
	existed := c.Has(cKey)
	if existed {
		t.Errorf("c.Has('%s') = %t; got TRUE, want FALSE", cKey, existed)
	}
	// delete an non-existed entry should fail silently
	c.Delete(cKey)
	// add an entry
	c.AddIfNotExist(cKey, p, op)
	existed = c.Has(cKey)
	if !existed {
		t.Errorf("c.Has('%s') = %t; got FALSE, wanted TRUE", cKey, existed)
	}
	// delete the entry
	c.Delete(cKey)
	existed = c.Has(cKey)
	if existed {
		t.Errorf("c.Has('%s') = %t; got TRUE, want FALSE", cKey, existed)
	}
}

func TestOperationStartTimeCacheUpdatePluginName(t *testing.T) {
	cKey := "unique-identifier"
	op := "some-volume-operation"
	p := "kubernetes.io/volume-plugin-name"
	np := "kubernetes.io/new-volume-plugin-name"
	c := NewOperationStartTimeCache()
	// add an entry
	c.AddIfNotExist(cKey, p, op)
	// load entry to check original values
	oldPlugin, operation, startTime, ok := c.Load(cKey)
	if !ok || oldPlugin != p || operation != op {
		t.Errorf("c.Load('%s') = %s, %s, %s, %t; expected: %s, %s, %s, true",
			cKey, oldPlugin, operation, startTime.String(), ok,
			p, op, startTime.String())
	}
	// update
	ok = c.UpdatePluginName(cKey, np)
	if !ok {
		t.Errorf("c.UpdatePluginName(%s, %s) = %v; expected: true", cKey, np, ok)
	}
	newPlugin, operation, newStartTime, ok := c.Load(cKey)
	if !ok || newPlugin != np || operation != op || newStartTime != startTime {
		t.Errorf("c.Load('%s') = %s, %s, %s, %t; expected: %s, %s, %s, true",
			cKey, newPlugin, operation, startTime.String(), ok,
			np, op, startTime.String())
	}
}
