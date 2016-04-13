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

package sinks

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	fuzz "github.com/google/gofuzz"
	_ "k8s.io/heapster/api/v1/types"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/sinks/cache"
	source_api "k8s.io/heapster/sources/api"
	hUtil "k8s.io/heapster/util"
	kube_api "k8s.io/kubernetes/pkg/api"

	"github.com/golang/glog"
)

type DummySink struct {
	Registered       int
	Unregistered     int
	StoredTimeseries int
	StoredEvents     int
}

func (d *DummySink) Register([]sink_api.MetricDescriptor) error {
	d.Registered++
	return nil
}

func (d *DummySink) Unregister([]sink_api.MetricDescriptor) error {
	d.Unregistered++
	return nil
}

func (d *DummySink) StoreTimeseries([]sink_api.Timeseries) error {
	d.StoredTimeseries++
	return nil
}

func (d *DummySink) StoreEvents([]kube_api.Event) error {
	d.StoredEvents++
	return nil
}

func (d *DummySink) DebugInfo() string {
	return "dummy sink"
}

func (d *DummySink) Name() string {
	return "dummy"
}

func TestSetSinksRegister(t *testing.T) {
	as := assert.New(t)
	s1 := &DummySink{}
	as.Equal(0, s1.Registered)
	m, err := NewExternalSinkManager([]sink_api.ExternalSink{s1}, nil, time.Minute)
	as.Nil(err)
	as.Equal(1, s1.Registered)
	err = m.SetSinks([]sink_api.ExternalSink{s1})
	as.Nil(err)
	s2 := &DummySink{}
	as.Equal(1, s1.Registered)
	as.Equal(0, s2.Registered)
	err = m.SetSinks([]sink_api.ExternalSink{s2})
	as.Nil(err)
	as.Equal(1, s1.Registered)
	as.Equal(1, s2.Registered)
	err = m.SetSinks([]sink_api.ExternalSink{})
	as.Nil(err)
	as.Equal(1, s1.Registered)
	as.Equal(1, s2.Registered)
}

func TestSetSinksUnregister(t *testing.T) {
	as := assert.New(t)
	s1 := &DummySink{}
	as.Equal(0, s1.Unregistered)
	m, err := NewExternalSinkManager([]sink_api.ExternalSink{s1}, nil, time.Minute)
	as.Nil(err)
	as.Equal(0, s1.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{s1})
	as.Nil(err)
	s2 := &DummySink{}
	as.Equal(0, s1.Unregistered)
	as.Equal(0, s2.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{s2})
	as.Nil(err)
	as.Equal(1, s1.Unregistered)
	as.Equal(0, s2.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{})
	as.Nil(err)
	as.Equal(1, s1.Unregistered)
	as.Equal(1, s2.Unregistered)
}

func TestSetSinksRegisterAgain(t *testing.T) {
	as := assert.New(t)
	s1 := &DummySink{}
	as.Equal(0, s1.Registered)
	as.Equal(0, s1.Unregistered)
	m, err := NewExternalSinkManager([]sink_api.ExternalSink{s1}, nil, time.Minute)
	as.Nil(err)
	as.Equal(1, s1.Registered)
	as.Equal(0, s1.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{})
	as.Nil(err)
	as.Equal(1, s1.Registered)
	as.Equal(1, s1.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{s1})
	as.Nil(err)
	as.Equal(2, s1.Registered)
	as.Equal(1, s1.Unregistered)
	err = m.SetSinks([]sink_api.ExternalSink{})
	as.Nil(err)
	as.Equal(2, s1.Registered)
	as.Equal(2, s1.Unregistered)
}

func newExternalSinkManager(externalSinks []sink_api.ExternalSink, cache cache.Cache, syncFrequency time.Duration) (*externalSinkManager, error) {
	m := &externalSinkManager{
		decoder:       sink_api.NewDecoder(),
		cache:         cache,
		lastSync:      lastSync{zeroTime, zeroTime, zeroTime},
		syncFrequency: syncFrequency,
	}
	if externalSinks != nil {
		if err := m.SetSinks(externalSinks); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func TestSyncLastUpdated(t *testing.T) {
	as := assert.New(t)
	s1 := &DummySink{}
	c := cache.NewCache(time.Hour, time.Minute)
	m, err := newExternalSinkManager([]sink_api.ExternalSink{s1}, c, time.Microsecond)
	as.Nil(err)
	var (
		pods                                        []source_api.Pod
		containers                                  []source_api.Container
		events                                      []*cache.Event
		expectedESync, expectedPSync, expectedNSync time.Time
	)
	f := fuzz.New().NumElements(10, 10).NilChance(0)
	f.Fuzz(&pods)
	now := time.Now()
	for pidx := range pods {
		for cidx := range pods[pidx].Containers {
			for sidx := range pods[pidx].Containers[cidx].Stats {
				ts := now.Add(time.Duration(sidx) * time.Minute)
				pods[pidx].Containers[cidx].Stats[sidx].Timestamp = ts
				expectedPSync = hUtil.GetLatest(expectedPSync, ts)
			}
		}
	}
	f.Fuzz(&containers)
	for cidx := range containers {
		for sidx := range containers[cidx].Stats {
			ts := now.Add(time.Duration(sidx) * time.Minute)
			containers[cidx].Stats[sidx].Timestamp = ts
			expectedNSync = hUtil.GetLatest(expectedNSync, ts)
		}
	}
	f.Fuzz(&events)
	for eidx := range events {
		ts := now.Add(time.Duration(eidx) * time.Minute)
		events[eidx].LastUpdate = ts
		events[eidx].UID = fmt.Sprintf("id:%d", eidx)
		expectedESync = hUtil.GetLatest(expectedESync, ts)
	}
	if err := c.StorePods(pods); err != nil {
		glog.Fatalf("Failed to store pods: %v", err)
	}
	if err := c.StoreContainers(containers); err != nil {
		glog.Fatalf("Failed to store containers: %v", err)
	}
	if err = c.StoreEvents(events); err != nil {
		glog.Fatalf("Failed to store events: %v", err)
	}
	m.store()
	as.Equal(m.lastSync.eventsSync, expectedESync, "Event now: %v, eventSync: %v, expected: %v", now, m.lastSync.eventsSync, expectedESync)
	as.Equal(m.lastSync.podSync, expectedPSync, "Pod now: %v, podSync: %v, expected: %v", now, m.lastSync.podSync, expectedPSync)
	as.Equal(m.lastSync.nodeSync, expectedNSync, "Node now: %v, nodeSync: %v, expected: %v", now, m.lastSync.nodeSync, expectedNSync)
}

func TestSetSinksStore(t *testing.T) {
	as := assert.New(t)
	s1 := &DummySink{}
	c := cache.NewCache(time.Hour, time.Minute)
	m, err := newExternalSinkManager([]sink_api.ExternalSink{s1}, c, time.Microsecond)
	as.Nil(err)
	as.Equal(0, s1.StoredTimeseries)
	as.Equal(0, s1.StoredEvents)
	var (
		pods       []source_api.Pod
		containers []source_api.Container
		events     []*cache.Event
	)
	f := fuzz.New().NumElements(1, 1).NilChance(0)

	time1 := time.Now()
	f.Fuzz(&pods)
	for pidx := range pods {
		for cidx := range pods[pidx].Containers {
			for sidx := range pods[pidx].Containers[cidx].Stats {
				pods[pidx].Containers[cidx].Stats[sidx].Timestamp = time1
			}
		}
	}
	f.Fuzz(&containers)
	for cidx := range containers {
		for sidx := range containers[cidx].Stats {
			containers[cidx].Stats[sidx].Timestamp = time1
		}
	}
	f.Fuzz(&events)
	for eidx := range events {
		events[eidx].LastUpdate = time1
		events[eidx].UID = fmt.Sprintf("id1:%d", eidx)
	}

	if err := c.StorePods(pods); err != nil {
		glog.Fatalf("Failed to store pods: %v", err)
	}
	if err := c.StoreContainers(containers); err != nil {
		glog.Fatalf("Failed to store containers: %v", err)
	}
	if err = c.StoreEvents(events); err != nil {
		glog.Fatalf("Failed to store events: %v", err)
	}
	m.sync()
	as.Equal(1, s1.StoredTimeseries)
	as.Equal(1, s1.StoredEvents)
	err = m.SetSinks([]sink_api.ExternalSink{})
	as.Nil(err)
	m.sync()
	as.Equal(1, s1.StoredTimeseries)
	as.Equal(1, s1.StoredEvents)
	err = m.SetSinks([]sink_api.ExternalSink{s1})
	as.Equal(1, s1.StoredTimeseries)
	as.Equal(1, s1.StoredEvents)
	as.Nil(err)

	time2 := time.Now()
	f.Fuzz(&pods)
	for pidx := range pods {
		for cidx := range pods[pidx].Containers {
			for sidx := range pods[pidx].Containers[cidx].Stats {
				pods[pidx].Containers[cidx].Stats[sidx].Timestamp = time2
			}
		}
	}
	f.Fuzz(&containers)
	for cidx := range containers {
		for sidx := range containers[cidx].Stats {
			containers[cidx].Stats[sidx].Timestamp = time2
		}
	}
	f.Fuzz(&events)
	for eidx := range events {
		events[eidx].LastUpdate = time1
		events[eidx].UID = fmt.Sprintf("id2:%d", eidx)
	}

	if err := c.StorePods(pods); err != nil {
		glog.Fatalf("Failed to store pods: %v", err)
	}
	if err := c.StoreContainers(containers); err != nil {
		glog.Fatalf("Failed to store containers: %v", err)
	}
	if err = c.StoreEvents(events); err != nil {
		glog.Fatalf("Failed to store events: %v", err)
	}
	m.sync()
	as.Equal(2, s1.StoredTimeseries)
	as.Equal(2, s1.StoredEvents)
}
