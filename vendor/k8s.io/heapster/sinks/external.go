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
	"bytes"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/sinks/cache"
	hUtil "k8s.io/heapster/util"
	kube_api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
)

type lastSync struct {
	podSync    time.Time
	nodeSync   time.Time
	eventsSync time.Time
}

type externalSinkManager struct {
	cache         cache.Cache
	lastSync      lastSync
	syncFrequency time.Duration
	decoder       sink_api.Decoder
	externalSinks []sink_api.ExternalSink
	sync.RWMutex  // Protects externalSinks
}

func supportedMetricsDescriptors() []sink_api.MetricDescriptor {
	// Get supported metrics.
	supportedMetrics := sink_api.SupportedStatMetrics()
	for i := range supportedMetrics {
		supportedMetrics[i].Labels = sink_api.SupportedLabels()
	}

	// Create the metrics.
	descriptors := make([]sink_api.MetricDescriptor, 0, len(supportedMetrics))
	for _, supported := range supportedMetrics {
		descriptors = append(descriptors, supported.MetricDescriptor)
	}
	return descriptors
}

// NewExternalSinkManager returns an external sink manager that will manage pushing data to all
// the sinks in 'externalSinks', which is a map of sink name to ExternalSink object.
func NewExternalSinkManager(externalSinks []sink_api.ExternalSink, cache cache.Cache, syncFrequency time.Duration) (ExternalSinkManager, error) {
	m := &externalSinkManager{
		decoder:       sink_api.NewDecoder(),
		cache:         cache,
		lastSync:      lastSync{},
		syncFrequency: syncFrequency,
	}
	if externalSinks != nil {
		if err := m.SetSinks(externalSinks); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (esm *externalSinkManager) Sync() chan<- struct{} {
	stopChan := make(chan struct{})
	go util.Until(esm.sync, esm.syncFrequency, stopChan)
	return stopChan
}

func (esm *externalSinkManager) sync() {
	if err := esm.store(); err != nil {
		glog.Errorf("failed to sync data to sinks - %v", err)
	}
}

var zeroTime = time.Time{}

// TODO(vmarmol): Paralellize this.
func (esm *externalSinkManager) store() error {
	pods := esm.cache.GetPods(esm.lastSync.podSync, zeroTime)
	for _, pod := range pods {
		esm.lastSync.podSync = hUtil.GetLatest(esm.lastSync.podSync, pod.LastUpdate)
	}
	containers := esm.cache.GetNodes(esm.lastSync.nodeSync, zeroTime)
	containers = append(containers, esm.cache.GetFreeContainers(esm.lastSync.nodeSync, zeroTime)...)
	for _, c := range containers {
		esm.lastSync.nodeSync = hUtil.GetLatest(esm.lastSync.nodeSync, c.LastUpdate)
	}
	// TODO: Store data in cache.
	timeseries, err := esm.decoder.TimeseriesFromPods(pods)
	if err != nil {
		return err
	}
	containerTimeseries, err := esm.decoder.TimeseriesFromContainers(containers)
	if err != nil {
		return err
	}
	timeseries = append(timeseries, containerTimeseries...)

	if len(timeseries) == 0 {
		glog.V(3).Infof("no timeseries data between %v and %v", esm.lastSync.nodeSync, zeroTime)
		// Continue here to push events data.
	}
	events := esm.cache.GetEvents(esm.lastSync.eventsSync.Add(time.Nanosecond), zeroTime)
	var kEvents []kube_api.Event
	for _, event := range events {
		kEvents = append(kEvents, event.Raw)
		esm.lastSync.eventsSync = hUtil.GetLatest(esm.lastSync.eventsSync, event.LastUpdate)
	}
	if len(timeseries) == 0 && len(events) == 0 {
		glog.V(5).Infof("Skipping sync loop")
		return nil
	}

	// Format metrics and push them.
	esm.RLock()
	defer esm.RUnlock()
	errorsLen := 2 * len(esm.externalSinks)
	errorsChan := make(chan error, errorsLen)
	for idx := range esm.externalSinks {
		sink := esm.externalSinks[idx]
		go func(sink sink_api.ExternalSink) {
			glog.V(2).Infof("Storing Timeseries to %q", sink.Name())
			errorsChan <- sink.StoreTimeseries(timeseries)
		}(sink)
		go func(sink sink_api.ExternalSink) {
			glog.V(2).Infof("Storing Events to %q", sink.Name())
			errorsChan <- sink.StoreEvents(kEvents)
		}(sink)
	}
	var errors []string
	for i := 0; i < errorsLen; i++ {
		if err := <-errorsChan; err != nil {
			strError := fmt.Sprintf("%v ", err)
			found := false
			for _, otherError := range errors {
				if otherError == strError {
					found = true
				}
			}
			if !found {
				errors = append(errors, strError)
			}
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("encountered the following errors: %s", strings.Join(errors, ";\n"))
	}
	return nil
}

func (esm *externalSinkManager) DebugInfo() string {
	b := &bytes.Buffer{}
	fmt.Fprintln(b, "External Sinks")
	// Add metrics being exported.
	fmt.Fprintln(b, "\tExported metrics:")
	for _, supported := range sink_api.SupportedStatMetrics() {
		fmt.Fprintf(b, "\t\t%s: %s\n", supported.Name, supported.Description)
	}

	// Add labels being used.
	fmt.Fprintln(b, "\tExported labels:")
	for _, label := range sink_api.SupportedLabels() {
		fmt.Fprintf(b, "\t\t%s: %s\n", label.Key, label.Description)
	}
	fmt.Fprintln(b, "\tExternal Sinks:")
	esm.RLock()
	defer esm.RUnlock()
	for _, externalSink := range esm.externalSinks {
		fmt.Fprintf(b, "\t\t%s\n", externalSink.DebugInfo())
	}

	return b.String()
}

// inSlice returns whether an external sink is part of a set (list) of sinks
func inSlice(sink sink_api.ExternalSink, sinks []sink_api.ExternalSink) bool {
	for _, s := range sinks {
		if sink == s {
			return true
		}
	}
	return false
}

func (esm *externalSinkManager) SetSinks(newSinks []sink_api.ExternalSink) error {
	esm.Lock()
	defer esm.Unlock()
	oldSinks := esm.externalSinks
	descriptors := supportedMetricsDescriptors()
	for _, sink := range oldSinks {
		if inSlice(sink, newSinks) {
			continue
		}
		if err := sink.Unregister(descriptors); err != nil {
			return err
		}
	}
	for _, sink := range newSinks {
		if inSlice(sink, oldSinks) {
			continue
		}
		if err := sink.Register(descriptors); err != nil {
			return err
		}
	}
	esm.externalSinks = newSinks
	glog.V(2).Infof("Updated sinks: %+v", esm.externalSinks)
	return nil
}
