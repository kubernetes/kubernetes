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

package manager

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/heapster/model"
	"k8s.io/heapster/sinks"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/sinks/cache"
	source_api "k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
)

// Manager provides an interface to control the core of heapster.
// Implementations are not required to be thread safe.
type Manager interface {
	// Export the latest data point of all metrics.
	ExportMetrics() ([]*sink_api.Point, error)

	// Set the sinks to use
	SetSinkUris(Uris) error

	// Get the sinks currently in use
	SinkUris() Uris

	// Get a reference to the cluster entity of the model, if it exists.
	GetModel() model.Model

	// Starts the manager.
	Start()

	// Stops the manager.
	Stop()
}

type realManager struct {
	sources       []source_api.Source
	cache         cache.Cache
	model         model.Model
	modelDuration time.Duration
	sinkManager   sinks.ExternalSinkManager
	sinkUris      Uris
	lastSync      time.Time
	resolution    time.Duration
	decoder       sink_api.Decoder
	mainStopChan  chan struct{}
	sinkStopChan  chan<- struct{}
	modelStopChan chan struct{}
}

type syncData struct {
	data  source_api.AggregateData
	mutex sync.Mutex
}

func NewManager(sources []source_api.Source, sinkManager sinks.ExternalSinkManager, res, bufferDuration time.Duration,
	c cache.Cache, useModel bool, modelRes, modelDuration time.Duration) (Manager, error) {

	var newModel model.Model
	if useModel {
		newModel = model.NewModel(modelRes)
		// Temporary semi-hack to get model storage garbage-collected.
		c.AddCacheListener(newModel.GetCacheListener())
	}

	return &realManager{
		sources:       sources,
		sinkManager:   sinkManager,
		cache:         c,
		model:         newModel,
		modelDuration: modelDuration,
		lastSync:      time.Now().Round(res),
		resolution:    res,
		decoder:       sink_api.NewDecoder(),
		mainStopChan:  make(chan struct{}),
		modelStopChan: make(chan struct{}),
		sinkStopChan:  sinkManager.Sync(),
	}, nil
}

func (rm *realManager) GetModel() model.Model {
	return rm.model
}

func (rm *realManager) scrapeSource(s source_api.Source, start, end time.Time, sd *syncData, errChan chan<- error) {
	glog.V(2).Infof("attempting to get data from source %q", s.Name())
	data, err := s.GetInfo(start, end)
	if err != nil {
		errChan <- fmt.Errorf("failed to get information from source %q - %v", s.Name(), err)
		return
	}
	sd.mutex.Lock()
	defer sd.mutex.Unlock()
	sd.data.Merge(&data)
	errChan <- nil
}

func (rm *realManager) Start() {
	go rm.Housekeep()
	if rm.model != nil {
		go rm.HousekeepModel()
	}
}

func (rm *realManager) Stop() {
	rm.mainStopChan <- struct{}{}
	if rm.model != nil {
		rm.modelStopChan <- struct{}{}
	}
	rm.sinkStopChan <- struct{}{}
}

// HousekeepModel periodically populates the manager model from the manager cache.
func (rm *realManager) HousekeepModel() {
	for {
		select {
		//TODO: better timing here
		case <-time.After(rm.modelDuration):
			if err := rm.model.Update(rm.cache); err != nil {
				glog.V(1).Infof("Model housekeeping returned error: %s", err.Error())
			}
		case <-rm.modelStopChan:
			return
		}
	}
}

func (rm *realManager) Housekeep() {
	for {
		start := rm.lastSync
		end := start.Add(rm.resolution)
		timeToNextSync := end.Sub(time.Now())
		// TODO: consider adding some delay here
		select {
		case <-time.After(timeToNextSync):
			rm.housekeep(start, end)
			rm.lastSync = end
		case <-rm.mainStopChan:
			return
		}
	}
}

func (rm *realManager) housekeep(start, end time.Time) {
	glog.V(2).Infof("starting to scrape data from sources start: %v end: %v", start, end)
	errChan := make(chan error, len(rm.sources))
	var sd syncData
	for idx := range rm.sources {
		s := rm.sources[idx]
		go rm.scrapeSource(s, start, end, &sd, errChan)
	}
	var errors []string
	for i := 0; i < len(rm.sources); i++ {
		if err := <-errChan; err != nil {
			errors = append(errors, err.Error())
		}
	}

	rm.mapInfraContainersToPods(&sd)

	glog.V(2).Infof("completed scraping data from sources. Errors: %v", errors)
	if err := rm.cache.StorePods(sd.data.Pods); err != nil {
		errors = append(errors, err.Error())
	}
	if err := rm.cache.StoreContainers(sd.data.Machine); err != nil {
		errors = append(errors, err.Error())
	}
	if err := rm.cache.StoreContainers(sd.data.Containers); err != nil {
		errors = append(errors, err.Error())
	}
	if len(errors) > 0 {
		glog.V(1).Infof("housekeeping resulted in following errors: %v", errors)
	}
}

// Map infra containers to pods - only if we have any pods of course.
// Clunky but required without significant API changes (coming soon!).
func (rm *realManager) mapInfraContainersToPods(sd *syncData) {
	if len(sd.data.Pods) > 0 {
		sd.mutex.Lock()
		defer sd.mutex.Unlock()

		// Iterate slice in reverse as we are going to be deleting containers if we manage to match infra containers.
		for containerIndex := len(sd.data.Containers) - 1; containerIndex >= 0; containerIndex-- {
			cont := sd.data.Containers[containerIndex]
			podName, podNameFound := cont.Spec.Labels[datasource.KubernetesPodNameLabel]
			podNamespace, podNamespaceFound := cont.Spec.Labels[datasource.KubernetesPodNamespaceLabel]

			// Not a pod container then continue.
			if !podNameFound || !podNamespaceFound {
				continue
			}

			// Find a pod with these details.
			for podIndex := range sd.data.Pods {
				pod := &sd.data.Pods[podIndex]
				// If we find a matching pod then add the container to the pod's containers slice.
				if pod.Name == podName && pod.Namespace == podNamespace {
					cont.Hostname = pod.Hostname
					cont.ExternalID = pod.ExternalID
					cont.Name = datasource.KubernetesPodInfraContainerName
					pod.Containers = append(pod.Containers, cont)

					// Delete the matched container so we don't duplicate containers.
					sd.data.Containers = append(sd.data.Containers[:containerIndex], sd.data.Containers[containerIndex+1:]...)

					break
				}
			}
		}
	}
}

func (rm *realManager) ExportMetrics() ([]*sink_api.Point, error) {
	var zero time.Time

	// Get all pods as points.
	pods := trimStatsForPods(rm.cache.GetPods(zero, zero))
	timeseries, err := rm.decoder.TimeseriesFromPods(pods)
	if err != nil {
		return []*sink_api.Point{}, err
	}
	points := make([]*sink_api.Point, 0, len(timeseries))
	points = appendPoints(points, timeseries)

	// Get all nodes as points.
	containers := trimStatsForContainers(rm.cache.GetNodes(zero, zero))
	timeseries, err = rm.decoder.TimeseriesFromContainers(containers)
	if err != nil {
		return []*sink_api.Point{}, err
	}
	points = appendPoints(points, timeseries)

	// Get all free containers as points.
	containers = trimStatsForContainers(rm.cache.GetFreeContainers(zero, zero))
	timeseries, err = rm.decoder.TimeseriesFromContainers(containers)
	if err != nil {
		return []*sink_api.Point{}, err
	}
	points = appendPoints(points, timeseries)

	return points, nil
}

// Extract the points from the specified timeseries and append them to output.
func appendPoints(output []*sink_api.Point, toExtract []sink_api.Timeseries) []*sink_api.Point {
	for i := range toExtract {
		output = append(output, toExtract[i].Point)
	}
	return output
}

// Only keep latest stats for the specified pods
func trimStatsForPods(pods []*cache.PodElement) []*cache.PodElement {
	for _, pod := range pods {
		trimStatsForContainers(pod.Containers)
	}
	return pods
}

// Only keep latest stats for the specified containers
func trimStatsForContainers(containers []*cache.ContainerElement) []*cache.ContainerElement {
	for _, cont := range containers {
		onlyKeepLatestStat(cont)
	}
	return containers
}

// Only keep the latest stats data point.
func onlyKeepLatestStat(cont *cache.ContainerElement) {
	if len(cont.Metrics) > 1 {
		cont.Metrics = cont.Metrics[0:1]
	}
}

func (rm *realManager) SetSinkUris(sinkUris Uris) error {
	sinks, err := newSinks(sinkUris, rm.resolution)
	if err != nil {
		// Skip sink setup errors here. We do not want to fail if one or more
		// sinks fail.
		glog.Error(err)
	}
	if err := rm.sinkManager.SetSinks(sinks); err != nil {
		return err
	}
	rm.sinkUris = sinkUris
	return nil
}

func (rm *realManager) SinkUris() Uris {
	return rm.sinkUris
}
