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

package model

import (
	"fmt"
	"time"

	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

// aggregationStep performs a Metric Aggregation step on the cluster.
// The Metrics fields of all Namespaces, Pods and the Cluster are populated,
// by Timeseries summation of the respective Metrics fields.
// aggregationStep should be called after new data is present in the cluster,
// but before the cluster timestamp is updated.

// The latestTime argument represents the latest time of metrics found in the model,
// which should cause all aggregated metrics to remain constant up until that time.
func (rc *realModel) aggregationStep(latestTime time.Time) error {
	rc.lock.Lock()
	defer rc.lock.Unlock()

	// Perform Node Metric Aggregation
	node_c := make(chan error)
	go rc.aggregateNodeMetrics(node_c, latestTime)

	// Initiate bottom-up aggregation for Kubernetes stats
	kube_c := make(chan error)
	go rc.aggregateKubeMetrics(kube_c, latestTime)

	errs := make([]error, 2)
	errs[0] = <-node_c
	errs[1] = <-kube_c

	if errs[0] != nil {
		return errs[0]
	}
	if errs[1] != nil {
		return errs[1]
	}

	return nil
}

// aggregateNodeMetrics populates the Cluster.InfoType.Metrics field by adding up all node metrics.
// Assumes an appropriate lock is already taken by the caller.
func (rc *realModel) aggregateNodeMetrics(c chan error, latestTime time.Time) {
	if len(rc.Nodes) == 0 {
		// Fail silently if the cluster has no nodes
		c <- nil
		return
	}

	sources := []*InfoType{}
	for _, node := range rc.Nodes {
		sources = append(sources, &(node.InfoType))
	}
	c <- rc.aggregateMetrics(&rc.ClusterInfo.InfoType, sources, latestTime)
}

// aggregateKubeMetrics initiates depth-first aggregation of Kubernetes metrics.
// Assumes an appropriate lock is already taken by the caller.
func (rc *realModel) aggregateKubeMetrics(c chan error, latestTime time.Time) {
	if len(rc.Namespaces) == 0 {
		// Fail silently if the cluster has no namespaces
		c <- nil
		return
	}

	// Perform aggregation for all the namespaces
	chans := make([]chan error, 0, len(rc.Namespaces))
	for _, namespace := range rc.Namespaces {
		chans = append(chans, make(chan error))
		go rc.aggregateNamespaceMetrics(namespace, chans[len(chans)-1], latestTime)
	}

	errs := make([]error, 0, len(chans))
	for _, channel := range chans {
		errs = append(errs, <-channel)
	}

	for _, err := range errs {
		if err != nil {
			c <- err
			return
		}
	}

	c <- nil
}

// aggregateNamespaceMetrics populates a NamespaceInfo.Metrics field by aggregating all PodInfo.
// Assumes an appropriate lock is already taken by the caller.
func (rc *realModel) aggregateNamespaceMetrics(namespace *NamespaceInfo, c chan error, latestTime time.Time) {
	if namespace == nil {
		c <- fmt.Errorf("nil Namespace pointer passed for aggregation")
		return
	}
	if len(namespace.Pods) == 0 {
		// Fail silently if the namespace has no pods
		c <- nil
		return
	}

	// Perform aggregation on all the Pods
	chans := make([]chan error, 0, len(namespace.Pods))
	for _, pod := range namespace.Pods {
		chans = append(chans, make(chan error))
		go rc.aggregatePodMetrics(pod, chans[len(chans)-1], latestTime)
	}

	errs := make([]error, 0, len(chans))
	for _, channel := range chans {
		errs = append(errs, <-channel)
	}

	for _, err := range errs {
		if err != nil {
			c <- err
			return
		}
	}

	// Collect the Pod InfoTypes after aggregation is complete
	sources := []*InfoType{}
	for _, pod := range namespace.Pods {
		sources = append(sources, &(pod.InfoType))
	}
	c <- rc.aggregateMetrics(&namespace.InfoType, sources, latestTime)
}

// aggregatePodMetrics populates a PodInfo.Metrics field by aggregating all ContainerInfo.
// Assumes an appropriate lock is already taken by the caller.
func (rc *realModel) aggregatePodMetrics(pod *PodInfo, c chan error, latestTime time.Time) {
	if pod == nil {
		c <- fmt.Errorf("nil Pod pointer passed for aggregation")
		return
	}
	if len(pod.Containers) == 0 {
		// Fail silently if the pod has no containers
		c <- nil
		return
	}

	// Collect the Container InfoTypes
	sources := []*InfoType{}
	for _, container := range pod.Containers {
		sources = append(sources, &(container.InfoType))
	}
	c <- rc.aggregateMetrics(&pod.InfoType, sources, latestTime)
}

// aggregateMetrics populates an InfoType by adding metrics across a slice of InfoTypes.
// Only metrics taken after the cluster timestamp are affected.
// Assumes an appropriate lock is already taken by the caller.
func (rc *realModel) aggregateMetrics(target *InfoType, sources []*InfoType, latestTime time.Time) error {
	zeroTime := time.Time{}

	if target == nil {
		return fmt.Errorf("nil InfoType pointer provided as aggregation target")
	}
	if len(sources) == 0 {
		return fmt.Errorf("empty sources slice provided")
	}
	for _, source := range sources {
		if source == nil {
			return fmt.Errorf("nil InfoType pointer provided as an aggregation source")
		}
		if source == target {
			return fmt.Errorf("target InfoType pointer is provided as a source")
		}
	}

	if latestTime.Equal(zeroTime) {
		return fmt.Errorf("aggregateMetrics called with a zero latestTime argument")
	}

	// Create a map of []TimePoint as a timeseries accumulator per metric
	newMetrics := make(map[string][]statstore.TimePoint)

	// Reduce the sources slice with timeseries addition for each metric
	for _, info := range sources {
		for key, ds := range info.Metrics {
			_, ok := newMetrics[key]
			if !ok {
				// Metric does not exist on target map, create a new timeseries
				newMetrics[key] = []statstore.TimePoint{}
			}
			// Perform timeseries addition between the accumulator and the current source
			sourceDS := (*ds).Hour.Get(rc.timestamp, zeroTime)
			newMetrics[key] = addMatchingTimeseries(newMetrics[key], sourceDS)
		}
	}

	// Put all the new values in the DayStores under target
	for key, tpSlice := range newMetrics {
		if len(tpSlice) == 0 {
			continue
		}
		_, ok := target.Metrics[key]
		if !ok {
			// Metric does not exist on target InfoType, create DayStore
			target.Metrics[key] = daystore.NewDayStore(epsilonFromMetric(key), rc.resolution)
		}

		// Put the added TimeSeries in the corresponding DayStore, in time-ascending order
		for i := len(tpSlice) - 1; i >= 0; i-- {
			err := target.Metrics[key].Put(tpSlice[i])
			if err != nil {
				return fmt.Errorf("error while performing aggregation: %s", err)
			}
		}

		// Put a TimePoint with the latest aggregated value at the latest model resolution.
		// Causes the DayStore to assume the aggregated metric remained constant until the -
		// next cluster timestamp.
		newTP := statstore.TimePoint{
			Timestamp: latestTime,
			Value:     tpSlice[0].Value,
		}
		err := target.Metrics[key].Put(newTP)
		if err != nil {
			return fmt.Errorf("error while performing aggregation: %s", err)
		}

	}

	// Set the creation time of the entity to the earliest one that we have found data for.
	earliestCreation := sources[0].Creation
	for _, info := range sources[1:] {
		if info.Creation.Before(earliestCreation) && info.Creation.After(time.Time{}) {
			earliestCreation = info.Creation
		}
	}
	if earliestCreation.Before(target.Creation) || target.Creation.Equal(time.Time{}) {
		target.Creation = earliestCreation
	}

	return nil
}
