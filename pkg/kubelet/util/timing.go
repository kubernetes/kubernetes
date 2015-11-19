/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

// RecordStartTimeByAnnotation returns the current time if the proper annotation
// is on the pod. Otherwise it reurns a zero time
func RecordStartTimeByAnnotation(pod *api.Pod) time.Time {
	if _, ok := pod.Annotations["kuberenetes.io/podTiming"]; ok {
		return time.Now()
	}
	return time.Time{}
}

// RecordEndTimeByAnnotation logs the time difference if startTime is non-zero
func RecordEndTimeByAnnotation(startTime time.Time, pod *api.Pod, dataPoint string) {
	if !startTime.IsZero() {
		dataPoints, err := json.Marshal(map[string]string{
			"UID":       string(pod.UID),
			"PodName":   pod.Name,
			"DataPoint": dataPoint,
			"Timing":    time.Now().Sub(startTime).String(),
		})
		// Log the the marshal error but don't interrupt functionality
		if err != nil {
			glog.Errorf("Unable to record the pod timings. Error=%s", err)
		}
		glog.Infof("Pod Timing: %s", string(dataPoints))
	}
}
