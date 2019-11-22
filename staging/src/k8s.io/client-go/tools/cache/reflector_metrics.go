/*
Copyright 2016 The Kubernetes Authors.

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

package cache

import (
	"github.com/prometheus/client_golang/prometheus"
)

const (
	reflectorSubsystem = "reflector"
)

var (
	// reflectorListCount is the number of list requests issued by a reflector in order to start
	// a watch broken down by the group-version-kind and the resource-version used.
	reflectorListCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: reflectorSubsystem,
			Name:      "watch_list_count",
			Help:      "Number of list requests issued by reflector to start a watch",
		},
		[]string{
			"resource_type",    // Type of the resource the reflector is watching.
			"resource_version", // Resource version [empty, zero, explicit] used in the list request.
		},
	)
)

func resourceVersionToLabelValue(rv string) string {
	switch rv {
	case "":
		return "empty"
	case "0":
		return "zero"
	default:
		return "explicit"
	}
}

func init() {
	prometheus.MustRegister(reflectorListCount)
}
