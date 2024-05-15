/*
Copyright 2024 The Kubernetes Authors.

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

// Package util contains kubeadm utilities.
package util

import (
	"time"
)

// startTime is a variable that represents the start time of the kubeadm process.
// It can be used to consistently use the same start time instead of calling time.Now()
// in multiple locations and ending up with minor time deviations.
var startTime time.Time

func init() {
	startTime = time.Now()
}

// StartTimeUTC returns startTime with its location set to UTC.
func StartTimeUTC() time.Time {
	return startTime.UTC()
}
