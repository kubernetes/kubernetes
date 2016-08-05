/*
Copyright 2015 The Kubernetes Authors.

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

package config

import (
	"k8s.io/kubernetes/pkg/api/resource"
)

const (
	DefaultLogMaxBackups   = 5 // how many backup to keep
	DefaultLogMaxAgeInDays = 7 // after how many days to rotate at most

	DefaultCgroupPrefix = "mesos"
)

// DefaultLogMaxSize returns the maximal log file size before rotation
func DefaultLogMaxSize() resource.Quantity {
	return *resource.NewQuantity(10*1024*1024, resource.BinarySI)
}
