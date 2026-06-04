/*
Copyright 2018 The Kubernetes Authors.

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

package eviction

import (
	"k8s.io/klog/v2"
	"time"
)

const (
	// this prevents constantly updating the memcg notifier if synchronize
	// is run frequently.
	notifierRefreshInterval = 10 * time.Second
)

// CgroupNotifierFactory knows how to make CgroupNotifiers which integrate with the kernel
type CgroupNotifierFactory struct{}

var _ NotifierFactory = &CgroupNotifierFactory{}

// NewCgroupNotifier implements the NotifierFactory interface
func (n *CgroupNotifierFactory) NewCgroupNotifier(logger klog.Logger, path, attribute string, threshold int64) (CgroupNotifier, error) {
	return NewCgroupNotifier(logger, path, attribute, threshold)
}
