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

package healthz

import (
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// InformerSyncHealthChecker is a health checker that verifies all informers have synced their caches.
type InformerSyncHealthChecker struct {
	controllerName string
	syncFuncs      []cache.InformerSynced
}

// NewInformerSyncHealthChecker creates a new InformerSyncHealthChecker that verifies
// all informers have synced their caches.
func NewInformerSyncHealthChecker(controllerName string, syncFuncs ...cache.InformerSynced) healthz.HealthChecker {
	return &InformerSyncHealthChecker{
		controllerName: controllerName,
		syncFuncs:      syncFuncs,
	}
}

// Check verifies that all informers have synced their caches.
func (c *InformerSyncHealthChecker) Check(_ *http.Request) error {
	if !cache.WaitForCacheSync(nil, c.syncFuncs...) {
		klog.Errorf("Controller %s informers have not started", c.controllerName)
		return fmt.Errorf("controller %s informers have not started", c.controllerName)
	}
	return nil
}

// Name returns the name of the health checker.
func (c *InformerSyncHealthChecker) Name() string {
	return fmt.Sprintf("informer-sync-%s", c.controllerName)
}
