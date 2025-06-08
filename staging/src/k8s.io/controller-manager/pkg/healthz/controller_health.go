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
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/tools/cache"
)

// ControllerHealthCheckable is a helper type that implements the controller.HealthCheckable
// interface for controllers that want to expose their informer sync status as a health check.
type ControllerHealthCheckable struct {
	controllerName string
	syncFuncs      []cache.InformerSynced
}

// NewControllerHealthCheckable creates a new ControllerHealthCheckable that implements
// the controller.HealthCheckable interface.
func NewControllerHealthCheckable(controllerName string, syncFuncs ...cache.InformerSynced) *ControllerHealthCheckable {
	return &ControllerHealthCheckable{
		controllerName: controllerName,
		syncFuncs:      syncFuncs,
	}
}

// HealthChecker returns a health checker that verifies all informers have synced their caches.
func (c *ControllerHealthCheckable) HealthChecker() healthz.HealthChecker {
	return NewInformerSyncHealthChecker(c.controllerName, c.syncFuncs...)
}

// WithInformerSyncHealthCheck is a helper function that makes it easy for controllers
// to implement the controller.HealthCheckable interface. It takes a controller name and
// a list of informer sync functions and returns a controller.HealthCheckable implementation.
//
// Example usage:
//
//	type MyController struct {
//		*healthz.ControllerHealthCheckable
//		// ... other fields
//	}
//
//	func NewMyController(...) *MyController {
//		syncFuncs := []cache.InformerSynced{
//			podInformer.Informer().HasSynced,
//			serviceInformer.Informer().HasSynced,
//		}
//		return &MyController{
//			ControllerHealthCheckable: healthz.NewControllerHealthCheckable("my-controller", syncFuncs...),
//			// ... initialize other fields
//		}
//	}
func WithInformerSyncHealthCheck(controllerName string, syncFuncs ...cache.InformerSynced) interface{ HealthChecker() healthz.HealthChecker } {
	return NewControllerHealthCheckable(controllerName, syncFuncs...)
}
