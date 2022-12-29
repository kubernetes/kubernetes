/*
Copyright 2014 The Kubernetes Authors.

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

package controlplane

import (
	"context"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
)

// Controller is only used to keep backwards compatibility
// and publish the "poststarthook/bootstrap-controller" health/readyz/livez handler
// All its functions were split to the following controllers:
// - start-default-service-controller
// - start-system-namespaces-controller
// - start-services-repair-controller
// and its state depend on the state of the previous.
type Controller struct {
	client kubernetes.Interface
}

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *completedConfig) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage, client kubernetes.Interface) (*Controller, error) {

	return &Controller{
		client: client,
	}, nil
}

// Start checks the controllers are ready
func (c *Controller) Start(stopCh <-chan struct{}) {
	klog.Infof("Starting bootstrap-controller")
	defer klog.Infof("Shutting down bootstrap-controller")
	requiredChecks := sets.NewString(
		"[+]poststarthook/start-default-service-controller ok",
		"[+]poststarthook/start-system-namespaces-controller ok",
		"[+]poststarthook/start-services-repair-controller ok",
	)
	wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		body, err := c.client.CoreV1().RESTClient().Get().RequestURI("/healthz?verbose=1").DoRaw(context.Background())
		if err == nil {
			klog.Fatal("API server ready but bootstrap controller is not ready")
		}
		checks := sets.NewString(strings.Split(string(body), "\n")...)
		if missing := requiredChecks.Difference(checks); missing.Len() > 0 {
			return false, nil
		}
		return true, nil
	}, stopCh)
}

// PostStartHook initiates the core controller loops that must exist for bootstrapping.
func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	c.Start(hookContext.StopCh)
	return nil
}
