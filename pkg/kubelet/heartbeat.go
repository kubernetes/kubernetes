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

package kubelet

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/client-go/util/clock"
)

const (
	// These constants should be reviewed and updated very carefully.
	// nodeStatusUpdateRetry the normal default for how many times kubelet retries when posting node status failed.
	// this constant is a convention that is expected to be depended on externaly by other components.
	NodeStatusUpdateRetry      = 5
	nodeRegistrationMaxBackoff = 7 * time.Second

	// Less critical constants subject to change.
	nodeStatusBurstRetry = 5 * time.Millisecond
)

// This module externalizes logic for heartbeating and retrying of communication with the API server.
// It is intended to externalize functionality that other components (like the controller) may depend on

// registerInitiallyWithInfiniteRetry continually attempts to register, returning an error if it ever gives up.
// We retry infinitely because registration is a prerequisite to proper kubelet functioning.
func registerInitiallyWithInfiniteRetry(theClock clock.Clock, register func() bool) error {
	try := 0
	step := 100 * time.Millisecond
	complete := false
	for !complete {
		glog.Infof("Attempting to register the apiserver (attempt %v)", try)
		try = try + 1
		theClock.Sleep(step)
		step = step * 2
		if step >= nodeRegistrationMaxBackoff {
			step = nodeRegistrationMaxBackoff
		}
		complete = register()
	}
	glog.Infof("Succesfully registered with API server after %v tries.", try)
	return nil
}

// updateNodeStatusWithRetry retries to update Node status several times in a short burst.  We don't have
// a traditional backoff here because we want to implement a heartbeat style for communication, controlled via
// the kubelet's sync loop.  This may change in the future.
func updateNodeStatusWithRetry(theClock clock.Clock, update func() error) error {
	for i := 0; i < NodeStatusUpdateRetry; i++ {
		if err := update(); err != nil {
			// TODO: Considering to put some small exponential backoff (1ms->100ms)
			theClock.Sleep(nodeStatusBurstRetry)
		} else {
			return nil
		}
	}
	return fmt.Errorf("Failed at updating status %v times consecutively.  Giving up !", NodeStatusUpdateRetry)
}
