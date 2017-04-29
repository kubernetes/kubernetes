/*
Copyright 2017 The Kubernetes Authors.

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

// registerInitiallyWithInfiniteRetry continually runs a function until it succeeds, backing off
// using specific timings we set for initial registration flow.  Specifically designed for initial registration,
//  externalized to make timing policy easily maintainable.
func registerInitiallyWithInfiniteRetry(theClock clock.Clock, register func() bool, message string) error {
	try := 0
	step := 100 * time.Millisecond
	complete := false
	for !complete {
		glog.Infof("Attempting to run registration function (%v) (attempt %v)", message, try)
		try = try + 1
		theClock.Sleep(step)
		step = step * 2
		if step >= nodeRegistrationMaxBackoff {
			step = nodeRegistrationMaxBackoff
		}
		complete = register()
	}
	glog.Infof("Succesfully completed registeration function (%v) after %v tries.", message, try)
	return nil
}

// updateNodeStatusWithBurstRetry runs a function several times with burst retries.  Specifically designed for
// node status updates, externalized to make timing policy easily maintainable.
func updateNodeStatusWithBurstRetry(theClock clock.Clock, update func() error, message string) error {
	for i := 0; i < NodeStatusUpdateRetry; i++ {
		if err := update(); err != nil {
			// TODO: Consider a small exponential backoff (1ms->100ms).
			theClock.Sleep(nodeStatusBurstRetry)
		} else {
			return nil
		}
	}
	return fmt.Errorf("Failed at running update function (%v), %v times consecutively.  Giving up !", message, NodeStatusUpdateRetry)
}
