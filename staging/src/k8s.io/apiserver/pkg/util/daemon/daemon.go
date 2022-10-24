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

package daemon

import (
	"github.com/coreos/go-systemd/v22/daemon"
	"k8s.io/klog/v2"
)

// If systemd is used, notify it that we have started
func AsyncSdNotify() {
	go func() {
		if _, err := daemon.SdNotify(false, "READY=1\n"); err != nil {
			klog.ErrorS(err, "Unable to send systemd daemon successful start message")
		}
	}()
}
