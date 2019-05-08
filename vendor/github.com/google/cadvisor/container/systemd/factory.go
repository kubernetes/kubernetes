// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package systemd

import (
	"fmt"
	"strings"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/watcher"

	"k8s.io/klog"
)

type systemdFactory struct{}

func (f *systemdFactory) String() string {
	return "systemd"
}

func (f *systemdFactory) NewContainerHandler(name string, inHostNamespace bool) (container.ContainerHandler, error) {
	return nil, fmt.Errorf("Not yet supported")
}

func (f *systemdFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	// on systemd using devicemapper each mount into the container has an associated cgroup that we ignore.
	// for details on .mount units: http://man7.org/linux/man-pages/man5/systemd.mount.5.html
	if strings.HasSuffix(name, ".mount") {
		return true, false, nil
	}
	return false, false, fmt.Errorf("%s not handled by systemd handler", name)
}

func (f *systemdFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

// Register registers the systemd container factory.
func Register(machineInfoFactory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics container.MetricSet) error {
	klog.V(1).Infof("Registering systemd factory")
	factory := &systemdFactory{}
	container.RegisterContainerHandlerFactory(factory, []watcher.ContainerWatchSource{watcher.Raw})
	return nil
}
