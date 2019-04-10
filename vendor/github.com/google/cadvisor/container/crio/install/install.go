// Copyright 2019 Google Inc. All Rights Reserved.
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

// The install package installs the crio container provider when imported
package install

import (
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/crio"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/watcher"
	"k8s.io/klog"
)

func init() {
	err := container.RegisterPlugin("crio", container.PluginFn{
		InitializeContextFn: func(context *fs.Context) error {
			crioClient, err := crio.Client()
			if err != nil {
				return err
			}

			crioInfo, err := crioClient.Info()
			if err != nil {
				klog.V(5).Infof("CRI-O not connected: %v", err)
			} else {
				context.Crio = fs.CrioContext{Root: crioInfo.StorageRoot}
			}
			return nil
		},
		RegisterFn: func(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics container.MetricSet) (watcher.ContainerWatcher, error) {
			err := crio.Register(factory, fsInfo, includedMetrics)
			return nil, err
		},
	})
	if err != nil {
		klog.Fatalf("Failed to register crio plugin: %v", err)
	}
}
