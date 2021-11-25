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

package docker

import (
	"time"

	"golang.org/x/net/context"
	"k8s.io/klog/v2"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/watcher"
)

const dockerClientTimeout = 10 * time.Second

// NewPlugin returns an implementation of container.Plugin suitable for passing to container.RegisterPlugin()
func NewPlugin() container.Plugin {
	return &plugin{}
}

type plugin struct{}

func (p *plugin) InitializeFSContext(context *fs.Context) error {
	SetTimeout(dockerClientTimeout)
	// Try to connect to docker indefinitely on startup.
	dockerStatus := retryDockerStatus()
	context.Docker = fs.DockerContext{
		Root:         RootDir(),
		Driver:       dockerStatus.Driver,
		DriverStatus: dockerStatus.DriverStatus,
	}
	return nil
}

func (p *plugin) Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics container.MetricSet) (watcher.ContainerWatcher, error) {
	err := Register(factory, fsInfo, includedMetrics)
	return nil, err
}

func retryDockerStatus() info.DockerStatus {
	startupTimeout := dockerClientTimeout
	maxTimeout := 4 * startupTimeout
	for {
		ctx, _ := context.WithTimeout(context.Background(), startupTimeout)
		dockerStatus, err := StatusWithContext(ctx)
		if err == nil {
			return dockerStatus
		}

		switch err {
		case context.DeadlineExceeded:
			klog.Warningf("Timeout trying to communicate with docker during initialization, will retry")
		default:
			klog.V(5).Infof("Docker not connected: %v", err)
			return info.DockerStatus{}
		}

		startupTimeout = 2 * startupTimeout
		if startupTimeout > maxTimeout {
			startupTimeout = maxTimeout
		}
	}
}
