// Copyright 2015 Google Inc. All Rights Reserved.
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

package cpuload

import (
	"fmt"

	info "github.com/google/cadvisor/info/v1"

	"github.com/golang/glog"
	"github.com/google/cadvisor/utils/cpuload/netlink"
)

type CpuLoadReader interface {
	// Start the reader.
	Start() error

	// Stop the reader and clean up internal state.
	Stop()

	// Retrieve Cpu load for a given group.
	// name is the full hierarchical name of the container.
	// Path is an absolute filesystem path for a container under CPU cgroup hierarchy.
	GetCpuLoad(name string, path string) (info.LoadStats, error)
}

func New() (CpuLoadReader, error) {
	reader, err := netlink.New()
	if err != nil {
		return nil, fmt.Errorf("failed to create a netlink based cpuload reader: %v", err)
	}
	glog.V(3).Info("Using a netlink-based load reader")
	return reader, nil
}
