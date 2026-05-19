//go:build !libpfm || !cgo
// +build !libpfm !cgo

// Copyright 2020 Google Inc. All Rights Reserved.
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

// Manager of perf events for containers.
package perf

import (
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/stats"

	"k8s.io/klog/v2"
)

func NewManager(configFile string, topology []info.Node) (stats.Manager, error) {
	klog.V(1).Info("cAdvisor is build without cgo and/or libpfm support. Perf event counters are not available.")
	return &stats.NoopManager{}, nil
}
