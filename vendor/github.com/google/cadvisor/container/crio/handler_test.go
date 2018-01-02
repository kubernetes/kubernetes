// Copyright 2017 Google Inc. All Rights Reserved.
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

package crio

import (
	"fmt"
	"testing"

	"github.com/google/cadvisor/container"
	containerlibcontainer "github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

func TestHandler(t *testing.T) {
	as := assert.New(t)
	type testCase struct {
		client             crioClient
		name               string
		machineInfoFactory info.MachineInfoFactory
		fsInfo             fs.FsInfo
		storageDriver      storageDriver
		storageDir         string
		cgroupSubsystems   *containerlibcontainer.CgroupSubsystems
		inHostNamespace    bool
		metadataEnvs       []string
		ignoreMetrics      container.MetricSet

		hasErr         bool
		errContains    string
		checkReference *info.ContainerReference
	}
	for _, ts := range []testCase{
		{
			mockCrioClient(Info{}, nil, fmt.Errorf("no client returned")),
			"/kubepods/pod068e8fa0-9213-11e7-a01f-507b9d4141fa/crio-81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
			nil,
			nil,
			"",
			"",
			&containerlibcontainer.CgroupSubsystems{},
			false,
			nil,
			nil,

			true,
			"no client returned",
			nil,
		},
		{
			mockCrioClient(Info{}, nil, nil),
			"/kubepods/pod068e8fa0-9213-11e7-a01f-507b9d4141fa/crio-81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
			nil,
			nil,
			"",
			"",
			&containerlibcontainer.CgroupSubsystems{},
			false,
			nil,
			nil,

			true,
			"no container with id 81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
			nil,
		},
		{
			mockCrioClient(
				Info{},
				map[string]*ContainerInfo{"81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f": {Name: "test", Labels: map[string]string{"io.kubernetes.container.name": "POD"}}},
				nil,
			),
			"/kubepods/pod068e8fa0-9213-11e7-a01f-507b9d4141fa/crio-81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
			nil,
			nil,
			"",
			"",
			&containerlibcontainer.CgroupSubsystems{},
			false,
			nil,
			nil,

			false,
			"",
			&info.ContainerReference{
				Id:        "81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
				Name:      "/kubepods/pod068e8fa0-9213-11e7-a01f-507b9d4141fa/crio-81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f",
				Aliases:   []string{"test", "81e5c2990803c383229c9680ce964738d5e566d97f5bd436ac34808d2ec75d5f"},
				Namespace: CrioNamespace,
				Labels:    map[string]string{"io.kubernetes.container.name": "POD"},
			},
		},
	} {
		handler, err := newCrioContainerHandler(ts.client, ts.name, ts.machineInfoFactory, ts.fsInfo, ts.storageDriver, ts.storageDir, ts.cgroupSubsystems, ts.inHostNamespace, ts.metadataEnvs, ts.ignoreMetrics)
		if ts.hasErr {
			as.NotNil(err)
			if ts.errContains != "" {
				as.Contains(err.Error(), ts.errContains)
			}
		}
		if ts.checkReference != nil {
			cr, err := handler.ContainerReference()
			as.Nil(err)
			as.Equal(*ts.checkReference, cr)
		}
	}
}
