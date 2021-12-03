//go:build !dockerless
// +build !dockerless

/*
Copyright 2020 The Kubernetes Authors.

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

package dockershim

import (
	"context"
	"fmt"
	"io"
	"testing"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/remotecommand"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	mockclient "k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker/testing"
)

func TestExecInContainer(t *testing.T) {

	testcases := []struct {
		description        string
		timeout            time.Duration
		returnCreateExec1  *dockertypes.IDResponse
		returnCreateExec2  error
		returnStartExec    error
		returnInspectExec1 *dockertypes.ContainerExecInspect
		returnInspectExec2 error
		execProbeTimeout   bool
		startExecDelay     time.Duration
		expectError        error
	}{{
		description:       "ExecInContainer succeeds",
		timeout:           time.Minute,
		returnCreateExec1: &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2: nil,
		returnStartExec:   nil,
		returnInspectExec1: &dockertypes.ContainerExecInspect{
			ExecID:      "200",
			ContainerID: "12345678",
			Running:     false,
			ExitCode:    0,
			Pid:         100},
		returnInspectExec2: nil,
		execProbeTimeout:   true,
		expectError:        nil,
	}, {
		description:        "CreateExec returns an error",
		timeout:            time.Minute,
		returnCreateExec1:  nil,
		returnCreateExec2:  fmt.Errorf("error in CreateExec()"),
		returnStartExec:    nil,
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		execProbeTimeout:   true,
		expectError:        fmt.Errorf("failed to exec in container - Exec setup failed - error in CreateExec()"),
	}, {
		description:        "StartExec returns an error",
		timeout:            time.Minute,
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		returnStartExec:    fmt.Errorf("error in StartExec()"),
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		execProbeTimeout:   true,
		expectError:        fmt.Errorf("error in StartExec()"),
	}, {
		description:        "InspectExec returns an error",
		timeout:            time.Minute,
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		returnStartExec:    nil,
		returnInspectExec1: nil,
		returnInspectExec2: fmt.Errorf("error in InspectExec()"),
		execProbeTimeout:   true,
		expectError:        fmt.Errorf("error in InspectExec()"),
	}, {
		description:       "ExecInContainer returns context DeadlineExceeded",
		timeout:           1 * time.Second,
		returnCreateExec1: &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2: nil,
		returnStartExec:   context.DeadlineExceeded,
		returnInspectExec1: &dockertypes.ContainerExecInspect{
			ExecID:      "200",
			ContainerID: "12345678",
			Running:     true,
			ExitCode:    0,
			Pid:         100},
		returnInspectExec2: nil,
		execProbeTimeout:   true,
		expectError:        context.DeadlineExceeded,
	}, {
		description:        "[ExecProbeTimeout=true] StartExec that takes longer than the probe timeout returns context.DeadlineExceeded",
		timeout:            1 * time.Second,
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		startExecDelay:     5 * time.Second,
		returnStartExec:    fmt.Errorf("error in StartExec()"),
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		execProbeTimeout:   true,
		expectError:        context.DeadlineExceeded,
	}, {
		description:        "[ExecProbeTimeout=false] StartExec that takes longer than the probe timeout returns a error",
		timeout:            1 * time.Second,
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		startExecDelay:     5 * time.Second,
		returnStartExec:    fmt.Errorf("error in StartExec()"),
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		execProbeTimeout:   false,
		expectError:        fmt.Errorf("error in StartExec()"),
	}}

	eh := &NativeExecHandler{}
	ctrl := gomock.NewController(t)
	container := getFakeContainerJSON()
	cmd := []string{"/bin/bash"}
	var stdin io.Reader
	var stdout, stderr io.WriteCloser
	var resize <-chan remotecommand.TerminalSize

	for _, tc := range testcases {
		// these tests cannot be run in parallel due to the fact that they are feature gate dependent
		tc := tc
		t.Run(tc.description, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExecProbeTimeout, tc.execProbeTimeout)()

			mockClient := mockclient.NewMockInterface(ctrl)
			mockClient.EXPECT().CreateExec(gomock.Any(), gomock.Any()).Return(
				tc.returnCreateExec1,
				tc.returnCreateExec2)
			mockClient.EXPECT().StartExec(gomock.Any(), gomock.Any(), gomock.Any()).Do(func(_ string, _ dockertypes.ExecStartCheck, _ libdocker.StreamOptions) { time.Sleep(tc.startExecDelay) }).Return(tc.returnStartExec)
			mockClient.EXPECT().InspectExec(gomock.Any()).Return(
				tc.returnInspectExec1,
				tc.returnInspectExec2)

			// use parent context of 2 minutes since that's the default remote
			// runtime connection timeout used by dockershim
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			err := eh.ExecInContainer(ctx, mockClient, container, cmd, stdin, stdout, stderr, false, resize, tc.timeout)
			assert.Equal(t, tc.expectError, err)
		})
	}
}

func getFakeContainerJSON() *dockertypes.ContainerJSON {
	return &dockertypes.ContainerJSON{
		ContainerJSONBase: &dockertypes.ContainerJSONBase{
			ID:    "12345678",
			Name:  "fake_name",
			Image: "fake_image",
			State: &dockertypes.ContainerState{
				Running:    false,
				ExitCode:   0,
				Pid:        100,
				StartedAt:  "2020-10-13T01:00:00-08:00",
				FinishedAt: "2020-10-13T02:00:00-08:00",
			},
			Created:    "2020-10-13T01:00:00-08:00",
			HostConfig: nil,
		},
		Config:          nil,
		NetworkSettings: &dockertypes.NetworkSettings{},
	}
}
