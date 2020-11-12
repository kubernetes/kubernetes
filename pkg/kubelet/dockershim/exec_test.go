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
	"fmt"
	"io"
	"testing"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	"k8s.io/client-go/tools/remotecommand"

	mockclient "k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker/testing"
)

func TestExecInContainer(t *testing.T) {

	testcases := []struct {
		description        string
		returnCreateExec1  *dockertypes.IDResponse
		returnCreateExec2  error
		returnStartExec    error
		returnInspectExec1 *dockertypes.ContainerExecInspect
		returnInspectExec2 error
		expectError        error
	}{{
		description:       "ExecInContainer succeeds",
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
		expectError:        nil,
	}, {
		description:        "CreateExec returns an error",
		returnCreateExec1:  nil,
		returnCreateExec2:  fmt.Errorf("error in CreateExec()"),
		returnStartExec:    nil,
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		expectError:        fmt.Errorf("failed to exec in container - Exec setup failed - error in CreateExec()"),
	}, {
		description:        "StartExec returns an error",
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		returnStartExec:    fmt.Errorf("error in StartExec()"),
		returnInspectExec1: nil,
		returnInspectExec2: nil,
		expectError:        fmt.Errorf("error in StartExec()"),
	}, {
		description:        "InspectExec returns an error",
		returnCreateExec1:  &dockertypes.IDResponse{ID: "12345678"},
		returnCreateExec2:  nil,
		returnStartExec:    nil,
		returnInspectExec1: nil,
		returnInspectExec2: fmt.Errorf("error in InspectExec()"),
		expectError:        fmt.Errorf("error in InspectExec()"),
	}}

	eh := &NativeExecHandler{}
	ctrl := gomock.NewController(t)
	container := getFakeContainerJSON()
	cmd := []string{"/bin/bash"}
	var stdin io.Reader
	var stdout, stderr io.WriteCloser
	var resize <-chan remotecommand.TerminalSize
	var timeout time.Duration

	for _, tc := range testcases {
		t.Logf("TestCase: %q", tc.description)

		mockClient := mockclient.NewMockInterface(ctrl)
		mockClient.EXPECT().CreateExec(gomock.Any(), gomock.Any()).Return(
			tc.returnCreateExec1,
			tc.returnCreateExec2)
		mockClient.EXPECT().StartExec(gomock.Any(), gomock.Any(), gomock.Any()).Return(tc.returnStartExec)
		mockClient.EXPECT().InspectExec(gomock.Any()).Return(
			tc.returnInspectExec1,
			tc.returnInspectExec2)
		err := eh.ExecInContainer(mockClient, container, cmd, stdin, stdout, stderr, false, resize, timeout)
		assert.Equal(t, err, tc.expectError)
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
