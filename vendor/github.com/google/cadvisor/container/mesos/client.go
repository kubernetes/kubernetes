// Copyright 2018 Google Inc. All Rights Reserved.
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

package mesos

import (
	"fmt"
	"github.com/Rican7/retry"
	"github.com/Rican7/retry/strategy"
	"github.com/mesos/mesos-go/api/v1/lib"
	"github.com/mesos/mesos-go/api/v1/lib/agent"
	"github.com/mesos/mesos-go/api/v1/lib/agent/calls"
	mclient "github.com/mesos/mesos-go/api/v1/lib/client"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/codecs"
	"github.com/mesos/mesos-go/api/v1/lib/httpcli"
	"net/url"
	"sync"
)

const (
	maxRetryAttempts = 3
	invalidPID       = -1
)

var (
	mesosClientOnce sync.Once
	mesosClient     *client
)

type client struct {
	hc *httpcli.Client
}

type mesosAgentClient interface {
	ContainerInfo(id string) (*containerInfo, error)
	ContainerPid(id string) (int, error)
}

type containerInfo struct {
	cntr   *mContainer
	labels map[string]string
}

// Client is an interface to query mesos agent http endpoints
func Client() (mesosAgentClient, error) {
	mesosClientOnce.Do(func() {
		// Start Client
		apiURL := url.URL{
			Scheme: "http",
			Host:   *MesosAgentAddress,
			Path:   "/api/v1",
		}

		mesosClient = &client{
			hc: httpcli.New(
				httpcli.Endpoint(apiURL.String()),
				httpcli.Codec(codecs.ByMediaType[codecs.MediaTypeProtobuf]),
				httpcli.Do(httpcli.With(httpcli.Timeout(*MesosAgentTimeout))),
			),
		}
	})
	return mesosClient, nil
}

// ContainerInfo returns the container information of the given container id
func (self *client) ContainerInfo(id string) (*containerInfo, error) {
	c, err := self.getContainer(id)
	if err != nil {
		return nil, err
	}

	// Get labels of the container
	l, err := self.getLabels(c)
	if err != nil {
		return nil, err
	}

	return &containerInfo{
		cntr:   c,
		labels: l,
	}, nil
}

// Get the Pid of the container
func (self *client) ContainerPid(id string) (int, error) {
	var pid int
	var err error
	err = retry.Retry(
		func(attempt uint) error {
			c, err := self.ContainerInfo(id)
			if err != nil {
				return err
			}

			if c.cntr.ContainerStatus != nil {
				pid = int(*c.cntr.ContainerStatus.ExecutorPID)
			} else {
				err = fmt.Errorf("error fetching Pid")
			}
			return err
		},
		strategy.Limit(maxRetryAttempts),
	)
	if err != nil {
		return invalidPID, fmt.Errorf("failed to fetch pid")
	}
	return pid, err
}

func (self *client) getContainer(id string) (*mContainer, error) {
	// Get all containers
	cntrs, err := self.getContainers()
	if err != nil {
		return nil, err
	}

	// Check if there is a container with given id and return the container
	for _, c := range cntrs.Containers {
		if c.ContainerID.Value == id {
			return &c, nil
		}
	}
	return nil, fmt.Errorf("can't locate container %s", id)
}

func (self *client) getContainers() (mContainers, error) {
	req := calls.NonStreaming(calls.GetContainers())
	result, err := self.fetchAndDecode(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get mesos containers: %v", err)
	}
	cntrs := result.GetContainers
	return cntrs, nil
}

func (self *client) getLabels(c *mContainer) (map[string]string, error) {
	// Get mesos agent state which contains all containers labels
	var s state
	req := calls.NonStreaming(calls.GetState())
	result, err := self.fetchAndDecode(req)
	if err != nil {
		return map[string]string{}, fmt.Errorf("failed to get mesos agent state: %v", err)
	}
	s.st = result.GetState

	// Fetch labels from state object
	labels, err := s.FetchLabels(c.FrameworkID.Value, c.ExecutorID.Value)
	if err != nil {
		return labels, fmt.Errorf("error while fetching labels from executor: %v", err)
	}

	return labels, nil
}

func (self *client) fetchAndDecode(req calls.RequestFunc) (*agent.Response, error) {
	var res mesos.Response
	var err error

	// Send request
	err = retry.Retry(
		func(attempt uint) error {
			res, err = mesosClient.hc.Send(req, mclient.ResponseClassSingleton, nil)
			return err
		},
		strategy.Limit(maxRetryAttempts),
	)
	if err != nil {
		return nil, fmt.Errorf("error fetching %s: %s", req.Call(), err)
	}

	// Decode the result
	var target agent.Response
	err = res.Decode(&target)
	if err != nil {
		return nil, fmt.Errorf("error while decoding response body from %s: %s", res, err)
	}

	return &target, nil
}
