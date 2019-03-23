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
	"github.com/mesos/mesos-go/api/v1/lib"
	"github.com/mesos/mesos-go/api/v1/lib/agent"
)

const (
	cpus         = "cpus"
	schedulerSLA = "scheduler_sla"
	framework    = "framework"
	source       = "source"
	revocable    = "revocable"
	nonRevocable = "non_revocable"
)

type mContainers *agent.Response_GetContainers
type mContainer = agent.Response_GetContainers_Container

type (
	state struct {
		st *agent.Response_GetState
	}
)

// GetFramework finds a framework with the given id and returns nil if not found. Note that
// this is different from the framework name.
func (s *state) GetFramework(id string) (*mesos.FrameworkInfo, error) {
	for _, fw := range s.st.GetFrameworks.Frameworks {
		if fw.FrameworkInfo.ID.Value == id {
			return &fw.FrameworkInfo, nil
		}
	}
	return nil, fmt.Errorf("unable to find framework id %s", id)
}

// GetExecutor finds an executor with the given ID and returns nil if not found. Note that
// this is different from the executor name.
func (s *state) GetExecutor(id string) (*mesos.ExecutorInfo, error) {
	for _, exec := range s.st.GetExecutors.Executors {
		if exec.ExecutorInfo.ExecutorID.Value == id {
			return &exec.ExecutorInfo, nil
		}
	}
	return nil, fmt.Errorf("unable to find executor with id %s", id)
}

// GetTask returns a task launched by given executor.
func (s *state) GetTask(exID string) (*mesos.Task, error) {
	// Check if task is in Launched Tasks list
	for _, t := range s.st.GetTasks.LaunchedTasks {
		if s.isMatchingTask(&t, exID) {
			return &t, nil
		}
	}

	// Check if task is in Queued Tasks list
	for _, t := range s.st.GetTasks.QueuedTasks {
		if s.isMatchingTask(&t, exID) {
			return &t, nil
		}
	}
	return nil, fmt.Errorf("unable to find task matching executor id %s", exID)
}

func (s *state) isMatchingTask(t *mesos.Task, exID string) bool {
	// MESOS-9111: For tasks launched through mesos command/default executor, the
	// executorID(which is same as the taskID) field is not filled in the TaskInfo object.
	// The workaround is compare with taskID field if executorID is empty
	if t.ExecutorID != nil {
		if t.ExecutorID.Value == exID {
			return true
		}
	} else {
		if t.TaskID.Value == exID {
			return true
		}
	}

	return false
}

func (s *state) fetchLabelsFromTask(exID string, labels map[string]string) error {
	t, err := s.GetTask(exID)
	if err != nil {
		return err
	}

	// Identify revocability. Can be removed once we have a proper label
	for _, resource := range t.Resources {
		if resource.Name == cpus {
			if resource.Revocable != nil {
				labels[schedulerSLA] = revocable
			} else {
				labels[schedulerSLA] = nonRevocable
			}
			break
		}
	}

	if t.Labels != nil {
		for _, l := range t.Labels.Labels {
			labels[l.Key] = *l.Value
		}
	}

	return nil
}

func (s *state) FetchLabels(fwID string, exID string) (map[string]string, error) {
	labels := make(map[string]string)

	// Look for the framework which launched the container.
	fw, err := s.GetFramework(fwID)
	if err != nil {
		return labels, fmt.Errorf("framework ID %q not found: %v", fwID, err)
	}
	labels[framework] = fw.Name

	// Get the executor info of the container which contains all the task info.
	exec, err := s.GetExecutor(exID)
	if err != nil {
		return labels, fmt.Errorf("executor ID %q not found: %v", exID, err)
	}

	labels[source] = *exec.Source

	err = s.fetchLabelsFromTask(exID, labels)
	if err != nil {
		return labels, fmt.Errorf("failed to fetch labels from task with executor ID %s", exID)
	}

	return labels, nil
}
