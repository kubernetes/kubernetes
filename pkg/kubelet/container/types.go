/*
Copyright 2014 Google Inc. All rights reserved.

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

package container

import "time"

// Container represents a container.
type Container struct {
	ID              string            `json:"id"`
	Name            string            `json:"name,omitempty"`
	Names           []string          `json:"names,omitempty"`
	Image           string            `json:"image,omitempty"`
	ImageID         string            `json:"imageID`
	Command         string            `json:"command,omitempty"`
	Created         time.Time         `json:"created,omitempty"`
	State           State             `json:"state,omitempty"`
	Status          string            `json:"status,omitempty"`
	NetworkSettings *NetworkSettings  `json:"networkSettings`
	Ports           []Port            `json:"ports,omitempty"`
	SizeRw          int64             `json:"sizeRw,omitempty"`
	SizeRootFs      int64             `json:"sizeRootFs,omitempty"`
	Volumes         map[string]string `json:"volumes,omitempty"`
}

// Port is a type that represents a port mapping.
type Port struct {
	PrivatePort int64  `json:"privatePort,omitempty"`
	PublicPort  int64  `json:"publicPort,omitempty"`
	Type        string `json:"type,omitempty"`
	IP          string `json:"ip,omitempty"`
}

// ListContainersOptions specify parameters to the ListContainers function.
type ListContainersOptions struct {
	All bool
}

// State represents the state of a container.
type State struct {
	Running    bool      `json:"running,omitempty"`
	Paused     bool      `json:"paused,omitempty"`
	OOMKilled  bool      `json:"oomkilled,omitempty"`
	Pid        int       `json:"pid,omitempty"`
	ExitCode   int       `json:"exitCode,omitempty"`
	Error      string    `json:"error,omitempty"`
	StartedAt  time.Time `json:"startedAt,omitempty"`
	FinishedAt time.Time `json:"finishedAt,omitempty"`
}

// NetworkSettings contains network-related information about a container.
type NetworkSettings struct {
	IPAddress string `json:"ipAddress,omitempty"`
}
