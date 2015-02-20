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

import (
	"errors"
	"io"
	"time"

	"github.com/fsouza/go-dockerclient"
)

var (
	ErrNoSuchImage = errors.New("no such image")
)

// Container represents a container.
type Container struct {
	ID              string            `json:"id"`
	Name            string            `json:"name,omitempty"`
	Image           string            `json:"image,omitempty"`
	ImageID         string            `json:"imageID`
	Command         string            `json:"command,omitempty"`
	Created         time.Time         `json:"created,omitempty"`
	State           State             `json:"state,omitempty"`
	Status          string            `json:"status,omitempty"`
	NetworkSettings *NetworkSettings  `json:"networkSettings`
	SizeRw          int64             `json:"sizeRw,omitempty"`
	SizeRootFs      int64             `json:"sizeRootFs,omitempty"`
	Volumes         map[string]string `json:"volumes,omitempty"`
	Hostname        string            `json:"hostname,omitempty"`
	Env             []string          `json:"env,omitempty"`
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

// HostConfig contains the container options related to starting a container on a given host.
type HostConfig struct {
	Binds        []string                 `json:"binds,omitempty"`
	CapAdd       []string                 `json:"capAdd,omitempty"`
	CapDrop      []string                 `json:"capDrop,omitempty"`
	Privileged   bool                     `json:"privileged,omitempty"`
	PortBindings map[string][]PortBinding `json:"portBindings,omitempty"`
	DNS          []string                 `json:"dns,omitempty"`
	DNSSearch    []string                 `json:"dnsSearch,omitempty"`
	NetworkMode  string                   `json:"networkMode,omitempty"`
	IPCMode      string                   `json:"ipcMode,omitempty"`
}

// PortBinding represents the host/container port mapping.
type PortBinding struct {
	HostIP   string `json:"hostIP,omitempty"`
	HostPort string `json:"hostPort,omitempty"`
}

// Image represents a container image.
type Image struct {
	ID string `json:"id,omitempty"`
}

// ListContainersOptions specify parameters to the ListContainers function.
type ListContainersOptions struct {
	All bool `json:"all,omitempty"`
}

// CreateContainerOptions specify parameters to the CreateContainer function.
type CreateContainerOptions struct {
	Name         string              `json:"name,omitempty"`
	Command      []string            `json:"cmd,omitempty"`
	Env          []string            `json:"env,omitempty"`
	ExposedPorts map[string]struct{} `json:"exposedPorts,omitempty"`
	Hostname     string              `json:"hostname,omitempty"`
	Image        string              `json:"image,omitempty"`
	Memory       int64               `json:"memory,omitempty"`
	CPUShares    int64               `json:"cpuShares,omitempty"`
	WorkingDir   string              `json:"workingDir,omitempty`
}

// RemoveContainerOptions specify parameters to the RemoveContainer funcion.
type RemoveContainerOptions struct {
	ID string `json:"id,omitempty"`
}

// ListImagesOptions specify parameters to the ListImages function.
type ListImagesOptions struct {
	All bool `json:"all,omitempty"`
}

// PullImageOptions specify parameters to the PullImage function.
type PullImageOptions struct {
	Repository       string                   `json:"repostory,omitempty"`
	Tag              string                   `json:"tag,omitempty"`
	DockerAuthConfig docker.AuthConfiguration `json:"dockerAuth,omitempty"`
}

// LogsOptions specify parameters to the Logs function.
type LogsOptions struct {
	ID           string    `json:"id,omitempty"`
	OutputStream io.Writer `json:"-"`
	ErrorStream  io.Writer `json:"-"`
	Follow       bool      `json:"follow,omitempty"`
	Stdout       bool      `json:"stdout,omitempty"`
	Stderr       bool      `json:"stderr,omitempty"`
	Timestamps   bool      `json:"timestamp,omitempty"`
	Tail         string    `json:"tail,omitempty"`
	RawTerminal  bool      `json:"rawTerminal,omitempty"`
}

// CreateExecOptions specify parameters to the CreateExec function.
type CreateExecOptions struct {
	AttachStdin  bool     `json:"attachStdin,omitempty"`
	AttachStdout bool     `json:"attachStdout,omitempty"`
	AttachStderr bool     `json:"attachStderr,omitempty"`
	TTY          bool     `json:"tty,omitempty"`
	Command      []string `json:"command,omitempty"`
	Container    string   `json:"container,omitempty"`
}

// Exec represents an exec object.
type Exec struct {
	ID string `json:"id,omitempty"`
}

// StartExecOptions specify parameters to StartExec function.
type StartExecOptions struct {
	Detach bool `json:"detach,omitempty"`

	TTY bool `json:"tty,omitempty"`

	InputStream  io.Reader `json:"-"`
	OutputStream io.Writer `json:"-"`
	ErrorStream  io.Writer `json:"-"`

	// Use raw terminal? Usually true when the container contains a TTY.
	RawTerminal bool `json:"-"`
}
