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
	"strings"
	"time"

	"github.com/fsouza/go-dockerclient"
)

var (
	ErrNoSuchImage = errors.New("no such image")
)

// Container represents a container.
type Container struct {
	// The id of the container.
	ID string `json:"id"`
	// The name of the container.
	Name string `json:"name,omitempty"`
	// The host name of the container.
	Hostname string `json:"hostname,omitempty"`
	// The name of the container's image.
	Image string `json:"image,omitempty"`
	// The id of the container's image.
	ImageID string `json:"image,omitempty"`
	// The State of the container, including when
	// the container is created/started.
	State State `json:"state,omitempty"`
	// The network settings of the container, including
	// the ip of the container and its port bindings.
	NetworkSettings *NetworkSettings `json:"networkSettings`
	// The volumes binded to the container.
	Volumes map[string]string `json:"volumes,omitempty"`
	// The environment variables of the container.
	Env []string `json:"env,omitempty"`
}

// State represents the state of a container.
type State struct {
	Running    bool      `json:"running,omitempty"`
	Paused     bool      `json:"paused,omitempty"`
	OOMKilled  bool      `json:"oomkilled,omitempty"`
	Pid        int       `json:"pid,omitempty"`
	ExitCode   int       `json:"exitCode,omitempty"`
	Error      string    `json:"error,omitempty"`
	CreatedAt  time.Time `json:"created,omitempty"`
	StartedAt  time.Time `json:"startedAt,omitempty"`
	FinishedAt time.Time `json:"finishedAt,omitempty"`
}

// NetworkSettings contains network-related information about a container.
type NetworkSettings struct {
	// The ip address of the container.
	IPAddress string `json:"ipAddress,omitempty"`
	// The port bindings of the container.
	PortBindings map[Port][]PortBinding `json:"portBindings,omitempty"`
}

// HostConfig contains the container options related to starting a container on a given host.
type HostConfig struct {
	// The bindings of the volumes.
	Binds []string `json:"binds,omitempty"`
	// The Linux capabilities added to the container.
	CapAdd []string `json:"capAdd,omitempty"`
	// The Linux capabilities dropped from the container.
	CapDrop []string `json:"capDrop,omitempty"`
	// If give extended privileges to the container.
	Privileged bool `json:"privileged,omitempty"`
	// The port bindings of the container.
	PortBindings map[Port][]PortBinding `json:"portBindings,omitempty"`
	// The custom DNS servers.
	DNS []string `json:"dns,omitempty"`
	// The custom DNS search domains.
	DNSSearch []string `json:"dnsSearch,omitempty"`
	// The network mode for the container.
	NetworkMode string `json:"networkMode,omitempty"`
	// The ipc mode for the container.
	IPCMode string `json:"ipcMode,omitempty"`
}

// Port represents the port number and the protocol, in the form <number>/<protocol>. For example: 80/tcp.
type Port string

// Port returns the number of the port.
func (p Port) Port() string {
	return strings.Split(string(p), "/")[0]
}

// Proto returns the name of the protocol.
func (p Port) Proto() string {
	parts := strings.Split(string(p), "/")
	if len(parts) == 1 {
		return "tcp"
	}
	return parts[1]
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
	Name         string            `json:"name,omitempty"`
	Command      []string          `json:"cmd,omitempty"`
	Env          []string          `json:"env,omitempty"`
	ExposedPorts map[Port]struct{} `json:"exposedPorts,omitempty"`
	Hostname     string            `json:"hostname,omitempty"`
	Image        string            `json:"image,omitempty"`
	Memory       int64             `json:"memory,omitempty"`
	CPUShares    int64             `json:"cpuShares,omitempty"`
	WorkingDir   string            `json:"workingDir,omitempty`
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
