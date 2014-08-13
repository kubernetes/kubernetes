// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"time"
)

type Container struct {
	ID      string    `yaml:"ID,omitempty" json:"ID,omitempty"`
	Created time.Time `yaml:"Created,omitempty" json:"Created,omitempty"`
	Path    string    `yaml:"Path,omitempty" json:"Path,omitempty"`
	Args    []string  `yaml:"Args,omitempty" json:"Args,omitempty"`
	Config  *Config   `yaml:"Config,omitempty" json:"Config,omitempty"`
	State   State     `yaml:"State,omitempty" json:"State,omitempty"`
	Image   string    `yaml:"Image,omitempty" json:"Image,omitempty"`

	NetworkSettings *NetworkSettings `yaml:"NetworkSettings,omitempty" json:"NetworkSettings,omitempty"`

	SysInitPath    string `yaml:"SysInitPath,omitempty" json:"SysInitPath,omitempty"`
	ResolvConfPath string `yaml:"ResolvConfPath,omitempty" json:"ResolvConfPath,omitempty"`
	HostnamePath   string `yaml:"HostnamePath,omitempty" json:"HostnamePath,omitempty"`
	HostsPath      string `yaml:"HostsPath,omitempty" json:"HostsPath,omitempty"`
	Name           string `yaml:"Name,omitempty" json:"Name,omitempty"`
	Driver         string `yaml:"Driver,omitempty" json:"Driver,omitempty"`

	Volumes    map[string]string `yaml:"Volumes,omitempty" json:"Volumes,omitempty"`
	VolumesRW  map[string]bool   `yaml:"VolumesRW,omitempty" json:"VolumesRW,omitempty"`
	HostConfig *HostConfig       `yaml:"HostConfig,omitempty" json:"HostConfig,omitempty"`
}

type Config struct {
	Hostname        string              `yaml:"Hostname,omitempty" json:"Hostname,omitempty"`
	Domainname      string              `yaml:"Domainname,omitempty" json:"Domainname,omitempty"`
	User            string              `yaml:"User,omitempty" json:"User,omitempty"`
	Memory          int64               `yaml:"Memory,omitempty" json:"Memory,omitempty"`
	MemorySwap      int64               `yaml:"MemorySwap,omitempty" json:"MemorySwap,omitempty"`
	CpuShares       int64               `yaml:"CpuShares,omitempty" json:"CpuShares,omitempty"`
	AttachStdin     bool                `yaml:"AttachStdin,omitempty" json:"AttachStdin,omitempty"`
	AttachStdout    bool                `yaml:"AttachStdout,omitempty" json:"AttachStdout,omitempty"`
	AttachStderr    bool                `yaml:"AttachStderr,omitempty" json:"AttachStderr,omitempty"`
	PortSpecs       []string            `yaml:"PortSpecs,omitempty" json:"PortSpecs,omitempty"`
	ExposedPorts    map[Port]struct{}   `yaml:"ExposedPorts,omitempty" json:"ExposedPorts,omitempty"`
	Tty             bool                `yaml:"Tty,omitempty" json:"Tty,omitempty"`
	OpenStdin       bool                `yaml:"OpenStdin,omitempty" json:"OpenStdin,omitempty"`
	StdinOnce       bool                `yaml:"StdinOnce,omitempty" json:"StdinOnce,omitempty"`
	Env             []string            `yaml:"Env,omitempty" json:"Env,omitempty"`
	Cmd             []string            `yaml:"Cmd,omitempty" json:"Cmd,omitempty"`
	Dns             []string            `yaml:"Dns,omitempty" json:"Dns,omitempty"`
	Image           string              `yaml:"Image,omitempty" json:"Image,omitempty"`
	Volumes         map[string]struct{} `yaml:"Volumes,omitempty" json:"Volumes,omitempty"`
	VolumesFrom     string              `yaml:"VolumesFrom,omitempty" json:"VolumesFrom,omitempty"`
	WorkingDir      string              `yaml:"WorkingDir,omitempty" json:"WorkingDir,omitempty"`
	Entrypoint      []string            `yaml:"Entrypoint,omitempty" json:"Entrypoint,omitempty"`
	NetworkDisabled bool                `yaml:"NetworkDisabled,omitempty" json:"NetworkDisabled,omitempty"`
}

type State struct {
	Running    bool      `yaml:"Running,omitempty" json:"Running,omitempty"`
	Paused     bool      `yaml:"Paused,omitempty" json:"Paused,omitempty"`
	Pid        int       `yaml:"Pid,omitempty" json:"Pid,omitempty"`
	ExitCode   int       `yaml:"ExitCode,omitempty" json:"ExitCode,omitempty"`
	StartedAt  time.Time `yaml:"StartedAt,omitempty" json:"StartedAt,omitempty"`
	FinishedAt time.Time `yaml:"FinishedAt,omitempty" json:"FinishedAt,omitempty"`
}

type PortBinding struct {
	HostIp   string `yaml:"HostIp,omitempty" json:"HostIp,omitempty"`
	HostPort string `yaml:"HostPort,omitempty" json:"HostPort,omitempty"`
}

type PortMapping map[string]string

type NetworkSettings struct {
	IPAddress   string                 `yaml:"IPAddress,omitempty" json:"IPAddress,omitempty"`
	IPPrefixLen int                    `yaml:"IPPrefixLen,omitempty" json:"IPPrefixLen,omitempty"`
	Gateway     string                 `yaml:"Gateway,omitempty" json:"Gateway,omitempty"`
	Bridge      string                 `yaml:"Bridge,omitempty" json:"Bridge,omitempty"`
	PortMapping map[string]PortMapping `yaml:"PortMapping,omitempty" json:"PortMapping,omitempty"`
	Ports       map[Port][]PortBinding `yaml:"Ports,omitempty" json:"Ports,omitempty"`
}

type KeyValuePair struct {
	Key   string `yaml:"Key,omitempty" json:"Key,omitempty"`
	Value string `yaml:"Value,omitempty" json:"Value,omitempty"`
}

type Port string

type HostConfig struct {
	Binds           []string               `yaml:"Binds,omitempty" json:"Binds,omitempty"`
	ContainerIDFile string                 `yaml:"ContainerIDFile,omitempty" json:"ContainerIDFile,omitempty"`
	LxcConf         []KeyValuePair         `yaml:"LxcConf,omitempty" json:"LxcConf,omitempty"`
	Privileged      bool                   `yaml:"Privileged,omitempty" json:"Privileged,omitempty"`
	PortBindings    map[Port][]PortBinding `yaml:"PortBindings,omitempty" json:"PortBindings,omitempty"`
	Links           []string               `yaml:"Links,omitempty" json:"Links,omitempty"`
	PublishAllPorts bool                   `yaml:"PublishAllPorts,omitempty" json:"PublishAllPorts,omitempty"`
	Dns             []string               `yaml:"Dns,omitempty" json:"Dns,omitempty"`
	DnsSearch       []string               `yaml:"DnsSearch,omitempty" json:"DnsSearch,omitempty"`
	VolumesFrom     []string               `yaml:"VolumesFrom,omitempty" json:"VolumesFrom,omitempty"`
	NetworkMode     string                 `yaml:"NetworkMode,omitempty" json:"NetworkMode,omitempty"`
}
