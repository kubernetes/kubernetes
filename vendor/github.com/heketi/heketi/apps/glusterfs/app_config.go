//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"encoding/json"
	"io"
	"os"

	"github.com/heketi/heketi/executors/kubeexec"
	"github.com/heketi/heketi/executors/sshexec"
)

type GlusterFSConfig struct {
	DBfile     string              `json:"db"`
	Executor   string              `json:"executor"`
	Allocator  string              `json:"allocator"`
	SshConfig  sshexec.SshConfig   `json:"sshexec"`
	KubeConfig kubeexec.KubeConfig `json:"kubeexec"`
	Loglevel   string              `json:"loglevel"`

	// advanced settings
	BrickMaxSize int `json:"brick_max_size_gb"`
	BrickMinSize int `json:"brick_min_size_gb"`
	BrickMaxNum  int `json:"max_bricks_per_volume"`
}

type ConfigFile struct {
	GlusterFS GlusterFSConfig `json:"glusterfs"`
}

func loadConfiguration(configIo io.Reader) *GlusterFSConfig {
	configParser := json.NewDecoder(configIo)

	var config ConfigFile
	if err := configParser.Decode(&config); err != nil {
		logger.LogError("Unable to parse config file: %v\n",
			err.Error())
		return nil
	}

	// Set environment variable to override configuration file
	env := os.Getenv("HEKETI_EXECUTOR")
	if env != "" {
		config.GlusterFS.Executor = env
	}

	return &config.GlusterFS
}
