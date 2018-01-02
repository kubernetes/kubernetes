//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package kubeexec

import (
	"github.com/heketi/heketi/executors/sshexec"
)

type KubeConfig struct {
	sshexec.CLICommandConfig
	Namespace        string `json:"namespace"`
	GlusterDaemonSet bool   `json:"gluster_daemonset"`

	// Use POD name instead of using label
	// to access POD
	UsePodNames bool `json:"use_pod_names"`
}
