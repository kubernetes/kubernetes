//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package sshexec

type CLICommandConfig struct {
	Fstab         string `json:"fstab"`
	Sudo          bool   `json:"sudo"`
	SnapShotLimit int    `json:"snapshot_limit"`

	// Experimental Settings
	RebalanceOnExpansion bool `json:"rebalance_on_expansion"`
}

type SshConfig struct {
	CLICommandConfig
	PrivateKeyFile string `json:"keyfile"`
	User           string `json:"user"`
	Port           string `json:"port"`
}
