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
	"os"
	"testing"

	restclient "k8s.io/client-go/rest"

	"github.com/heketi/heketi/executors/sshexec"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func init() {
	inClusterConfig = func() (*restclient.Config, error) {
		return &restclient.Config{}, nil
	}
	logger.SetLevel(utils.LEVEL_NOLOG)
}

func TestNewKubeExecutor(t *testing.T) {
	config := &KubeConfig{
		CLICommandConfig: sshexec.CLICommandConfig{
			Fstab: "myfstab",
		},
		Namespace: "mynamespace",
	}

	k, err := NewKubeExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, k.Fstab == "myfstab")
	tests.Assert(t, k.Throttlemap != nil)
	tests.Assert(t, k.config != nil)
}

func TestNewKubeExecutorNoNamespace(t *testing.T) {
	config := &KubeConfig{
		CLICommandConfig: sshexec.CLICommandConfig{
			Fstab: "myfstab",
		},
	}

	k, err := NewKubeExecutor(config)
	tests.Assert(t, err != nil)
	tests.Assert(t, k == nil)
}

func TestNewKubeExecutorRebalanceOnExpansion(t *testing.T) {

	// This tests access to configurations
	// from the sshconfig exector

	config := &KubeConfig{
		CLICommandConfig: sshexec.CLICommandConfig{
			Fstab: "myfstab",
		},
		Namespace: "mynamespace",
	}

	k, err := NewKubeExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, k.Fstab == "myfstab")
	tests.Assert(t, k.Throttlemap != nil)
	tests.Assert(t, k.config != nil)
	tests.Assert(t, k.RebalanceOnExpansion() == false)

	config = &KubeConfig{
		CLICommandConfig: sshexec.CLICommandConfig{
			Fstab:                "myfstab",
			RebalanceOnExpansion: true,
		},
		Namespace: "mynamespace",
	}

	k, err = NewKubeExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, k.Fstab == "myfstab")
	tests.Assert(t, k.Throttlemap != nil)
	tests.Assert(t, k.config != nil)
	tests.Assert(t, k.RebalanceOnExpansion() == true)
}

func TestKubeExecutorEnvVariables(t *testing.T) {

	// set environment
	err := os.Setenv("HEKETI_SNAPSHOT_LIMIT", "999")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_SNAPSHOT_LIMIT")

	err = os.Setenv("HEKETI_FSTAB", "anotherfstab")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_FSTAB")

	config := &KubeConfig{
		CLICommandConfig: sshexec.CLICommandConfig{
			Fstab: "myfstab",
		},
		Namespace: "mynamespace",
	}

	k, err := NewKubeExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, k.Throttlemap != nil)
	tests.Assert(t, k.config != nil)
	tests.Assert(t, k.Fstab == "anotherfstab")
	tests.Assert(t, k.SnapShotLimit() == 999)

}
