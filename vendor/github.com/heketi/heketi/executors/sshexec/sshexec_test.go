//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package sshexec

import (
	"os"
	"testing"

	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

// Mock SSH calls
type FakeSsh struct {
	FakeConnectAndExec func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error)
}

func NewFakeSsh() *FakeSsh {
	f := &FakeSsh{}

	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {
		return []string{""}, nil
	}

	return f
}

func (f *FakeSsh) ConnectAndExec(host string,
	commands []string,
	timeoutMinutes int,
	useSudo bool) ([]string, error) {
	return f.FakeConnectAndExec(host, commands, timeoutMinutes, useSudo)

}

func TestNewSshExec(t *testing.T) {

	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		Port:           "100",
		CLICommandConfig: CLICommandConfig{
			Fstab: "xfstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)
	tests.Assert(t, s.private_keyfile == config.PrivateKeyFile)
	tests.Assert(t, s.user == config.User)
	tests.Assert(t, s.port == config.Port)
	tests.Assert(t, s.Fstab == config.Fstab)
	tests.Assert(t, s.exec != nil)
}

func TestSshExecRebalanceOnExpansion(t *testing.T) {

	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		Port:           "100",
		CLICommandConfig: CLICommandConfig{
			Fstab: "xfstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)
	tests.Assert(t, s.private_keyfile == config.PrivateKeyFile)
	tests.Assert(t, s.user == config.User)
	tests.Assert(t, s.port == config.Port)
	tests.Assert(t, s.Fstab == config.Fstab)
	tests.Assert(t, s.exec != nil)
	tests.Assert(t, s.RebalanceOnExpansion() == false)

	config = &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		Port:           "100",
		CLICommandConfig: CLICommandConfig{
			Fstab:                "xfstab",
			RebalanceOnExpansion: true,
		},
	}

	s, err = NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)
	tests.Assert(t, s.private_keyfile == config.PrivateKeyFile)
	tests.Assert(t, s.user == config.User)
	tests.Assert(t, s.port == config.Port)
	tests.Assert(t, s.Fstab == config.Fstab)
	tests.Assert(t, s.exec != nil)
	tests.Assert(t, s.RebalanceOnExpansion() == true)

}

func TestNewSshExecDefaults(t *testing.T) {
	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)
	tests.Assert(t, s.private_keyfile == "xkeyfile")
	tests.Assert(t, s.user == "heketi")
	tests.Assert(t, s.port == "22")
	tests.Assert(t, s.Fstab == "/etc/fstab")
	tests.Assert(t, s.exec != nil)

}

func TestNewSshExecBadPrivateKeyLocation(t *testing.T) {
	config := &SshConfig{}

	s, err := NewSshExecutor(config)
	tests.Assert(t, s == nil)
	tests.Assert(t, err != nil)
}

func TestSshExecutorEnvVariables(t *testing.T) {

	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	// set environment
	err := os.Setenv("HEKETI_SNAPSHOT_LIMIT", "999")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_SNAPSHOT_LIMIT")

	err = os.Setenv("HEKETI_FSTAB", "anotherfstab")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_FSTAB")

	err = os.Setenv("HEKETI_SSH_KEYFILE", "ykeyfile")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_SSH_KEYFILE")

	err = os.Setenv("HEKETI_SSH_USER", "yuser")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_SSH_USER")

	err = os.Setenv("HEKETI_SSH_PORT", "33")
	tests.Assert(t, err == nil)
	defer os.Unsetenv("HEKETI_SSH_PORT")

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		Port:           "100",
		CLICommandConfig: CLICommandConfig{
			Fstab: "xfstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)
	tests.Assert(t, s.Throttlemap != nil)
	tests.Assert(t, s.config != nil)
	tests.Assert(t, s.Fstab == "anotherfstab")
	tests.Assert(t, s.SnapShotLimit() == 999)
	tests.Assert(t, s.private_keyfile == "ykeyfile")
	tests.Assert(t, s.user == "yuser")
	tests.Assert(t, s.port == "33")
	tests.Assert(t, s.exec != nil)

}
