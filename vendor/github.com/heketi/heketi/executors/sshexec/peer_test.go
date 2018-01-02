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
	"testing"

	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func TestSshExecPeerProbe(t *testing.T) {

	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		CLICommandConfig: CLICommandConfig{
			Fstab: "/my/fstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "host:22", host)
		tests.Assert(t, len(commands) == 1)
		tests.Assert(t, commands[0] == "gluster peer probe newnode", commands)

		return nil, nil
	}

	// Call function
	err = s.PeerProbe("host", "newnode")
	tests.Assert(t, err == nil, err)

	// Now set the snapshot limit
	config = &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		CLICommandConfig: CLICommandConfig{
			Fstab:         "/my/fstab",
			SnapShotLimit: 14,
		},
	}

	s, err = NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Mock ssh function
	count := 0
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		switch count {
		case 0:
			tests.Assert(t, host == "host:22", host)
			tests.Assert(t, len(commands) == 1)
			tests.Assert(t, commands[0] == "gluster peer probe newnode", commands)

		case 1:
			tests.Assert(t, host == "host:22", host)
			tests.Assert(t, len(commands) == 1)
			tests.Assert(t, commands[0] == "gluster --mode=script snapshot config snap-max-hard-limit 14", commands)

		default:
			tests.Assert(t, false, "Should not be reached")
		}
		count++

		return nil, nil
	}

	// Call function
	err = s.PeerProbe("host", "newnode")
	tests.Assert(t, err == nil, err)
	tests.Assert(t, count == 2)

}

func TestSshExecGlusterdCheck(t *testing.T) {
	f := NewFakeSsh()
	defer tests.Patch(&sshNew,
		func(logger *utils.Logger, user string, file string) (Ssher, error) {
			return f, nil
		}).Restore()

	config := &SshConfig{
		PrivateKeyFile: "xkeyfile",
		User:           "xuser",
		CLICommandConfig: CLICommandConfig{
			Fstab: "/my/fstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "newhost:22", host)
		tests.Assert(t, len(commands) == 1)
		tests.Assert(t, commands[0] == "systemctl status glusterd", commands)

		return nil, nil
	}

	// Call function
	err = s.GlusterdCheck("newhost")
	tests.Assert(t, err == nil, err)
}
