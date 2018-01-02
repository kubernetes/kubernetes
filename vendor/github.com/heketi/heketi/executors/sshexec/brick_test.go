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
	"strings"
	"testing"

	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func TestSshExecBrickCreate(t *testing.T) {

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
			Fstab: "/my/fstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Create a Brick
	b := &executors.BrickRequest{
		VgId:             "xvgid",
		Name:             "id",
		TpSize:           100,
		Size:             10,
		PoolMetadataSize: 5,
	}

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "myhost:100", host)
		tests.Assert(t, len(commands) == 6)

		for i, cmd := range commands {
			cmd = strings.Trim(cmd, " ")
			switch i {
			case 0:
				tests.Assert(t,
					cmd == "mkdir -p /var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 1:
				tests.Assert(t,
					cmd == "lvcreate --poolmetadatasize 5K "+
						"-c 256K -L 100K -T vg_xvgid/tp_id -V 10K -n brick_id", cmd)

			case 2:
				tests.Assert(t,
					cmd == "mkfs.xfs -i size=512 "+
						"-n size=8192 /dev/mapper/vg_xvgid-brick_id", cmd)

			case 3:
				tests.Assert(t,
					cmd == "echo \"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id "+
						"xfs rw,inode64,noatime,nouuid 1 2\" | "+
						"tee -a /my/fstab > /dev/null", cmd)

			case 4:
				tests.Assert(t,
					cmd == "mount -o rw,inode64,noatime,nouuid "+
						"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 5:
				tests.Assert(t,
					cmd == "mkdir "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id/brick", cmd)
			}
		}

		return nil, nil
	}

	// Create Brick
	_, err = s.BrickCreate("myhost", b)
	tests.Assert(t, err == nil, err)

}

func TestSshExecBrickCreateWithGid(t *testing.T) {

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
			Fstab: "/my/fstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Create a Brick
	b := &executors.BrickRequest{
		VgId:             "xvgid",
		Name:             "id",
		TpSize:           100,
		Size:             10,
		PoolMetadataSize: 5,
		Gid:              1234,
	}

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "myhost:100", host)
		tests.Assert(t, len(commands) == 8)

		for i, cmd := range commands {
			cmd = strings.Trim(cmd, " ")
			switch i {
			case 0:
				tests.Assert(t,
					cmd == "mkdir -p /var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 1:
				tests.Assert(t,
					cmd == "lvcreate --poolmetadatasize 5K "+
						"-c 256K -L 100K -T vg_xvgid/tp_id -V 10K -n brick_id", cmd)

			case 2:
				tests.Assert(t,
					cmd == "mkfs.xfs -i size=512 "+
						"-n size=8192 /dev/mapper/vg_xvgid-brick_id", cmd)

			case 3:
				tests.Assert(t,
					cmd == "echo \"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id "+
						"xfs rw,inode64,noatime,nouuid 1 2\" | "+
						"tee -a /my/fstab > /dev/null", cmd)

			case 4:
				tests.Assert(t,
					cmd == "mount -o rw,inode64,noatime,nouuid "+
						"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 5:
				tests.Assert(t,
					cmd == "mkdir "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id/brick", cmd)

			case 6:
				tests.Assert(t,
					cmd == "chown :1234 "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id/brick", cmd)

			case 7:
				tests.Assert(t,
					cmd == "chmod 2775 "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id/brick", cmd)
			}
		}

		return nil, nil
	}

	// Create Brick
	_, err = s.BrickCreate("myhost", b)
	tests.Assert(t, err == nil, err)

}

func TestSshExecBrickCreateSudo(t *testing.T) {

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
			Fstab: "/my/fstab",
			Sudo:  true,
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Create a Brick
	b := &executors.BrickRequest{
		VgId:             "xvgid",
		Name:             "id",
		TpSize:           100,
		Size:             10,
		PoolMetadataSize: 5,
	}

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "myhost:100", host)
		tests.Assert(t, len(commands) == 6)
		tests.Assert(t, useSudo == true)

		for i, cmd := range commands {
			cmd = strings.Trim(cmd, " ")
			switch i {
			case 0:
				tests.Assert(t,
					cmd == "mkdir -p /var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 1:
				tests.Assert(t,
					cmd == "lvcreate --poolmetadatasize 5K "+
						"-c 256K -L 100K -T vg_xvgid/tp_id -V 10K -n brick_id", cmd)

			case 2:
				tests.Assert(t,
					cmd == "mkfs.xfs -i size=512 "+
						"-n size=8192 /dev/mapper/vg_xvgid-brick_id", cmd)

			case 3:
				tests.Assert(t,
					cmd == "echo \"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id "+
						"xfs rw,inode64,noatime,nouuid 1 2\" | "+
						"tee -a /my/fstab > /dev/null", cmd)

			case 4:
				tests.Assert(t,
					cmd == "mount -o rw,inode64,noatime,nouuid "+
						"/dev/mapper/vg_xvgid-brick_id "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case 5:
				tests.Assert(t,
					cmd == "mkdir "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id/brick", cmd)
			}
		}

		return nil, nil
	}

	// Create Brick
	_, err = s.BrickCreate("myhost", b)
	tests.Assert(t, err == nil, err)

}

func TestSshExecBrickDestroy(t *testing.T) {

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
			Fstab: "/my/fstab",
		},
	}

	s, err := NewSshExecutor(config)
	tests.Assert(t, err == nil)
	tests.Assert(t, s != nil)

	// Create a Brick
	b := &executors.BrickRequest{
		VgId:             "xvgid",
		Name:             "id",
		TpSize:           100,
		Size:             10,
		PoolMetadataSize: 5,
	}

	// Mock ssh function
	f.FakeConnectAndExec = func(host string,
		commands []string,
		timeoutMinutes int,
		useSudo bool) ([]string, error) {

		tests.Assert(t, host == "myhost:100", host)

		for _, cmd := range commands {
			cmd = strings.Trim(cmd, " ")
			switch {
			case strings.Contains(cmd, "umount"):
				tests.Assert(t,
					cmd == "umount "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case strings.Contains(cmd, "lvremove"):
				tests.Assert(t,
					cmd == "lvremove -f vg_xvgid/tp_id", cmd)

			case strings.Contains(cmd, "rmdir"):
				tests.Assert(t,
					cmd == "rmdir "+
						"/var/lib/heketi/mounts/vg_xvgid/brick_id", cmd)

			case strings.Contains(cmd, "sed"):
				tests.Assert(t,
					cmd == "sed -i.save "+
						"\"/brick_id/d\" /my/fstab", cmd)
			}
		}

		return nil, nil
	}

	// Create Brick
	err = s.BrickDestroy("myhost", b)
	tests.Assert(t, err == nil, err)
}
