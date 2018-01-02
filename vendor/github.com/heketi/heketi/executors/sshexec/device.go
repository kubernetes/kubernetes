//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package sshexec

import (
	"errors"
	"fmt"
	"github.com/heketi/heketi/executors"
	"strconv"
	"strings"
)

const (
	VGDISPLAY_SIZE_KB                  = 11
	VGDISPLAY_PHYSICAL_EXTENT_SIZE     = 12
	VGDISPLAY_TOTAL_NUMBER_EXTENTS     = 13
	VGDISPLAY_ALLOCATED_NUMBER_EXTENTS = 14
	VGDISPLAY_FREE_NUMBER_EXTENTS      = 15
)

// Read:
// https://access.redhat.com/documentation/en-US/Red_Hat_Storage/3.1/html/Administration_Guide/Brick_Configuration.html
//

func (s *SshExecutor) DeviceSetup(host, device, vgid string) (d *executors.DeviceInfo, e error) {

	// Setup commands
	commands := []string{
		fmt.Sprintf("pvcreate --metadatasize=128M --dataalignment=256K %v", device),
		fmt.Sprintf("vgcreate %v %v", s.vgName(vgid), device),
	}

	// Execute command
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 5)
	if err != nil {
		return nil, err
	}

	// Create a cleanup function if anything fails
	defer func() {
		if e != nil {
			s.DeviceTeardown(host, device, vgid)
		}
	}()

	// Vg info
	d = &executors.DeviceInfo{}
	err = s.getVgSizeFromNode(d, host, device, vgid)
	if err != nil {
		return nil, err
	}

	return d, nil
}

func (s *SshExecutor) DeviceTeardown(host, device, vgid string) error {

	// Setup commands
	commands := []string{
		fmt.Sprintf("vgremove %v", s.vgName(vgid)),
		fmt.Sprintf("pvremove %v", device),
	}

	// Execute command
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 5)
	if err != nil {
		logger.LogError("Error while deleting device %v on %v with id %v",
			device, host, vgid)
	}

	commands = []string{
		fmt.Sprintf("ls %v/%v", rootMountPoint, s.vgName(vgid)),
	}
	_, err = s.RemoteExecutor.RemoteCommandExecute(host, commands, 5)
	if err != nil {
		return nil
	}

	commands = []string{
		fmt.Sprintf("rmdir %v/%v", rootMountPoint, s.vgName(vgid)),
	}

	_, err = s.RemoteExecutor.RemoteCommandExecute(host, commands, 5)
	if err != nil {
		logger.LogError("Error while removing the VG directory")
		return nil
	}

	return nil
}

func (s *SshExecutor) getVgSizeFromNode(
	d *executors.DeviceInfo,
	host, device, vgid string) error {

	// Setup command
	commands := []string{
		fmt.Sprintf("vgdisplay -c %v", s.vgName(vgid)),
	}

	// Execute command
	b, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 5)
	if err != nil {
		return err
	}

	// Example:
	// sampleVg:r/w:772:-1:0:0:0:-1:0:4:4:2097135616:4096:511996:0:511996:rJ0bIG-3XNc-NoS0-fkKm-batK-dFyX-xbxHym
	vginfo := strings.Split(b[0], ":")

	// See vgdisplay manpage
	if len(vginfo) < 17 {
		return errors.New("vgdisplay returned an invalid string")
	}

	extent_size, err :=
		strconv.ParseUint(vginfo[VGDISPLAY_PHYSICAL_EXTENT_SIZE], 10, 64)
	if err != nil {
		return err
	}

	free_extents, err :=
		strconv.ParseUint(vginfo[VGDISPLAY_FREE_NUMBER_EXTENTS], 10, 64)
	if err != nil {
		return err
	}

	d.Size = free_extents * extent_size
	d.ExtentSize = extent_size
	logger.Debug("Size of %v in %v is %v", device, host, d.Size)
	return nil
}
