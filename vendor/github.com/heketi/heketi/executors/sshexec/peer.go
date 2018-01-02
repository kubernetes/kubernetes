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
	"fmt"

	"github.com/lpabon/godbc"
)

// :TODO: Rename this function to NodeInit or something
func (s *SshExecutor) PeerProbe(host, newnode string) error {

	godbc.Require(host != "")
	godbc.Require(newnode != "")

	logger.Info("Probing: %v -> %v", host, newnode)
	// create the commands
	commands := []string{
		fmt.Sprintf("gluster peer probe %v", newnode),
	}
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		return err
	}

	// Determine if there is a snapshot limit configuration setting
	if s.RemoteExecutor.SnapShotLimit() > 0 {
		logger.Info("Setting snapshot limit")
		commands = []string{
			fmt.Sprintf("gluster --mode=script snapshot config snap-max-hard-limit %v",
				s.RemoteExecutor.SnapShotLimit()),
		}
		_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
		if err != nil {
			return err
		}
	}

	return nil
}

func (s *SshExecutor) PeerDetach(host, detachnode string) error {
	godbc.Require(host != "")
	godbc.Require(detachnode != "")

	// create the commands
	logger.Info("Detaching node %v", detachnode)
	commands := []string{
		fmt.Sprintf("gluster peer detach %v", detachnode),
	}
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		logger.Err(err)
	}

	return nil
}

func (s *SshExecutor) GlusterdCheck(host string) error {
	godbc.Require(host != "")

	logger.Info("Check Glusterd service status in node %v", host)
	commands := []string{
		fmt.Sprintf("systemctl status glusterd"),
	}
	_, err := s.RemoteExecutor.RemoteCommandExecute(host, commands, 10)
	if err != nil {
		logger.Err(err)
		return err
	}

	return nil
}
