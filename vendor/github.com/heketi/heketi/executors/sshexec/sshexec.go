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
	"os"
	"strconv"
	"sync"

	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/heketi/pkg/utils/ssh"
	"github.com/lpabon/godbc"
)

type RemoteCommandTransport interface {
	RemoteCommandExecute(host string, commands []string, timeoutMinutes int) ([]string, error)
	RebalanceOnExpansion() bool
	SnapShotLimit() int
}

type Ssher interface {
	ConnectAndExec(host string, commands []string, timeoutMinutes int, useSudo bool) ([]string, error)
}

type SshExecutor struct {
	// "Public"
	Throttlemap    map[string]chan bool
	Lock           sync.Mutex
	RemoteExecutor RemoteCommandTransport
	Fstab          string

	// Private
	private_keyfile string
	user            string
	exec            Ssher
	config          *SshConfig
	port            string
}

var (
	logger           = utils.NewLogger("[sshexec]", utils.LEVEL_DEBUG)
	ErrSshPrivateKey = errors.New("Unable to read private key file")
	sshNew           = func(logger *utils.Logger, user string, file string) (Ssher, error) {
		s := ssh.NewSshExecWithKeyFile(logger, user, file)
		if s == nil {
			return nil, ErrSshPrivateKey
		}
		return s, nil
	}
)

func setWithEnvVariables(config *SshConfig) {
	var env string

	env = os.Getenv("HEKETI_SSH_KEYFILE")
	if "" != env {
		config.PrivateKeyFile = env
	}

	env = os.Getenv("HEKETI_SSH_USER")
	if "" != env {
		config.User = env
	}

	env = os.Getenv("HEKETI_SSH_PORT")
	if "" != env {
		config.Port = env
	}

	env = os.Getenv("HEKETI_FSTAB")
	if "" != env {
		config.Fstab = env
	}

	env = os.Getenv("HEKETI_SNAPSHOT_LIMIT")
	if "" != env {
		i, err := strconv.Atoi(env)
		if err == nil {
			config.SnapShotLimit = i
		}
	}

}

func NewSshExecutor(config *SshConfig) (*SshExecutor, error) {
	// Override configuration
	setWithEnvVariables(config)

	s := &SshExecutor{}
	s.RemoteExecutor = s
	s.Throttlemap = make(map[string]chan bool)

	// Set configuration
	if config.PrivateKeyFile == "" {
		return nil, fmt.Errorf("Missing ssh private key file in configuration")
	}
	s.private_keyfile = config.PrivateKeyFile

	if config.User == "" {
		s.user = "heketi"
	} else {
		s.user = config.User
	}

	if config.Port == "" {
		s.port = "22"
	} else {
		s.port = config.Port
	}

	if config.Fstab == "" {
		s.Fstab = "/etc/fstab"
	} else {
		s.Fstab = config.Fstab
	}

	// Save the configuration
	s.config = config

	// Show experimental settings
	if s.config.RebalanceOnExpansion {
		logger.Warning("Rebalance on volume expansion has been enabled.  This is an EXPERIMENTAL feature")
	}

	// Setup key
	var err error
	s.exec, err = sshNew(logger, s.user, s.private_keyfile)
	if err != nil {
		logger.Err(err)
		return nil, err
	}

	godbc.Ensure(s != nil)
	godbc.Ensure(s.config == config)
	godbc.Ensure(s.user != "")
	godbc.Ensure(s.private_keyfile != "")
	godbc.Ensure(s.port != "")
	godbc.Ensure(s.Fstab != "")

	return s, nil
}

func (s *SshExecutor) SetLogLevel(level string) {
	switch level {
	case "none":
		logger.SetLevel(utils.LEVEL_NOLOG)
	case "critical":
		logger.SetLevel(utils.LEVEL_CRITICAL)
	case "error":
		logger.SetLevel(utils.LEVEL_ERROR)
	case "warning":
		logger.SetLevel(utils.LEVEL_WARNING)
	case "info":
		logger.SetLevel(utils.LEVEL_INFO)
	case "debug":
		logger.SetLevel(utils.LEVEL_DEBUG)
	}
}

func (s *SshExecutor) AccessConnection(host string) {

	var (
		c  chan bool
		ok bool
	)

	s.Lock.Lock()
	if c, ok = s.Throttlemap[host]; !ok {
		c = make(chan bool, 1)
		s.Throttlemap[host] = c
	}
	s.Lock.Unlock()

	c <- true
}

func (s *SshExecutor) FreeConnection(host string) {
	s.Lock.Lock()
	c := s.Throttlemap[host]
	s.Lock.Unlock()

	<-c
}

func (s *SshExecutor) RemoteCommandExecute(host string,
	commands []string,
	timeoutMinutes int) ([]string, error) {

	// Throttle
	s.AccessConnection(host)
	defer s.FreeConnection(host)

	// Execute
	return s.exec.ConnectAndExec(host+":"+s.port, commands, timeoutMinutes, s.config.Sudo)
}

func (s *SshExecutor) vgName(vgId string) string {
	return "vg_" + vgId
}

func (s *SshExecutor) brickName(brickId string) string {
	return "brick_" + brickId
}

func (s *SshExecutor) tpName(brickId string) string {
	return "tp_" + brickId
}

func (s *SshExecutor) RebalanceOnExpansion() bool {
	return s.config.RebalanceOnExpansion
}

func (s *SshExecutor) SnapShotLimit() int {
	return s.config.SnapShotLimit
}
