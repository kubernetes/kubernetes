//
// Copyright (c) 2014 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package ssh

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"os"
	"time"

	"github.com/heketi/heketi/pkg/utils"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/agent"
)

type SshExec struct {
	clientConfig *ssh.ClientConfig
	logger       *utils.Logger
}

func getKeyFile(file string) (key ssh.Signer, err error) {
	buf, err := ioutil.ReadFile(file)
	if err != nil {
		return
	}
	key, err = ssh.ParsePrivateKey(buf)
	if err != nil {
		fmt.Print(err)
		return
	}
	return
}

func NewSshExecWithAuth(logger *utils.Logger, user string) *SshExec {

	sshexec := &SshExec{}
	sshexec.logger = logger

	authSocket := os.Getenv("SSH_AUTH_SOCK")
	if authSocket == "" {
		log.Fatal("SSH_AUTH_SOCK required, check that your ssh agent is running")
		return nil
	}

	agentUnixSock, err := net.Dial("unix", authSocket)
	if err != nil {
		log.Fatal(err)
		return nil
	}

	agent := agent.NewClient(agentUnixSock)
	signers, err := agent.Signers()
	if err != nil {
		log.Fatal(err)
		return nil
	}

	sshexec.clientConfig = &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signers...)},
	}

	return sshexec
}

func NewSshExecWithKeyFile(logger *utils.Logger, user string, file string) *SshExec {

	var key ssh.Signer
	var err error

	sshexec := &SshExec{}
	sshexec.logger = logger

	// Now in the main function DO:
	if key, err = getKeyFile(file); err != nil {
		fmt.Println("Unable to get keyfile")
		return nil
	}
	// Define the Client Config as :
	sshexec.clientConfig = &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(key),
		},
	}

	return sshexec
}

// This function requires the password string to be crypt encrypted
func NewSshExecWithPassword(logger *utils.Logger, user string, password string) *SshExec {

	sshexec := &SshExec{}
	sshexec.logger = logger

	// Define the Client Config as :
	sshexec.clientConfig = &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{ssh.Password(password)},
	}

	return sshexec
}

// This function was based from https://github.com/coreos/etcd-manager/blob/master/main.go
func (s *SshExec) ConnectAndExec(host string, commands []string, timeoutMinutes int, useSudo bool) ([]string, error) {

	buffers := make([]string, len(commands))

	// :TODO: Will need a timeout here in case the server does not respond
	client, err := ssh.Dial("tcp", host, s.clientConfig)
	if err != nil {
		s.logger.Warning("Failed to create SSH connection to %v: %v", host, err)
		return nil, err
	}
	defer client.Close()

	// Execute each command
	for index, command := range commands {

		session, err := client.NewSession()
		if err != nil {
			s.logger.LogError("Unable to create SSH session: %v", err)
			return nil, err
		}
		defer session.Close()

		// Create a buffer to trap session output
		var b bytes.Buffer
		var berr bytes.Buffer
		session.Stdout = &b
		session.Stderr = &berr

		// Execute command in a shell
		command = "/bin/bash -c '" + command + "'"

		// Check if we need to use sudo for the entire command
		if useSudo {
			command = "sudo " + command
		}

		// Execute command
		err = session.Start(command)
		if err != nil {
			return nil, err
		}

		// Spawn function to wait for results
		errch := make(chan error)
		go func() {
			errch <- session.Wait()
		}()

		// Set the timeout
		timeout := time.After(time.Minute * time.Duration(timeoutMinutes))

		// Wait for either the command completion or timeout
		select {
		case err := <-errch:
			if err != nil {
				s.logger.LogError("Failed to run command [%v] on %v: Err[%v]: Stdout [%v]: Stderr [%v]",
					command, host, err, b.String(), berr.String())
				return nil, fmt.Errorf("%s", berr.String())
			}
			s.logger.Debug("Host: %v Command: %v\nResult: %v", host, command, b.String())
			buffers[index] = b.String()

		case <-timeout:
			s.logger.LogError("Timeout on command [%v] on %v: Err[%v]: Stdout [%v]: Stderr [%v]",
				command, host, err, b.String(), berr.String())
			err := session.Signal(ssh.SIGKILL)
			if err != nil {
				s.logger.LogError("Unable to send kill signal to command [%v] on host [%v]: %v",
					command, host, err)
			}
			return nil, errors.New("SSH command timeout")
		}
	}

	return buffers, nil
}
