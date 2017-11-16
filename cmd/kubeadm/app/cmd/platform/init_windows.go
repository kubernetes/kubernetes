/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package platform

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/user"
	"strings"
)

// InitCopyCredentialsForUser will copy the cluster admin credentials to the home path of a non-root user
func InitCopyCredentialsForUser(copyCredentialsForUser string, adminKubeConfigPath string) error {
	var err error
	var usr *user.User
	var src, dest *os.File
	var kubeDir, configFilePath string

	usr, err = user.Lookup(copyCredentialsForUser)
	if err != nil {
		return fmt.Errorf(InitErrorNoSuchUser, copyCredentialsForUser, err)
	}
	// check if the given user is an admin.
	// if the list of users returned by `net localgroup administrators`
	// contains the username then the user is an administrator.
	// the list is separated by \r\n, also user names are case insensitive on Windows.
	out, err := exec.Command("net", "localgroup", "administrators").Output()
	if err != nil {
		return fmt.Errorf("failed to run 'net' command: %v", err)
	}
	outLower := strings.ToLower(string(out))
	if strings.Index(outLower, "\r\n"+strings.ToLower(copyCredentialsForUser)+"\r\n") != -1 {
		return fmt.Errorf(InitErrorUserIsRoot, copyCredentialsForUser)
	}

	// User.HomeDir don't work on Windows
	usr.HomeDir = ""
	out, err = exec.Command("cmd", "/c", "reg", "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\ProfileList\\"+usr.Uid, "/v", "ProfileImagePath").Output()
	if err != nil {
		return fmt.Errorf("no registry key for the home path of user %q: %v", copyCredentialsForUser, err)
	}
	usr.HomeDir = strings.Fields(string(out))[4]
	if usr.HomeDir == "" {
		return fmt.Errorf(InitErrorNoHomeDir, copyCredentialsForUser)
	}
	kubeDir = usr.HomeDir + InitPathKube
	configFilePath = kubeDir + InitPathConfig

	fmt.Printf(InitMessageCopyingCredentials, configFilePath, adminKubeConfigPath)

	if err := os.MkdirAll(kubeDir, 0700); err != nil {
		return fmt.Errorf(InitErrorCannotCreateDir, kubeDir, err)
	}

	// Chown() doesn't work on Windows. Attempt to use 'icacls' but if it's missing try 'cacls'.
	if _, err := exec.Command("icacls", kubeDir, "/e", "/g", copyCredentialsForUser+":f").Output(); err != nil {
		if _, err := exec.Command("cacls", kubeDir, "/e", "/g", copyCredentialsForUser+":f").Output(); err != nil {
			return fmt.Errorf(InitErrorCannotChown, kubeDir, err)
		}
	}

	src, err = os.Open(adminKubeConfigPath)
	if err != nil {
		return fmt.Errorf(InitErrorCannotOpenFileForReading, adminKubeConfigPath, err)
	}
	defer src.Close()

	dest, err = os.OpenFile(configFilePath, os.O_RDWR|os.O_CREATE, 0700)
	if err != nil {
		return fmt.Errorf(InitErrorCannotOpenFileForWriting, configFilePath, err)
	}
	defer dest.Close()

	if _, err := io.Copy(dest, src); err != nil {
		return fmt.Errorf(InitErrorCannotIoCopy, err)
	}

	if _, err := exec.Command("icacls", configFilePath, "/e", "/g", copyCredentialsForUser+":f").Output(); err != nil {
		if _, err := exec.Command("cacls", configFilePath, "/e", "/g", copyCredentialsForUser+":f").Output(); err != nil {
			return fmt.Errorf(InitErrorCannotChown, kubeDir, err)
		}
	}

	return nil
}
