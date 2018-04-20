// +build !windows

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
	"os/user"
	"strconv"
)

// InitCopyCredentialsForUser will copy the cluster admin credentials to the home path of a non-root user
func InitCopyCredentialsForUser(copyCredentialsForUser string, adminKubeConfigPath string) error {

	var err error
	var uid, gid int
	var usr *user.User
	var src, dest *os.File
	var kubeDir, configFilePath string

	usr, err = user.Lookup(copyCredentialsForUser)
	if err != nil {
		return fmt.Errorf(InitErrorNoSuchUser, copyCredentialsForUser, err)
	}

	if usr.Gid == "0" {
		return fmt.Errorf(InitErrorUserIsRoot, copyCredentialsForUser)
	}

	if usr.HomeDir == "" {
		return fmt.Errorf(InitErrorNoHomeDir, copyCredentialsForUser)
	}
	kubeDir = usr.HomeDir + InitPathKube
	configFilePath = kubeDir + InitPathConfig

	fmt.Printf(InitMessageCopyingCredentials, configFilePath, adminKubeConfigPath)

	uid, err = strconv.Atoi(usr.Uid)
	if err != nil {
		return fmt.Errorf(InitErrorAtoiString, usr.Uid, err)
	}

	gid, err = strconv.Atoi(usr.Gid)
	if err != nil {
		return fmt.Errorf(InitErrorAtoiString, usr.Gid, err)
	}

	if err := os.MkdirAll(kubeDir, 0700); err != nil {
		return fmt.Errorf(InitErrorCannotCreateDir, kubeDir, err)
	}

	// Chown doesn't work on Windows
	if err := os.Chown(kubeDir, uid, gid); err != nil {
		return fmt.Errorf(InitErrorCannotChown, kubeDir, err)
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

	if err := os.Chown(configFilePath, uid, gid); err != nil {
		return fmt.Errorf(InitErrorCannotChown, configFilePath, err)
	}

	return nil
}
