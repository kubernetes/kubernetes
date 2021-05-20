// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package users

import (
	"bytes"
	"io/fs"
	"os"
	"os/user"
	"path/filepath"
	"strconv"

	"github.com/pkg/errors"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/utils/exec"
)

var allUsers = []string{kubeadmconstants.KubeSchedulerUserName,
	kubeadmconstants.KubeControllerManagerUserName,
	kubeadmconstants.KubeAPIServerUserName,
	kubeadmconstants.EtcdUserName,
}

var allGroups = []string{kubeadmconstants.ServiceAccountKeyReadersGroupName}

// RemoveUsersAndGroups removes a all users and groups that kubeadm creates to
// run the control plane as non-root.
func RemoveUsersAndGroups() error {
	for _, user := range allUsers {
		_, found, err := getControlPlaneUserId(user)
		if err != nil {
			return err
		}
		if found {
			if err := removeUser(user); err != nil {
				return err
			}
		}
	}
	for _, group := range allGroups {
		_, found, err := getControlPlaneGroupId(group)
		if err != nil {
			return err
		}
		if found {
			if err := removeGroup(group); err != nil {
				return err
			}
		}
	}
	return nil
}

// GetUserAndGroupIDs returns a map of username/groupname to uid/gid for all the
// users and groups created when the kubeadm control-plane is running as
// non-root.
func GetUserAndGroupIDs() (map[string]int64, error) {
	result := make(map[string]int64)
	for _, user := range allUsers {
		uid, err := getOrCreateControlPlanUser(user)
		if err != nil {
			return nil, err
		}
		result[user] = uid
	}
	for _, group := range allGroups {
		gid, err := getOrCreateControlPlanGroup(group)
		if err != nil {
			return nil, err
		}
		result[group] = gid
	}
	return result, nil
}

// UpdateFileOwnership updates the ownership and permissions of the file.
func UpdateFileOwnership(filepath string, uid int64, gid int64, permissions uint32) error {
	if err := os.Chown(filepath, int(uid), int(gid)); err != nil {
		return err
	}
	if err := os.Chmod(filepath, fs.FileMode(permissions)); err != nil {
		return err
	}
	return nil
}

// UpdateDirectoryOwnership updates the ownership and permissions of all files
// and sub-directories within the directory.
func UpdateDirectoryOwnership(directorypath string, uid int64, gid int64, permissions uint32) error {
	err := filepath.WalkDir(directorypath, func(path string, d fs.DirEntry, err error) error {
		if err := os.Chown(path, int(uid), int(gid)); err != nil {
			return err
		}
		if err := os.Chmod(path, fs.FileMode(permissions)); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	return nil
}

func getControlPlaneUserId(username string) (int64, bool, error) {
	u, err := user.Lookup(username)
	if err != nil {
		if _, ok := err.(user.UnknownUserError); !ok {
			return 0, false, err
		} else {
			return 0, false, nil
		}
	}
	uid, err := strconv.ParseInt(u.Uid, 10, 64)
	if err != nil {
		return 0, false, err
	}
	return uid, true, nil
}

func getControlPlaneGroupId(groupname string) (int64, bool, error) {
	g, err := user.LookupGroup(groupname)
	if err != nil {
		if _, ok := err.(user.UnknownGroupError); !ok {
			return 0, false, err
		} else {
			return 0, false, nil
		}
	}
	gid, err := strconv.ParseInt(g.Gid, 10, 64)
	if err != nil {
		return 0, false, err
	}
	return gid, true, nil
}

func createControlPlaneGroup(groupname string) error {
	exec := exec.New()
	cmd := exec.Command("groupadd", "-r", groupname)
	buff := bytes.Buffer{}
	cmd.SetStderr(&buff)
	if err := cmd.Run(); err != nil {
		return errors.Wrapf(err, "failed to add group %s: %s", groupname, buff.String())
	}
	return nil
}

func createControlPlanUser(username string) error {
	exec := exec.New()
	cmd := exec.Command("useradd", "-r", "-s", "/bin/false", username)
	buff := bytes.Buffer{}
	cmd.SetStderr(&buff)
	if err := cmd.Run(); err != nil {
		return errors.Wrapf(err, "failed to add user %s: %s", username, buff.String())
	}
	return nil
}

func getOrCreateControlPlanGroup(groupname string) (int64, error) {
	gid, found, err := getControlPlaneGroupId(groupname)
	if err != nil {
		return 0, err
	}
	if found {
		return gid, nil
	}
	if err := createControlPlaneGroup(groupname); err != nil {
		return 0, err
	}
	gid, _, err = getControlPlaneGroupId(groupname)
	if err != nil {
		return 0, err
	}
	return gid, nil
}

func getOrCreateControlPlanUser(username string) (int64, error) {
	uid, found, err := getControlPlaneUserId(username)
	if err != nil {
		return 0, err
	}
	if found {
		return uid, nil
	}
	if err := createControlPlanUser(username); err != nil {
		return 0, err
	}
	uid, _, err = getControlPlaneUserId(username)
	if err != nil {
		return 0, err
	}
	return uid, nil
}

func removeUser(username string) error {
	exec := exec.New()
	cmd := exec.Command("userdel", username)
	buff := bytes.Buffer{}
	cmd.SetStderr(&buff)
	if err := cmd.Run(); err != nil {
		return errors.Wrapf(err, "failed to remove user %s : %s", username, buff.String())
	}
	return nil
}

func removeGroup(groupname string) error {
	exec := exec.New()
	cmd := exec.Command("groupdel", groupname)
	buff := bytes.Buffer{}
	cmd.SetStderr(&buff)
	if err := cmd.Run(); err != nil {
		return errors.Wrapf(err, "failed to remove group %s: %s", groupname, buff.String())
	}
	return nil
}
