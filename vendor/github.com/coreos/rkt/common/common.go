// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package common defines values shared by different parts
// of rkt (e.g. stage0 and stage1)
package common

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

const (
	sharedVolumesDir = "/sharedVolumes"
	stage1Dir        = "/stage1"
	stage2Dir        = "/opt/stage2"
	AppsInfoDir      = "/appsinfo"

	EnvLockFd                    = "RKT_LOCK_FD"
	EnvSELinuxContext            = "RKT_SELINUX_CONTEXT"
	EnvSELinuxMountContext       = "RKT_SELINUX_MOUNT_CONTEXT"
	Stage1TreeStoreIDFilename    = "stage1TreeStoreID"
	AppTreeStoreIDFilename       = "treeStoreID"
	OverlayPreparedFilename      = "overlay-prepared"
	PrivateUsersPreparedFilename = "private-users-prepared"

	PrepareLock = "prepareLock"

	MetadataServicePort    = 18112
	MetadataServiceRegSock = "/run/rkt/metadata-svc.sock"

	APIServiceListenAddr = "localhost:15441"

	DefaultLocalConfigDir  = "/etc/rkt"
	DefaultSystemConfigDir = "/usr/lib/rkt"
)

// Stage1ImagePath returns the path where the stage1 app image (unpacked ACI) is rooted,
// (i.e. where its contents are extracted during stage0).
func Stage1ImagePath(root string) string {
	return filepath.Join(root, stage1Dir)
}

// Stage1RootfsPath returns the path to the stage1 rootfs
func Stage1RootfsPath(root string) string {
	return filepath.Join(Stage1ImagePath(root), aci.RootfsDir)
}

// Stage1ManifestPath returns the path to the stage1's manifest file inside the expanded ACI.
func Stage1ManifestPath(root string) string {
	return filepath.Join(Stage1ImagePath(root), aci.ManifestFile)
}

// PodManifestPath returns the path in root to the Pod Manifest
func PodManifestPath(root string) string {
	return filepath.Join(root, "pod")
}

// AppsPath returns the path where the apps within a pod live.
func AppsPath(root string) string {
	return filepath.Join(Stage1RootfsPath(root), stage2Dir)
}

// AppPath returns the path to an app's rootfs.
func AppPath(root string, appName types.ACName) string {
	return filepath.Join(AppsPath(root), appName.String())
}

// AppRootfsPath returns the path to an app's rootfs.
func AppRootfsPath(root string, appName types.ACName) string {
	return filepath.Join(AppPath(root, appName), aci.RootfsDir)
}

// RelAppPath returns the path of an app relative to the stage1 chroot.
func RelAppPath(appName types.ACName) string {
	return filepath.Join(stage2Dir, appName.String())
}

// RelAppRootfsPath returns the path of an app's rootfs relative to the stage1 chroot.
func RelAppRootfsPath(appName types.ACName) string {
	return filepath.Join(RelAppPath(appName), aci.RootfsDir)
}

// ImageManifestPath returns the path to the app's manifest file of a pod.
func ImageManifestPath(root string, appName types.ACName) string {
	return filepath.Join(AppPath(root, appName), aci.ManifestFile)
}

// AppsInfoPath returns the path to the appsinfo directory of a pod.
func AppsInfoPath(root string) string {
	return filepath.Join(root, AppsInfoDir)
}

// AppInfoPath returns the path to the app's appsinfo directory of a pod.
func AppInfoPath(root string, appName types.ACName) string {
	return filepath.Join(AppsInfoPath(root), appName.String())
}

// AppTreeStoreIDPath returns the path to the app's treeStoreID file of a pod.
func AppTreeStoreIDPath(root string, appName types.ACName) string {
	return filepath.Join(AppInfoPath(root, appName), AppTreeStoreIDFilename)
}

// AppImageManifestPath returns the path to the app's ImageManifest file
func AppImageManifestPath(root string, appName types.ACName) string {
	return filepath.Join(AppInfoPath(root, appName), aci.ManifestFile)
}

// SharedVolumesPath returns the path to the shared (empty) volumes of a pod.
func SharedVolumesPath(root string) string {
	return filepath.Join(root, sharedVolumesDir)
}

// MetadataServicePublicURL returns the public URL used to host the metadata service
func MetadataServicePublicURL(ip net.IP, token string) string {
	return fmt.Sprintf("http://%v:%v/%v", ip, MetadataServicePort, token)
}

func GetRktLockFD() (int, error) {
	if v := os.Getenv(EnvLockFd); v != "" {
		fd, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return -1, err
		}
		return int(fd), nil
	}
	return -1, fmt.Errorf("%v env var is not set", EnvLockFd)
}

// SupportsOverlay returns whether the system supports overlay filesystem
func SupportsOverlay() bool {
	exec.Command("modprobe", "overlay").Run()

	f, err := os.Open("/proc/filesystems")
	if err != nil {
		fmt.Println("error opening /proc/filesystems")
		return false
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		if s.Text() == "nodev\toverlay" {
			return true
		}
	}
	return false
}

// SupportsUserNS returns whether the kernel has CONFIG_USER_NS set
func SupportsUserNS() bool {
	if _, err := os.Stat("/proc/self/uid_map"); err == nil {
		return true
	}

	return false
}

// NetList implements the flag.Value interface to allow specification of --net with and without values
// Example: --net="all,net1:k1=v1;k2=v2,net2:l1=w1"
type NetList struct {
	mapping map[string]string
}

func (l *NetList) String() string {
	return strings.Join(l.Strings(), ",")
}

func (l *NetList) Set(value string) error {
	if l.mapping == nil {
		l.mapping = make(map[string]string)
	}
	for _, s := range strings.Split(value, ",") {
		netArgsPair := strings.Split(s, ":")
		netName := netArgsPair[0]

		if netName == "" {
			return fmt.Errorf("netname must not be empty")
		}

		if _, duplicate := l.mapping[netName]; duplicate {
			return fmt.Errorf("found duplicate netname %q", netName)
		}

		switch {
		case len(netArgsPair) == 1:
			l.mapping[netName] = ""
		case len(netArgsPair) == 2:
			if netName == "all" ||
				netName == "host" {
				return fmt.Errorf("arguments are not supported by special netname %q", netName)
			}
			l.mapping[netName] = netArgsPair[1]
		case len(netArgsPair) > 2:
			return fmt.Errorf("network %q provided with invalid arguments: %v", netName, netArgsPair[1:])
		default:
			return fmt.Errorf("unexpected case when processing network %q", s)
		}
	}
	return nil
}

func (l *NetList) Type() string {
	return "netList"
}

func (l *NetList) Strings() []string {
	if len(l.mapping) == 0 {
		return []string{"default"}
	}

	var list []string
	for k, v := range l.mapping {
		if v == "" {
			list = append(list, k)
		} else {
			list = append(list, fmt.Sprintf("%s:%s", k, v))
		}
	}
	return list
}

func (l *NetList) StringsOnlyNames() []string {
	var list []string
	for k, _ := range l.mapping {
		list = append(list, k)
	}
	return list
}

// Check if host networking has been requested
func (l *NetList) Host() bool {
	return l.Specific("host")
}

// Check if 'none' (loopback only) networking has been requested
func (l *NetList) None() bool {
	return l.Specific("none")
}

// Check if the container needs to be put in a separate network namespace
func (l *NetList) Contained() bool {
	return !l.Host() && len(l.mapping) > 0
}

func (l *NetList) Specific(net string) bool {
	_, exists := l.mapping[net]
	return exists
}

func (l *NetList) SpecificArgs(net string) string {
	return l.mapping[net]
}

func (l *NetList) All() bool {
	return l.Specific("all")
}

// LookupPath search for bin in paths. If found, it returns its absolute path,
// if not, an error
func LookupPath(bin string, paths string) (string, error) {
	pathsArr := filepath.SplitList(paths)
	for _, path := range pathsArr {
		binPath := filepath.Join(path, bin)
		binAbsPath, err := filepath.Abs(binPath)
		if err != nil {
			return "", fmt.Errorf("unable to find absolute path for %s", binPath)
		}
		d, err := os.Stat(binAbsPath)
		if err != nil {
			continue
		}
		// Check the executable bit, inspired by os.exec.LookPath()
		if m := d.Mode(); !m.IsDir() && m&0111 != 0 {
			return binAbsPath, nil
		}
	}
	return "", fmt.Errorf("unable to find %q in %q", bin, paths)
}

// SystemdVersion parses and returns the version of a given systemd binary
func SystemdVersion(systemdBinaryPath string) (int, error) {
	versionBytes, err := exec.Command(systemdBinaryPath, "--version").CombinedOutput()
	if err != nil {
		return -1, errwrap.Wrap(fmt.Errorf("unable to probe %s version", systemdBinaryPath), err)
	}
	versionStr := strings.SplitN(string(versionBytes), "\n", 2)[0]
	var version int
	n, err := fmt.Sscanf(versionStr, "systemd %d", &version)
	if err != nil || n != 1 {
		return -1, fmt.Errorf("cannot parse version: %q", versionStr)
	}

	return version, nil
}
