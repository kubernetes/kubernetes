// Copyright 2014 Google Inc. All Rights Reserved.
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

package docker

import (
	"flag"
	"fmt"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/blang/semver"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/devicemapper"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/machine"
	"github.com/google/cadvisor/manager/watcher"
	dockerutil "github.com/google/cadvisor/utils/docker"

	docker "github.com/docker/engine-api/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

var ArgDockerEndpoint = flag.String("docker", "unix:///var/run/docker.sock", "docker endpoint")

// The namespace under which Docker aliases are unique.
const DockerNamespace = "docker"

// Regexp that identifies docker cgroups, containers started with
// --cgroup-parent have another prefix than 'docker'
var dockerCgroupRegexp = regexp.MustCompile(`([a-z0-9]{64})`)

var dockerEnvWhitelist = flag.String("docker_env_metadata_whitelist", "", "a comma-separated list of environment variable keys that needs to be collected for docker containers")

var (
	// Basepath to all container specific information that libcontainer stores.
	dockerRootDir string

	dockerRootDirFlag = flag.String("docker_root", "/var/lib/docker", "DEPRECATED: docker root is read from docker info (this is a fallback, default: /var/lib/docker)")

	dockerRootDirOnce sync.Once

	// flag that controls globally disabling thin_ls pending future enhancements.
	// in production, it has been found that thin_ls makes excessive use of iops.
	// in an iops restricted environment, usage of thin_ls must be controlled via blkio.
	// pending that enhancement, disable its usage.
	disableThinLs = true
)

func RootDir() string {
	dockerRootDirOnce.Do(func() {
		status, err := Status()
		if err == nil && status.RootDir != "" {
			dockerRootDir = status.RootDir
		} else {
			dockerRootDir = *dockerRootDirFlag
		}
	})
	return dockerRootDir
}

type storageDriver string

const (
	devicemapperStorageDriver storageDriver = "devicemapper"
	aufsStorageDriver         storageDriver = "aufs"
	overlayStorageDriver      storageDriver = "overlay"
	zfsStorageDriver          storageDriver = "zfs"
)

type dockerFactory struct {
	machineInfoFactory info.MachineInfoFactory

	storageDriver storageDriver
	storageDir    string

	client *docker.Client

	// Information about the mounted cgroup subsystems.
	cgroupSubsystems libcontainer.CgroupSubsystems

	// Information about mounted filesystems.
	fsInfo fs.FsInfo

	dockerVersion []int

	ignoreMetrics container.MetricSet

	thinPoolWatcher *devicemapper.ThinPoolWatcher
}

func (self *dockerFactory) String() string {
	return DockerNamespace
}

func (self *dockerFactory) NewContainerHandler(name string, inHostNamespace bool) (handler container.ContainerHandler, err error) {
	client, err := Client()
	if err != nil {
		return
	}

	metadataEnvs := strings.Split(*dockerEnvWhitelist, ",")

	handler, err = newDockerContainerHandler(
		client,
		name,
		self.machineInfoFactory,
		self.fsInfo,
		self.storageDriver,
		self.storageDir,
		&self.cgroupSubsystems,
		inHostNamespace,
		metadataEnvs,
		self.dockerVersion,
		self.ignoreMetrics,
		self.thinPoolWatcher,
	)
	return
}

// Returns the Docker ID from the full container name.
func ContainerNameToDockerId(name string) string {
	id := path.Base(name)

	if matches := dockerCgroupRegexp.FindStringSubmatch(id); matches != nil {
		return matches[1]
	}

	return id
}

// isContainerName returns true if the cgroup with associated name
// corresponds to a docker container.
func isContainerName(name string) bool {
	// always ignore .mount cgroup even if associated with docker and delegate to systemd
	if strings.HasSuffix(name, ".mount") {
		return false
	}
	return dockerCgroupRegexp.MatchString(path.Base(name))
}

// Docker handles all containers under /docker
func (self *dockerFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	// if the container is not associated with docker, we can't handle it or accept it.
	if !isContainerName(name) {
		return false, false, nil
	}

	// Check if the container is known to docker and it is active.
	id := ContainerNameToDockerId(name)

	// We assume that if Inspect fails then the container is not known to docker.
	ctnr, err := self.client.ContainerInspect(context.Background(), id)
	if err != nil || !ctnr.State.Running {
		return false, true, fmt.Errorf("error inspecting container: %v", err)
	}

	return true, true, nil
}

func (self *dockerFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

var (
	version_regexp_string = `(\d+)\.(\d+)\.(\d+)`
	version_re            = regexp.MustCompile(version_regexp_string)
)

func startThinPoolWatcher(dockerInfo *dockertypes.Info) (*devicemapper.ThinPoolWatcher, error) {
	_, err := devicemapper.ThinLsBinaryPresent()
	if err != nil {
		return nil, err
	}

	if err := ensureThinLsKernelVersion(machine.KernelVersion()); err != nil {
		return nil, err
	}

	if disableThinLs {
		return nil, fmt.Errorf("usage of thin_ls is disabled to preserve iops")
	}

	dockerThinPoolName, err := dockerutil.DockerThinPoolName(*dockerInfo)
	if err != nil {
		return nil, err
	}

	dockerMetadataDevice, err := dockerutil.DockerMetadataDevice(*dockerInfo)
	if err != nil {
		return nil, err
	}

	thinPoolWatcher, err := devicemapper.NewThinPoolWatcher(dockerThinPoolName, dockerMetadataDevice)
	if err != nil {
		return nil, err
	}

	go thinPoolWatcher.Start()
	return thinPoolWatcher, nil
}

func ensureThinLsKernelVersion(kernelVersion string) error {
	// kernel 4.4.0 has the proper bug fixes to allow thin_ls to work without corrupting the thin pool
	minKernelVersion := semver.MustParse("4.4.0")
	// RHEL 7 kernel 3.10.0 release >= 366 has the proper bug fixes backported from 4.4.0 to allow
	// thin_ls to work without corrupting the thin pool
	minRhel7KernelVersion := semver.MustParse("3.10.0")

	matches := version_re.FindStringSubmatch(kernelVersion)
	if len(matches) < 4 {
		return fmt.Errorf("error parsing kernel version: %q is not a semver", kernelVersion)
	}

	sem, err := semver.Make(matches[0])
	if err != nil {
		return err
	}

	if sem.GTE(minKernelVersion) {
		// kernel 4.4+ - good
		return nil
	}

	// Certain RHEL/Centos 7.x kernels have a backport to fix the corruption bug
	if !strings.Contains(kernelVersion, ".el7") {
		// not a RHEL 7.x kernel - won't work
		return fmt.Errorf("kernel version 4.4.0 or later is required to use thin_ls - you have %q", kernelVersion)
	}

	// RHEL/Centos 7.x from here on
	if sem.Major != 3 {
		// only 3.x kernels *may* work correctly
		return fmt.Errorf("RHEL/Centos 7.x kernel version 3.10.0-366 or later is required to use thin_ls - you have %q", kernelVersion)
	}

	if sem.GT(minRhel7KernelVersion) {
		// 3.10.1+ - good
		return nil
	}

	if sem.EQ(minRhel7KernelVersion) {
		// need to check release
		releaseRE := regexp.MustCompile(`^[^-]+-([0-9]+)\.`)
		releaseMatches := releaseRE.FindStringSubmatch(kernelVersion)
		if len(releaseMatches) != 2 {
			return fmt.Errorf("unable to determine RHEL/Centos 7.x kernel release from %q", kernelVersion)
		}

		release, err := strconv.Atoi(releaseMatches[1])
		if err != nil {
			return fmt.Errorf("error parsing release %q: %v", releaseMatches[1], err)
		}

		if release >= 366 {
			return nil
		}
	}

	return fmt.Errorf("RHEL/Centos 7.x kernel version 3.10.0-366 or later is required to use thin_ls - you have %q", kernelVersion)
}

// Register root container before running this function!
func Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo, ignoreMetrics container.MetricSet) error {
	client, err := Client()
	if err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}

	dockerInfo, err := ValidateInfo()
	if err != nil {
		return fmt.Errorf("failed to validate Docker info: %v", err)
	}

	// Version already validated above, assume no error here.
	dockerVersion, _ := parseDockerVersion(dockerInfo.ServerVersion)

	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems()
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}

	var thinPoolWatcher *devicemapper.ThinPoolWatcher
	if storageDriver(dockerInfo.Driver) == devicemapperStorageDriver {
		thinPoolWatcher, err = startThinPoolWatcher(dockerInfo)
		if err != nil {
			glog.Errorf("devicemapper filesystem stats will not be reported: %v", err)
		}
	}

	glog.Infof("Registering Docker factory")
	f := &dockerFactory{
		cgroupSubsystems:   cgroupSubsystems,
		client:             client,
		dockerVersion:      dockerVersion,
		fsInfo:             fsInfo,
		machineInfoFactory: factory,
		storageDriver:      storageDriver(dockerInfo.Driver),
		storageDir:         RootDir(),
		ignoreMetrics:      ignoreMetrics,
		thinPoolWatcher:    thinPoolWatcher,
	}

	container.RegisterContainerHandlerFactory(f, []watcher.ContainerWatchSource{watcher.Raw})
	return nil
}
