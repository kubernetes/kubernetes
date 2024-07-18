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
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/blang/semver/v4"
	dockertypes "github.com/docker/docker/api/types"

	"github.com/google/cadvisor/container"
	dockerutil "github.com/google/cadvisor/container/docker/utils"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/devicemapper"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/machine"
	"github.com/google/cadvisor/watcher"
	"github.com/google/cadvisor/zfs"

	docker "github.com/docker/docker/client"
	"golang.org/x/net/context"
	"k8s.io/klog/v2"
)

var ArgDockerEndpoint = flag.String("docker", "unix:///var/run/docker.sock", "docker endpoint")
var ArgDockerTLS = flag.Bool("docker-tls", false, "use TLS to connect to docker")
var ArgDockerCert = flag.String("docker-tls-cert", "cert.pem", "path to client certificate")
var ArgDockerKey = flag.String("docker-tls-key", "key.pem", "path to private key")
var ArgDockerCA = flag.String("docker-tls-ca", "ca.pem", "path to trusted CA")

var dockerEnvMetadataWhiteList = flag.String("docker_env_metadata_whitelist", "", "DEPRECATED: this flag will be removed, please use `env_metadata_whitelist`. A comma-separated list of environment variable keys matched with specified prefix that needs to be collected for docker containers")

// The namespace under which Docker aliases are unique.
const DockerNamespace = "docker"

// The retry times for getting docker root dir
const rootDirRetries = 5

// The retry period for getting docker root dir, Millisecond
const rootDirRetryPeriod time.Duration = 1000 * time.Millisecond

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
		for i := 0; i < rootDirRetries; i++ {
			status, err := Status()
			if err == nil && status.RootDir != "" {
				dockerRootDir = status.RootDir
				break
			} else {
				time.Sleep(rootDirRetryPeriod)
			}
		}
		if dockerRootDir == "" {
			dockerRootDir = *dockerRootDirFlag
		}
	})
	return dockerRootDir
}

type StorageDriver string

const (
	DevicemapperStorageDriver StorageDriver = "devicemapper"
	AufsStorageDriver         StorageDriver = "aufs"
	OverlayStorageDriver      StorageDriver = "overlay"
	Overlay2StorageDriver     StorageDriver = "overlay2"
	ZfsStorageDriver          StorageDriver = "zfs"
	VfsStorageDriver          StorageDriver = "vfs"
)

type dockerFactory struct {
	machineInfoFactory info.MachineInfoFactory

	storageDriver StorageDriver
	storageDir    string

	client *docker.Client

	// Information about the mounted cgroup subsystems.
	cgroupSubsystems map[string]string

	// Information about mounted filesystems.
	fsInfo fs.FsInfo

	dockerVersion []int

	dockerAPIVersion []int

	includedMetrics container.MetricSet

	thinPoolName    string
	thinPoolWatcher *devicemapper.ThinPoolWatcher

	zfsWatcher *zfs.ZfsWatcher
}

func (f *dockerFactory) String() string {
	return DockerNamespace
}

func (f *dockerFactory) NewContainerHandler(name string, metadataEnvAllowList []string, inHostNamespace bool) (handler container.ContainerHandler, err error) {
	client, err := Client()
	if err != nil {
		return
	}

	dockerMetadataEnvAllowList := strings.Split(*dockerEnvMetadataWhiteList, ",")

	// prefer using the unified metadataEnvAllowList
	if len(metadataEnvAllowList) != 0 {
		dockerMetadataEnvAllowList = metadataEnvAllowList
	}

	handler, err = newDockerContainerHandler(
		client,
		name,
		f.machineInfoFactory,
		f.fsInfo,
		f.storageDriver,
		f.storageDir,
		f.cgroupSubsystems,
		inHostNamespace,
		dockerMetadataEnvAllowList,
		f.dockerVersion,
		f.includedMetrics,
		f.thinPoolName,
		f.thinPoolWatcher,
		f.zfsWatcher,
	)
	return
}

// Docker handles all containers under /docker
func (f *dockerFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	// if the container is not associated with docker, we can't handle it or accept it.
	if !dockerutil.IsContainerName(name) {
		return false, false, nil
	}

	// Check if the container is known to docker and it is active.
	id := dockerutil.ContainerNameToId(name)

	// We assume that if Inspect fails then the container is not known to docker.
	ctnr, err := f.client.ContainerInspect(context.Background(), id)
	if err != nil || !ctnr.State.Running {
		return false, true, fmt.Errorf("error inspecting container: %v", err)
	}

	return true, true, nil
}

func (f *dockerFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

var (
	versionRegexpString    = `(\d+)\.(\d+)\.(\d+)`
	VersionRe              = regexp.MustCompile(versionRegexpString)
	apiVersionRegexpString = `(\d+)\.(\d+)`
	apiVersionRe           = regexp.MustCompile(apiVersionRegexpString)
)

func StartThinPoolWatcher(dockerInfo *dockertypes.Info) (*devicemapper.ThinPoolWatcher, error) {
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

func StartZfsWatcher(dockerInfo *dockertypes.Info) (*zfs.ZfsWatcher, error) {
	filesystem, err := dockerutil.DockerZfsFilesystem(*dockerInfo)
	if err != nil {
		return nil, err
	}

	zfsWatcher, err := zfs.NewZfsWatcher(filesystem)
	if err != nil {
		return nil, err
	}

	go zfsWatcher.Start()
	return zfsWatcher, nil
}

func ensureThinLsKernelVersion(kernelVersion string) error {
	// kernel 4.4.0 has the proper bug fixes to allow thin_ls to work without corrupting the thin pool
	minKernelVersion := semver.MustParse("4.4.0")
	// RHEL 7 kernel 3.10.0 release >= 366 has the proper bug fixes backported from 4.4.0 to allow
	// thin_ls to work without corrupting the thin pool
	minRhel7KernelVersion := semver.MustParse("3.10.0")

	matches := VersionRe.FindStringSubmatch(kernelVersion)
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
func Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics container.MetricSet) error {
	client, err := Client()
	if err != nil {
		return fmt.Errorf("unable to communicate with docker daemon: %v", err)
	}

	dockerInfo, err := ValidateInfo(Info, VersionString)
	if err != nil {
		return fmt.Errorf("failed to validate Docker info: %v", err)
	}

	// Version already validated above, assume no error here.
	dockerVersion, _ := ParseVersion(dockerInfo.ServerVersion, VersionRe, 3)

	dockerAPIVersion, _ := APIVersion()

	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems(includedMetrics)
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}

	var (
		thinPoolWatcher *devicemapper.ThinPoolWatcher
		thinPoolName    string
		zfsWatcher      *zfs.ZfsWatcher
	)
	if includedMetrics.Has(container.DiskUsageMetrics) {
		if StorageDriver(dockerInfo.Driver) == DevicemapperStorageDriver {
			thinPoolWatcher, err = StartThinPoolWatcher(dockerInfo)
			if err != nil {
				klog.Errorf("devicemapper filesystem stats will not be reported: %v", err)
			}

			// Safe to ignore error - driver status should always be populated.
			status, _ := StatusFromDockerInfo(*dockerInfo)
			thinPoolName = status.DriverStatus[dockerutil.DriverStatusPoolName]
		}

		if StorageDriver(dockerInfo.Driver) == ZfsStorageDriver {
			zfsWatcher, err = StartZfsWatcher(dockerInfo)
			if err != nil {
				klog.Errorf("zfs filesystem stats will not be reported: %v", err)
			}
		}
	}

	klog.V(1).Infof("Registering Docker factory")
	f := &dockerFactory{
		cgroupSubsystems:   cgroupSubsystems,
		client:             client,
		dockerVersion:      dockerVersion,
		dockerAPIVersion:   dockerAPIVersion,
		fsInfo:             fsInfo,
		machineInfoFactory: factory,
		storageDriver:      StorageDriver(dockerInfo.Driver),
		storageDir:         RootDir(),
		includedMetrics:    includedMetrics,
		thinPoolName:       thinPoolName,
		thinPoolWatcher:    thinPoolWatcher,
		zfsWatcher:         zfsWatcher,
	}

	container.RegisterContainerHandlerFactory(f, []watcher.ContainerWatchSource{watcher.Raw})
	return nil
}
