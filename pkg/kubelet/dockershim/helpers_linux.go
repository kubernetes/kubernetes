// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package dockershim

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/blang/semver"
	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
)

func DefaultMemorySwap() int64 {
	return 0
}

func (ds *dockerService) getSecurityOpts(seccompProfile string, separator rune) ([]string, error) {
	// Apply seccomp options.
	seccompSecurityOpts, err := getSeccompSecurityOpts(seccompProfile, separator)
	if err != nil {
		return nil, fmt.Errorf("failed to generate seccomp security options for container: %v", err)
	}

	return seccompSecurityOpts, nil
}

func getSeccompDockerOpts(seccompProfile string) ([]dockerOpt, error) {
	if seccompProfile == "" || seccompProfile == "unconfined" {
		// return early the default
		return defaultSeccompOpt, nil
	}

	if seccompProfile == "docker/default" {
		// return nil so docker will load the default seccomp profile
		return nil, nil
	}

	if !strings.HasPrefix(seccompProfile, "localhost/") {
		return nil, fmt.Errorf("unknown seccomp profile option: %s", seccompProfile)
	}

	fname := strings.TrimPrefix(seccompProfile, "localhost/") // by pod annotation validation, name is a valid subpath
	file, err := ioutil.ReadFile(filepath.FromSlash(fname))
	if err != nil {
		return nil, fmt.Errorf("cannot load seccomp profile %q: %v", fname, err)
	}

	b := bytes.NewBuffer(nil)
	if err := json.Compact(b, file); err != nil {
		return nil, err
	}
	// Rather than the full profile, just put the filename & md5sum in the event log.
	msg := fmt.Sprintf("%s(md5:%x)", fname, md5.Sum(file))

	return []dockerOpt{{"seccomp", b.String(), msg}}, nil
}

// getSeccompSecurityOpts gets container seccomp options from container seccomp profile.
// It is an experimental feature and may be promoted to official runtime api in the future.
func getSeccompSecurityOpts(seccompProfile string, separator rune) ([]string, error) {
	seccompOpts, err := getSeccompDockerOpts(seccompProfile)
	if err != nil {
		return nil, err
	}
	return fmtDockerOpts(seccompOpts, separator), nil
}

func (ds *dockerService) updateCreateConfig(
	createConfig *dockertypes.ContainerCreateConfig,
	config *runtimeapi.ContainerConfig,
	sandboxConfig *runtimeapi.PodSandboxConfig,
	podSandboxID string, securityOptSep rune, apiVersion *semver.Version) error {
	// Apply Linux-specific options if applicable.
	if lc := config.GetLinux(); lc != nil {
		// TODO: Check if the units are correct.
		// TODO: Can we assume the defaults are sane?
		rOpts := lc.GetResources()
		if rOpts != nil {
			createConfig.HostConfig.Resources = dockercontainer.Resources{
				Memory:     rOpts.MemoryLimitInBytes,
				MemorySwap: DefaultMemorySwap(),
				CPUShares:  rOpts.CpuShares,
				CPUQuota:   rOpts.CpuQuota,
				CPUPeriod:  rOpts.CpuPeriod,
			}
			createConfig.HostConfig.OomScoreAdj = int(rOpts.OomScoreAdj)
		}
		// Note: ShmSize is handled in kube_docker_client.go

		// Apply security context.
		if err := applyContainerSecurityContext(lc, podSandboxID, createConfig.Config, createConfig.HostConfig, securityOptSep); err != nil {
			return fmt.Errorf("failed to apply container security context for container %q: %v", config.Metadata.Name, err)
		}
		modifyPIDNamespaceOverrides(ds.disableSharedPID, apiVersion, createConfig.HostConfig)
	}

	// Apply cgroupsParent derived from the sandbox config.
	if lc := sandboxConfig.GetLinux(); lc != nil {
		// Apply Cgroup options.
		cgroupParent, err := ds.GenerateExpectedCgroupParent(lc.CgroupParent)
		if err != nil {
			return fmt.Errorf("failed to generate cgroup parent in expected syntax for container %q: %v", config.Metadata.Name, err)
		}
		createConfig.HostConfig.CgroupParent = cgroupParent
	}

	return nil
}

func (ds *dockerService) determinePodIPBySandboxID(uid string) string {
	return ""
}
