// +build !windows

package daemon

import (
	"context"
	"os/exec"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// FillPlatformInfo fills the platform related info.
func (daemon *Daemon) FillPlatformInfo(v *types.Info, sysInfo *sysinfo.SysInfo) {
	v.MemoryLimit = sysInfo.MemoryLimit
	v.SwapLimit = sysInfo.SwapLimit
	v.KernelMemory = sysInfo.KernelMemory
	v.OomKillDisable = sysInfo.OomKillDisable
	v.CPUCfsPeriod = sysInfo.CPUCfsPeriod
	v.CPUCfsQuota = sysInfo.CPUCfsQuota
	v.CPUShares = sysInfo.CPUShares
	v.CPUSet = sysInfo.Cpuset
	v.Runtimes = daemon.configStore.GetAllRuntimes()
	v.DefaultRuntime = daemon.configStore.GetDefaultRuntimeName()
	v.InitBinary = daemon.configStore.GetInitPath()

	v.ContainerdCommit.Expected = dockerversion.ContainerdCommitID
	if sv, err := daemon.containerd.GetServerVersion(context.Background()); err == nil {
		v.ContainerdCommit.ID = sv.Revision
	} else {
		logrus.Warnf("failed to retrieve containerd version: %v", err)
		v.ContainerdCommit.ID = "N/A"
	}

	v.RuncCommit.Expected = dockerversion.RuncCommitID
	defaultRuntimeBinary := daemon.configStore.GetRuntime(daemon.configStore.GetDefaultRuntimeName()).Path
	if rv, err := exec.Command(defaultRuntimeBinary, "--version").Output(); err == nil {
		parts := strings.Split(strings.TrimSpace(string(rv)), "\n")
		if len(parts) == 3 {
			parts = strings.Split(parts[1], ": ")
			if len(parts) == 2 {
				v.RuncCommit.ID = strings.TrimSpace(parts[1])
			}
		}

		if v.RuncCommit.ID == "" {
			logrus.Warnf("failed to retrieve %s version: unknown output format: %s", defaultRuntimeBinary, string(rv))
			v.RuncCommit.ID = "N/A"
		}
	} else {
		logrus.Warnf("failed to retrieve %s version: %v", defaultRuntimeBinary, err)
		v.RuncCommit.ID = "N/A"
	}

	defaultInitBinary := daemon.configStore.GetInitPath()
	if rv, err := exec.Command(defaultInitBinary, "--version").Output(); err == nil {
		ver, err := parseInitVersion(string(rv))

		if err != nil {
			logrus.Warnf("failed to retrieve %s version: %s", defaultInitBinary, err)
		}
		v.InitCommit = ver
	} else {
		logrus.Warnf("failed to retrieve %s version: %s", defaultInitBinary, err)
		v.InitCommit.ID = "N/A"
	}
}

// parseInitVersion parses a Tini version string, and extracts the version.
func parseInitVersion(v string) (types.Commit, error) {
	version := types.Commit{ID: "", Expected: dockerversion.InitCommitID}
	parts := strings.Split(strings.TrimSpace(v), " - ")

	if len(parts) >= 2 {
		gitParts := strings.Split(parts[1], ".")
		if len(gitParts) == 2 && gitParts[0] == "git" {
			version.ID = gitParts[1]
			version.Expected = dockerversion.InitCommitID[0:len(version.ID)]
		}
	}
	if version.ID == "" && strings.HasPrefix(parts[0], "tini version ") {
		version.ID = "v" + strings.TrimPrefix(parts[0], "tini version ")
	}
	if version.ID == "" {
		version.ID = "N/A"
		return version, errors.Errorf("unknown output format: %s", v)
	}
	return version, nil
}
