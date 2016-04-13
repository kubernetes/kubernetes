// +build linux

package server

import (
	"testing"

	"github.com/docker/docker/pkg/version"
	"github.com/docker/docker/runconfig"
)

func TestAdjustCpuSharesOldApi(t *testing.T) {
	apiVersion := version.Version("1.18")
	hostConfig := &runconfig.HostConfig{
		CpuShares: linuxMinCpuShares - 1,
	}
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != linuxMinCpuShares {
		t.Errorf("Expected CpuShares to be %d", linuxMinCpuShares)
	}

	hostConfig.CpuShares = linuxMaxCpuShares + 1
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != linuxMaxCpuShares {
		t.Errorf("Expected CpuShares to be %d", linuxMaxCpuShares)
	}

	hostConfig.CpuShares = 0
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != 0 {
		t.Error("Expected CpuShares to be unchanged")
	}

	hostConfig.CpuShares = 1024
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != 1024 {
		t.Error("Expected CpuShares to be unchanged")
	}
}

func TestAdjustCpuSharesNoAdjustment(t *testing.T) {
	apiVersion := version.Version("1.19")
	hostConfig := &runconfig.HostConfig{
		CpuShares: linuxMinCpuShares - 1,
	}
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != linuxMinCpuShares-1 {
		t.Errorf("Expected CpuShares to be %d", linuxMinCpuShares-1)
	}

	hostConfig.CpuShares = linuxMaxCpuShares + 1
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != linuxMaxCpuShares+1 {
		t.Errorf("Expected CpuShares to be %d", linuxMaxCpuShares+1)
	}

	hostConfig.CpuShares = 0
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != 0 {
		t.Error("Expected CpuShares to be unchanged")
	}

	hostConfig.CpuShares = 1024
	adjustCpuShares(apiVersion, hostConfig)
	if hostConfig.CpuShares != 1024 {
		t.Error("Expected CpuShares to be unchanged")
	}
}
