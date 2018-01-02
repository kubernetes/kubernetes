//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package mockexec

import (
	"github.com/heketi/heketi/executors"
)

type MockExecutor struct {
	// These functions can be overwritten for testing
	MockGlusterdCheck      func(host string) error
	MockPeerProbe          func(exec_host, newnode string) error
	MockPeerDetach         func(exec_host, newnode string) error
	MockDeviceSetup        func(host, device, vgid string) (*executors.DeviceInfo, error)
	MockDeviceTeardown     func(host, device, vgid string) error
	MockBrickCreate        func(host string, brick *executors.BrickRequest) (*executors.BrickInfo, error)
	MockBrickDestroy       func(host string, brick *executors.BrickRequest) error
	MockBrickDestroyCheck  func(host string, brick *executors.BrickRequest) error
	MockVolumeCreate       func(host string, volume *executors.VolumeRequest) (*executors.Volume, error)
	MockVolumeExpand       func(host string, volume *executors.VolumeRequest) (*executors.Volume, error)
	MockVolumeDestroy      func(host string, volume string) error
	MockVolumeDestroyCheck func(host, volume string) error
	MockVolumeReplaceBrick func(host string, volume string, oldBrick *executors.BrickInfo, newBrick *executors.BrickInfo) error
	MockVolumeInfo         func(host string, volume string) (*executors.Volume, error)
}

func NewMockExecutor() (*MockExecutor, error) {
	m := &MockExecutor{}

	m.MockGlusterdCheck = func(host string) error {
		return nil
	}

	m.MockPeerProbe = func(exec_host, newnode string) error {
		return nil
	}

	m.MockPeerDetach = func(exec_host, newnode string) error {
		return nil
	}

	m.MockDeviceSetup = func(host, device, vgid string) (*executors.DeviceInfo, error) {
		d := &executors.DeviceInfo{}
		d.Size = 500 * 1024 * 1024 // Size in KB
		d.ExtentSize = 4096
		return d, nil
	}

	m.MockDeviceTeardown = func(host, device, vgid string) error {
		return nil
	}

	m.MockBrickCreate = func(host string, brick *executors.BrickRequest) (*executors.BrickInfo, error) {
		b := &executors.BrickInfo{
			Path: "/mockpath",
		}
		return b, nil
	}

	m.MockBrickDestroy = func(host string, brick *executors.BrickRequest) error {
		return nil
	}

	m.MockBrickDestroyCheck = func(host string, brick *executors.BrickRequest) error {
		return nil
	}

	m.MockVolumeCreate = func(host string, volume *executors.VolumeRequest) (*executors.Volume, error) {
		return &executors.Volume{}, nil
	}

	m.MockVolumeExpand = func(host string, volume *executors.VolumeRequest) (*executors.Volume, error) {
		return &executors.Volume{}, nil
	}

	m.MockVolumeDestroy = func(host string, volume string) error {
		return nil
	}

	m.MockVolumeDestroyCheck = func(host, volume string) error {
		return nil
	}

	m.MockVolumeReplaceBrick = func(host string, volume string, oldBrick *executors.BrickInfo, newBrick *executors.BrickInfo) error {
		return nil
	}

	m.MockVolumeInfo = func(host string, volume string) (*executors.Volume, error) {
		var bricks []executors.Brick
		brick := executors.Brick{Name: host + ":/mockpath"}
		bricks = append(bricks, brick)
		brick = executors.Brick{Name: host + ":/mockpath"}
		bricks = append(bricks, brick)
		brick = executors.Brick{Name: host + ":/mockpath"}
		bricks = append(bricks, brick)
		Bricks := executors.Bricks{
			BrickList: bricks,
		}
		vinfo := &executors.Volume{
			Bricks: Bricks,
		}
		return vinfo, nil
	}

	return m, nil
}

func (m *MockExecutor) SetLogLevel(level string) {

}

func (m *MockExecutor) GlusterdCheck(host string) error {
	return m.MockGlusterdCheck(host)
}

func (m *MockExecutor) PeerProbe(exec_host, newnode string) error {
	return m.MockPeerProbe(exec_host, newnode)
}

func (m *MockExecutor) PeerDetach(exec_host, newnode string) error {
	return m.MockPeerDetach(exec_host, newnode)
}

func (m *MockExecutor) DeviceSetup(host, device, vgid string) (*executors.DeviceInfo, error) {
	return m.MockDeviceSetup(host, device, vgid)
}

func (m *MockExecutor) DeviceTeardown(host, device, vgid string) error {
	return m.MockDeviceTeardown(host, device, vgid)
}

func (m *MockExecutor) BrickCreate(host string, brick *executors.BrickRequest) (*executors.BrickInfo, error) {
	return m.MockBrickCreate(host, brick)
}

func (m *MockExecutor) BrickDestroy(host string, brick *executors.BrickRequest) error {
	return m.MockBrickDestroy(host, brick)
}

func (m *MockExecutor) BrickDestroyCheck(host string, brick *executors.BrickRequest) error {
	return m.MockBrickDestroyCheck(host, brick)
}

func (m *MockExecutor) VolumeCreate(host string, volume *executors.VolumeRequest) (*executors.Volume, error) {
	return m.MockVolumeCreate(host, volume)
}

func (m *MockExecutor) VolumeExpand(host string, volume *executors.VolumeRequest) (*executors.Volume, error) {
	return m.MockVolumeExpand(host, volume)
}

func (m *MockExecutor) VolumeDestroy(host string, volume string) error {
	return m.MockVolumeDestroy(host, volume)
}

func (m *MockExecutor) VolumeDestroyCheck(host string, volume string) error {
	return m.MockVolumeDestroyCheck(host, volume)
}

func (m *MockExecutor) VolumeReplaceBrick(host string, volume string, oldBrick *executors.BrickInfo, newBrick *executors.BrickInfo) error {
	return m.MockVolumeReplaceBrick(host, volume, oldBrick, newBrick)
}

func (m *MockExecutor) VolumeInfo(host string, volume string) (*executors.Volume, error) {
	return m.MockVolumeInfo(host, volume)
}
