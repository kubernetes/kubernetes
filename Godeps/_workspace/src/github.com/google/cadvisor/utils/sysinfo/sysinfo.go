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

package sysinfo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/sysfs"
)

var schedulerRegExp = regexp.MustCompile(".*\\[(.*)\\].*")

// Get information about block devices present on the system.
// Uses the passed in system interface to retrieve the low level OS information.
func GetBlockDeviceInfo(sysfs sysfs.SysFs) (map[string]info.DiskInfo, error) {
	disks, err := sysfs.GetBlockDevices()
	if err != nil {
		return nil, err
	}

	diskMap := make(map[string]info.DiskInfo)
	for _, disk := range disks {
		name := disk.Name()
		// Ignore non-disk devices.
		// TODO(rjnagal): Maybe just match hd, sd, and dm prefixes.
		if strings.HasPrefix(name, "loop") || strings.HasPrefix(name, "ram") || strings.HasPrefix(name, "sr") {
			continue
		}
		disk_info := info.DiskInfo{
			Name: name,
		}
		dev, err := sysfs.GetBlockDeviceNumbers(name)
		if err != nil {
			return nil, err
		}
		n, err := fmt.Sscanf(dev, "%d:%d", &disk_info.Major, &disk_info.Minor)
		if err != nil || n != 2 {
			return nil, fmt.Errorf("could not parse device numbers from %s for device %s", dev, name)
		}
		out, err := sysfs.GetBlockDeviceSize(name)
		if err != nil {
			return nil, err
		}
		// Remove trailing newline before conversion.
		size, err := strconv.ParseUint(strings.TrimSpace(out), 10, 64)
		if err != nil {
			return nil, err
		}
		// size is in 512 bytes blocks.
		disk_info.Size = size * 512

		sched, err := sysfs.GetBlockDeviceScheduler(name)
		if err != nil {
			sched = "none"
		} else {
			matches := schedulerRegExp.FindSubmatch([]byte(sched))
			if len(matches) < 2 {
				sched = "none"
			} else {
				sched = string(matches[1])
			}
		}
		disk_info.Scheduler = sched
		device := fmt.Sprintf("%d:%d", disk_info.Major, disk_info.Minor)
		diskMap[device] = disk_info
	}
	return diskMap, nil
}

// Get information about network devices present on the system.
func GetNetworkDevices(sysfs sysfs.SysFs) ([]info.NetInfo, error) {
	devs, err := sysfs.GetNetworkDevices()
	if err != nil {
		return nil, err
	}
	netDevices := []info.NetInfo{}
	for _, dev := range devs {
		name := dev.Name()
		// Ignore docker, loopback, and veth devices.
		ignoredDevices := []string{"lo", "veth", "docker"}
		ignored := false
		for _, prefix := range ignoredDevices {
			if strings.HasPrefix(name, prefix) {
				ignored = true
				break
			}
		}
		if ignored {
			continue
		}
		address, err := sysfs.GetNetworkAddress(name)
		if err != nil {
			return nil, err
		}
		mtuStr, err := sysfs.GetNetworkMtu(name)
		if err != nil {
			return nil, err
		}
		var mtu int64
		n, err := fmt.Sscanf(mtuStr, "%d", &mtu)
		if err != nil || n != 1 {
			return nil, fmt.Errorf("could not parse mtu from %s for device %s", mtuStr, name)
		}
		netInfo := info.NetInfo{
			Name:       name,
			MacAddress: strings.TrimSpace(address),
			Mtu:        mtu,
		}
		speed, err := sysfs.GetNetworkSpeed(name)
		// Some devices don't set speed.
		if err == nil {
			var s int64
			n, err := fmt.Sscanf(speed, "%d", &s)
			if err != nil || n != 1 {
				return nil, fmt.Errorf("could not parse speed from %s for device %s", speed, name)
			}
			netInfo.Speed = s
		}
		netDevices = append(netDevices, netInfo)
	}
	return netDevices, nil
}

func GetCacheInfo(sysFs sysfs.SysFs, id int) ([]sysfs.CacheInfo, error) {
	caches, err := sysFs.GetCaches(id)
	if err != nil {
		return nil, err
	}

	info := []sysfs.CacheInfo{}
	for _, cache := range caches {
		if !strings.HasPrefix(cache.Name(), "index") {
			continue
		}
		cacheInfo, err := sysFs.GetCacheInfo(id, cache.Name())
		if err != nil {
			return nil, err
		}
		info = append(info, cacheInfo)
	}
	return info, nil
}

func GetNetworkStats(name string) (info.InterfaceStats, error) {
	// TODO(rjnagal): Take syfs as an argument.
	sysFs, err := sysfs.NewRealSysFs()
	if err != nil {
		return info.InterfaceStats{}, err
	}
	return getNetworkStats(name, sysFs)
}

func getNetworkStats(name string, sysFs sysfs.SysFs) (info.InterfaceStats, error) {
	var stats info.InterfaceStats
	var err error
	stats.Name = name
	stats.RxBytes, err = sysFs.GetNetworkStatValue(name, "rx_bytes")
	if err != nil {
		return stats, err
	}
	stats.RxPackets, err = sysFs.GetNetworkStatValue(name, "rx_packets")
	if err != nil {
		return stats, err
	}
	stats.RxErrors, err = sysFs.GetNetworkStatValue(name, "rx_errors")
	if err != nil {
		return stats, err
	}
	stats.RxDropped, err = sysFs.GetNetworkStatValue(name, "rx_dropped")
	if err != nil {
		return stats, err
	}
	stats.TxBytes, err = sysFs.GetNetworkStatValue(name, "tx_bytes")
	if err != nil {
		return stats, err
	}
	stats.TxPackets, err = sysFs.GetNetworkStatValue(name, "tx_packets")
	if err != nil {
		return stats, err
	}
	stats.TxErrors, err = sysFs.GetNetworkStatValue(name, "tx_errors")
	if err != nil {
		return stats, err
	}
	stats.TxDropped, err = sysFs.GetNetworkStatValue(name, "tx_dropped")
	if err != nil {
		return stats, err
	}
	return stats, nil
}

func GetSystemUUID(sysFs sysfs.SysFs) (string, error) {
	return sysFs.GetSystemUUID()
}
