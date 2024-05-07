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

package app

import (
	"errors"
	"os"
	"strconv"
	"strings"

	"k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
)

// Conntracker is an interface to the global sysctl. Descriptions of the various
// sysctl fields can be found here:
//
// https://www.kernel.org/doc/Documentation/networking/nf_conntrack-sysctl.txt
type Conntracker interface {
	// SetMax adjusts nf_conntrack_max.
	SetMax(max int) error
	// SetTCPEstablishedTimeout adjusts nf_conntrack_tcp_timeout_established.
	SetTCPEstablishedTimeout(seconds int) error
	// SetTCPCloseWaitTimeout adjusts nf_conntrack_tcp_timeout_close_wait.
	SetTCPCloseWaitTimeout(seconds int) error
	// SetTCPBeLiberal adjusts nf_conntrack_tcp_be_liberal.
	SetTCPBeLiberal(value int) error
	// SetUDPTimeout adjusts nf_conntrack_udp_timeout.
	SetUDPTimeout(seconds int) error
	// SetUDPStreamTimeout adjusts nf_conntrack_udp_timeout_stream.
	SetUDPStreamTimeout(seconds int) error
}

type realConntracker struct {
	logger klog.Logger
}

var errReadOnlySysFS = errors.New("readOnlySysFS")

func (rct realConntracker) SetMax(max int) error {
	if err := rct.setIntSysCtl("nf_conntrack_max", max); err != nil {
		return err
	}
	rct.logger.Info("Setting nf_conntrack_max", "nfConntrackMax", max)

	// Linux does not support writing to /sys/module/nf_conntrack/parameters/hashsize
	// when the writer process is not in the initial network namespace
	// (https://github.com/torvalds/linux/blob/v4.10/net/netfilter/nf_conntrack_core.c#L1795-L1796).
	// Usually that's fine. But in some configurations such as with github.com/kinvolk/kubeadm-nspawn,
	// kube-proxy is in another netns.
	// Therefore, check if writing in hashsize is necessary and skip the writing if not.
	hashsize, err := readIntStringFile("/sys/module/nf_conntrack/parameters/hashsize")
	if err != nil {
		return err
	}
	if hashsize >= (max / 4) {
		return nil
	}

	// sysfs is expected to be mounted as 'rw'. However, it may be
	// unexpectedly mounted as 'ro' by docker because of a known docker
	// issue (https://github.com/docker/docker/issues/24000). Setting
	// conntrack will fail when sysfs is readonly. When that happens, we
	// don't set conntrack hashsize and return a special error
	// errReadOnlySysFS here. The caller should deal with
	// errReadOnlySysFS differently.
	writable, err := rct.isSysFSWritable()
	if err != nil {
		return err
	}
	if !writable {
		return errReadOnlySysFS
	}
	// TODO: generify this and sysctl to a new sysfs.WriteInt()
	rct.logger.Info("Setting conntrack hashsize", "conntrackHashsize", max/4)
	return writeIntStringFile("/sys/module/nf_conntrack/parameters/hashsize", max/4)
}

func (rct realConntracker) SetTCPEstablishedTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_tcp_timeout_established", seconds)
}

func (rct realConntracker) SetTCPCloseWaitTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_tcp_timeout_close_wait", seconds)
}

func (rct realConntracker) SetTCPBeLiberal(value int) error {
	return rct.setIntSysCtl("nf_conntrack_tcp_be_liberal", value)
}

func (rct realConntracker) SetUDPTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_udp_timeout", seconds)
}

func (rct realConntracker) SetUDPStreamTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_udp_timeout_stream", seconds)
}

func (rct realConntracker) setIntSysCtl(name string, value int) error {
	entry := "net/netfilter/" + name

	sys := sysctl.New()
	if val, _ := sys.GetSysctl(entry); val != value {
		rct.logger.Info("Set sysctl", "entry", entry, "value", value)
		if err := sys.SetSysctl(entry, value); err != nil {
			return err
		}
	}
	return nil
}

// isSysFSWritable checks /proc/mounts to see whether sysfs is 'rw' or not.
func (rct realConntracker) isSysFSWritable() (bool, error) {
	const permWritable = "rw"
	const sysfsDevice = "sysfs"
	m := mount.New("" /* default mount path */)
	mountPoints, err := m.List()
	if err != nil {
		rct.logger.Error(err, "Failed to list mount points")
		return false, err
	}

	for _, mountPoint := range mountPoints {
		if mountPoint.Type != sysfsDevice {
			continue
		}
		// Check whether sysfs is 'rw'
		if len(mountPoint.Opts) > 0 && mountPoint.Opts[0] == permWritable {
			return true, nil
		}
		rct.logger.Error(nil, "Sysfs is not writable", "mountPoint", mountPoint, "mountOptions", mountPoint.Opts)
		return false, errReadOnlySysFS
	}

	return false, errors.New("no sysfs mounted")
}

func readIntStringFile(filename string) (int, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return -1, err
	}
	return strconv.Atoi(strings.TrimSpace(string(b)))
}

func writeIntStringFile(filename string, value int) error {
	return os.WriteFile(filename, []byte(strconv.Itoa(value)), 0640)
}
