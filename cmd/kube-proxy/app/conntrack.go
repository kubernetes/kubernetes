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
	"io/ioutil"
	"strconv"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/sysctl"
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
	// SetTCPCloseWaitTimeout nf_conntrack_tcp_timeout_close_wait.
	SetTCPCloseWaitTimeout(seconds int) error
}

type realConntracker struct{}

var readOnlySysFSError = errors.New("readOnlySysFS")

func (rct realConntracker) SetMax(max int) error {
	if err := rct.setIntSysCtl("nf_conntrack_max", max); err != nil {
		return err
	}
	// sysfs is expected to be mounted as 'rw'. However, it may be
	// unexpectedly mounted as 'ro' by docker because of a known docker
	// issue (https://github.com/docker/docker/issues/24000). Setting
	// conntrack will fail when sysfs is readonly. When that happens, we
	// don't set conntrack hashsize and return a special error
	// readOnlySysFSError here. The caller should deal with
	// readOnlySysFSError differently.
	writable, err := isSysFSWritable()
	if err != nil {
		return err
	}
	if !writable {
		return readOnlySysFSError
	}
	// TODO: generify this and sysctl to a new sysfs.WriteInt()
	glog.Infof("Setting conntrack hashsize to %d", max/4)
	return writeIntStringFile("/sys/module/nf_conntrack/parameters/hashsize", max/4)
}

func (rct realConntracker) SetTCPEstablishedTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_tcp_timeout_established", seconds)
}

func (rct realConntracker) SetTCPCloseWaitTimeout(seconds int) error {
	return rct.setIntSysCtl("nf_conntrack_tcp_timeout_close_wait", seconds)
}

func (realConntracker) setIntSysCtl(name string, value int) error {
	entry := "net/netfilter/" + name

	glog.Infof("Set sysctl '%v' to %v", entry, value)
	if err := sysctl.New().SetSysctl(entry, value); err != nil {
		return err
	}
	return nil
}

// isSysFSWritable checks /proc/mounts to see whether sysfs is 'rw' or not.
func isSysFSWritable() (bool, error) {
	const permWritable = "rw"
	const sysfsDevice = "sysfs"
	m := mount.New("" /* default mount path */)
	mountPoints, err := m.List()
	if err != nil {
		glog.Errorf("failed to list mount points: %v", err)
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
		glog.Errorf("sysfs is not writable: %+v (mount options are %v)",
			mountPoint, mountPoint.Opts)
		return false, readOnlySysFSError
	}

	return false, errors.New("No sysfs mounted")
}

func writeIntStringFile(filename string, value int) error {
	return ioutil.WriteFile(filename, []byte(strconv.Itoa(value)), 0640)
}
