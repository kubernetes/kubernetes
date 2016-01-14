/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"strconv"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util/sysctl"
)

type Conntracker interface {
	SetMax(max int) error
	SetTCPEstablishedTimeout(seconds int) error
}

type realConntracker struct{}

func (realConntracker) SetMax(max int) error {
	glog.Infof("Setting nf_conntrack_max to %d", max)
	if err := sysctl.SetSysctl("net/netfilter/nf_conntrack_max", max); err != nil {
		return err
	}
	// TODO: generify this and sysctl to a new sysfs.WriteInt()
	glog.Infof("Setting conntrack hashsize to %d", max/4)
	return ioutil.WriteFile("/sys/module/nf_conntrack/parameters/hashsize", []byte(strconv.Itoa(max/4)), 0640)
}

func (realConntracker) SetTCPEstablishedTimeout(seconds int) error {
	glog.Infof("Setting nf_conntrack_tcp_timeout_established to %d", seconds)
	return sysctl.SetSysctl("net/netfilter/nf_conntrack_tcp_timeout_established", seconds)
}
