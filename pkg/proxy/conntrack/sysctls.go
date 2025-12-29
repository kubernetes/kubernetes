//go:build linux
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

package conntrack

import (
	"context"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/machine"
	"github.com/google/cadvisor/utils/sysfs"

	"k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/klog/v2"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
)

func SetSysctls(ctx context.Context, config *proxyconfigapi.KubeProxyConntrackConfiguration) error {
	return setSysctls(ctx, realConntrackConfigurer{}, config)
}

// conntrackConfigurer is a mockable interface for setting conntrack sysctls.
//
// Descriptions of the various sysctl fields can be found here:
// https://www.kernel.org/doc/Documentation/networking/nf_conntrack-sysctl.txt
type conntrackConfigurer interface {
	// SetMax adjusts nf_conntrack_max.
	SetMax(ctx context.Context, max int) error
	// SetTCPEstablishedTimeout adjusts nf_conntrack_tcp_timeout_established.
	SetTCPEstablishedTimeout(ctx context.Context, seconds int) error
	// SetTCPCloseWaitTimeout adjusts nf_conntrack_tcp_timeout_close_wait.
	SetTCPCloseWaitTimeout(ctx context.Context, seconds int) error
	// SetTCPBeLiberal adjusts nf_conntrack_tcp_be_liberal.
	SetTCPBeLiberal(ctx context.Context, value int) error
	// SetUDPTimeout adjusts nf_conntrack_udp_timeout.
	SetUDPTimeout(ctx context.Context, seconds int) error
	// SetUDPStreamTimeout adjusts nf_conntrack_udp_timeout_stream.
	SetUDPStreamTimeout(ctx context.Context, seconds int) error
}

func setSysctls(ctx context.Context, ct conntrackConfigurer, config *proxyconfigapi.KubeProxyConntrackConfiguration) error {
	max, err := getConntrackMax(ctx, config)
	if err != nil {
		return err
	}
	if max > 0 {
		err := ct.SetMax(ctx, max)
		if err != nil {
			return err
		}
	}

	if config.TCPEstablishedTimeout != nil && config.TCPEstablishedTimeout.Duration > 0 {
		timeout := int(config.TCPEstablishedTimeout.Duration / time.Second)
		if err := ct.SetTCPEstablishedTimeout(ctx, timeout); err != nil {
			return err
		}
	}

	if config.TCPCloseWaitTimeout != nil && config.TCPCloseWaitTimeout.Duration > 0 {
		timeout := int(config.TCPCloseWaitTimeout.Duration / time.Second)
		if err := ct.SetTCPCloseWaitTimeout(ctx, timeout); err != nil {
			return err
		}
	}

	if config.TCPBeLiberal {
		if err := ct.SetTCPBeLiberal(ctx, 1); err != nil {
			return err
		}
	}

	if config.UDPTimeout.Duration > 0 {
		timeout := int(config.UDPTimeout.Duration / time.Second)
		if err := ct.SetUDPTimeout(ctx, timeout); err != nil {
			return err
		}
	}

	if config.UDPStreamTimeout.Duration > 0 {
		timeout := int(config.UDPStreamTimeout.Duration / time.Second)
		if err := ct.SetUDPStreamTimeout(ctx, timeout); err != nil {
			return err
		}
	}

	return nil
}

func getConntrackMax(ctx context.Context, config *proxyconfigapi.KubeProxyConntrackConfiguration) (int, error) {
	logger := klog.FromContext(ctx)
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		scaled := int(*config.MaxPerCore) * detectNumCPU()
		if scaled > floor {
			logger.V(3).Info("GetConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		logger.V(3).Info("GetConntrackMax: using conntrack-min")
		return floor, nil
	}
	return 0, nil
}

func detectNumCPU() int {
	// try get numCPU from /sys firstly due to a known issue (https://github.com/kubernetes/kubernetes/issues/99225)
	_, numCPU, err := machine.GetTopology(sysfs.NewRealSysFs())
	if err != nil || numCPU < 1 {
		return runtime.NumCPU()
	}
	return numCPU
}

type realConntrackConfigurer struct {
}

func (rct realConntrackConfigurer) SetMax(ctx context.Context, max int) error {
	logger := klog.FromContext(ctx)
	logger.Info("Setting nf_conntrack_max", "nfConntrackMax", max)
	if err := rct.setIntSysCtl(ctx, "nf_conntrack_max", max); err != nil {
		return err
	}

	// Check if hashsize is large enough for the nf_conntrack_max value.
	hashsize, err := readIntStringFile("/sys/module/nf_conntrack/parameters/hashsize")
	if err != nil {
		return err
	}
	if hashsize >= (max / 4) {
		return nil
	}

	logger.Info("Setting conntrack hashsize", "conntrackHashsize", max/4)
	return writeIntStringFile("/sys/module/nf_conntrack/parameters/hashsize", max/4)
}

func (rct realConntrackConfigurer) SetTCPEstablishedTimeout(ctx context.Context, seconds int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_tcp_timeout_established", seconds)
}

func (rct realConntrackConfigurer) SetTCPCloseWaitTimeout(ctx context.Context, seconds int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_tcp_timeout_close_wait", seconds)
}

func (rct realConntrackConfigurer) SetTCPBeLiberal(ctx context.Context, value int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_tcp_be_liberal", value)
}

func (rct realConntrackConfigurer) SetUDPTimeout(ctx context.Context, seconds int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_udp_timeout", seconds)
}

func (rct realConntrackConfigurer) SetUDPStreamTimeout(ctx context.Context, seconds int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_udp_timeout_stream", seconds)
}

func (rct realConntrackConfigurer) setIntSysCtl(ctx context.Context, name string, value int) error {
	logger := klog.FromContext(ctx)
	entry := "net/netfilter/" + name

	sys := sysctl.New()
	if val, _ := sys.GetSysctl(entry); val != value {
		logger.Info("Set sysctl", "entry", entry, "value", value)
		if err := sys.SetSysctl(entry, value); err != nil {
			return err
		}
	}
	return nil
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
