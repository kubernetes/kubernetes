//go:build linux

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

	"k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/cpuset"
)

func SetSysctls(ctx context.Context, config *kubeproxyconfig.KubeProxyConntrackConfiguration) error {
	return setSysctls(ctx, realConntrackConfigurer{sys: sysctl.New()}, config)
}

// conntrackConfigurer is a mockable interface for setting conntrack sysctls.
//
// Descriptions of the various sysctl fields can be found here:
// https://www.kernel.org/doc/Documentation/networking/nf_conntrack-sysctl.txt
type conntrackConfigurer interface {
	// GetMax gets the current value of nf_conntrack_max.
	GetMax(ctx context.Context) (int, error)
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

	// GetHashsize gets the conntrack module "hashsize" parameter
	GetHashsize(ctx context.Context) (int, error)
	// SetHashsize sets the conntrack module "hashsize" parameter
	SetHashsize(ctx context.Context, value int) error

	// DetectNumCPU returns the number of CPU cores in the system
	DetectNumCPU() int
}

func setSysctls(ctx context.Context, ct conntrackConfigurer, config *kubeproxyconfig.KubeProxyConntrackConfiguration) error {
	max, err := getConntrackMax(ctx, config, ct.DetectNumCPU())
	if err != nil {
		return err
	}
	if max > 0 {
		if curMax, err := ct.GetMax(ctx); err != nil || curMax < max {
			err := ct.SetMax(ctx, max)
			if err != nil {
				return err
			}
		}

		// Check if hashsize is large enough for the nf_conntrack_max value.
		hashsize, err := ct.GetHashsize(ctx)
		if err != nil {
			return err
		}
		if hashsize < max/4 {
			err = ct.SetHashsize(ctx, max/4)
			if err != nil {
				return err
			}
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

func getConntrackMax(ctx context.Context, config *kubeproxyconfig.KubeProxyConntrackConfiguration, numCPU int) (int, error) {
	logger := klog.FromContext(ctx)
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		scaled := int(*config.MaxPerCore) * numCPU
		// Cap the value to 1M to avoid excessive memory usage on high-core machines
		const maxLimit = 1048576
		if scaled > maxLimit {
			logger.V(3).Info("GetConntrackMax: capping scaled conntrack-max-per-core", "scaled", scaled, "limit", maxLimit)
			return maxLimit, nil
		}
		if scaled > floor {
			logger.V(3).Info("GetConntrackMax: using scaled conntrack-max-per-core")
			return scaled, nil
		}
		logger.V(3).Info("GetConntrackMax: using conntrack-min")
		return floor, nil
	}
	return 0, nil
}

type realConntrackConfigurer struct {
	sys sysctl.Interface
}

// DetectNumCPU returns the CPU count used to size nf_conntrack_max. That limit
// is host-wide, so it must be based on the node's CPU count, not runtime.NumCPU():
// runtime.NumCPU() honors the process cpuset and undercounts when kube-proxy
// runs under a static CPU policy. cpuset.NumCPU() reads the node's online CPU
// count from sysfs instead, falling back to runtime.NumCPU() if it can't.
func (rct realConntrackConfigurer) DetectNumCPU() int {
	if n, err := cpuset.NumCPU(); err == nil && n > 0 {
		return n
	}
	return runtime.NumCPU()
}

func (rct realConntrackConfigurer) GetMax(_ context.Context) (int, error) {
	return rct.sys.GetSysctl("net/netfilter/nf_conntrack_max")
}

func (rct realConntrackConfigurer) SetMax(ctx context.Context, max int) error {
	return rct.setIntSysCtl(ctx, "nf_conntrack_max", max)
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

	if val, _ := rct.sys.GetSysctl(entry); val != value {
		logger.Info("Set sysctl", "entry", entry, "value", value)
		if err := rct.sys.SetSysctl(entry, value); err != nil {
			return err
		}
	}
	return nil
}

func (rct realConntrackConfigurer) GetHashsize(_ context.Context) (int, error) {
	b, err := os.ReadFile("/sys/module/nf_conntrack/parameters/hashsize")
	if err != nil {
		return -1, err
	}
	return strconv.Atoi(strings.TrimSpace(string(b)))
}

func (rct realConntrackConfigurer) SetHashsize(ctx context.Context, value int) error {
	klog.FromContext(ctx).Info("Setting conntrack hashsize", "conntrackHashsize", value)
	return os.WriteFile("/sys/module/nf_conntrack/parameters/hashsize", []byte(strconv.Itoa(value)), 0640)
}
