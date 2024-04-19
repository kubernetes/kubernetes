//go:build !linux
// +build !linux

package network

import (
	"net"
	"time"

	"k8s.io/klog/v2"
)

func dialerWithDefaultOptions() DialContext {
	klog.V(2).Info("Creating the default network Dialer (unsupported platform). It may take up to 15 minutes to detect broken connections and establish a new one")
	nd := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	return nd.DialContext
}
