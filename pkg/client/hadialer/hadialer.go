/*
Copyright 2014 The Kubernetes Authors.

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

package hadialer

import (
	"context"
	"math/rand"
	"net"
	"time"

	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

type DialFunc func(ctx context.Context, network, address string) (net.Conn, error)

type Dialer struct {
	DialFunc
}

func OverrideDial(clientConfig *restclient.Config) {
	klog.Info("wrapping client's dial with hadialer")

	dial := clientConfig.Dial
	if dial == nil {
		dial = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	}

	dialer := Dialer{dial}
	clientConfig.Dial = dialer.Dial
}

func (d Dialer) Dial(ctx context.Context, network, addr string) (conn net.Conn, err error) {
	// at least verbosity 1 to get logs here
	klog := klog.V(1)

	klog.Infof("dial %q %q", network, addr)

	dial := d.DialFunc

	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		// failed to split host and port, fallback to default behaviour
		return dial(ctx, network, addr)
	}

	addrs, err := net.LookupHost(host)

	klog.Infof("lookup host resolved %q to %v", host, addrs)

	if err != nil || len(addrs) == 0 {
		// failed to lookup host, fallback to default behaviour
		return dial(ctx, network, addr)
	}

	// randomize address order
	rand.Shuffle(len(addrs), func(i, j int) {
		addrs[i], addrs[j] = addrs[j], addrs[i]
	})

	for _, addr := range addrs {
		addr := net.JoinHostPort(addr, port)

		conn, err = dial(ctx, network, addr)
		if err != nil {
			klog.Info("dial to ", addr, " failed: ", err)
			continue
		}

		// dial successful, return the conn
		klog.Info("dial to ", addr, " succeeded")
		return
	}

	// no dial target worked, return the last error
	return
}
