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
	"net/netip"
	"time"

	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

type DialFunc func(ctx context.Context, network, address string) (net.Conn, error)

type Dialer DialFunc

func OverrideDial(clientConfig *restclient.Config) {
	dial := clientConfig.Dial
	if dial == nil {
		dial = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	}

	dialer := Dialer(dial)
	clientConfig.Dial = dialer.Dial
}

func (d Dialer) Dial(ctx context.Context, network, addr string) (net.Conn, error) {
	// at least verbosity 1 to get logs here
	klog := klog.V(1)

	klog.Infof("dial %q %q", network, addr)

	dial := DialFunc(d)

	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		// failed to split host and port, fallback to default behaviour
		return dial(ctx, network, addr)
	}

	addrs, err := net.DefaultResolver.LookupHost(ctx, host)

	if err != nil || len(addrs) == 0 {
		// failed to lookup host, fallback to default behaviour
		return dial(ctx, network, addr)
	}

	klog.Infof("lookup host resolved %q to %v", host, addrs)

	if len(addrs) == 1 {
		// host resolved to only one address, avoid the parallel dial overload for the simple case
		return dial(ctx, network, addrs[0])
	}

	// randomize address order
	rand.Shuffle(len(addrs), func(i, j int) {
		addrs[i], addrs[j] = addrs[j], addrs[i]
	})

	// interleave v4 and v6 addresses to handle the case where one of those networks timeouts but not the other
	interleaveIPV4AndV6(&addrs)

	// the chan to notify dial routines that we've returned
	returned := make(chan struct{})
	defer close(returned)

	// dial results are sent to an unbuffered chan
	type dialResult struct {
		addr string
		net.Conn
		error
	}
	results := make(chan dialResult) // unbuffered

	// dial a specific address, shipping the result.
	// Closes the connection if dial succeeded but we've returned since then.
	dialAddr := func(ctx context.Context, addr string) {
		c, err := dial(ctx, network, addr)
		select {
		case results <- dialResult{addr: addr, Conn: c, error: err}:
		case <-returned:
			if c != nil {
				_ = c.Close()
			}
		}
	}

	var (
		running    int
		firstError error
	)

	// Handle a dial result:
	// - returns the connection if dial succeeded;
	// - sets firstError on the first error received.
	handleResult := func(res dialResult) (conn net.Conn) {
		running--

		switch {
		case res.error != nil:
			klog.Info("dial to ", res.addr, " failed: ", res.error)

			if firstError == nil {
				firstError = res.error
			}

		case res.Conn != nil:
			klog.Info("dial to ", res.addr, " succeeded")
			conn = res.Conn
		}
		return
	}

	// returns the connection if a dial succeeded within the try delay.
	waitForResult := func() net.Conn {
		// start the timer for giving up
		timer := time.NewTimer(300 * time.Millisecond)
		defer timer.Stop()

		// wait for the timer unless no dial is running anymore
		for running != 0 {
			select {
			case <-timer.C:
				// no result within the delay, give up waiting
				return nil
			case res := <-results:
				conn := handleResult(res)
				if conn != nil {
					return conn
				}
			}
		}

		return nil
	}

	// try every address until one succeed
	for _, addr := range addrs {
		addr := net.JoinHostPort(addr, port)

		ctx, ctxCancel := context.WithCancel(ctx)
		defer ctxCancel()

		running++
		go dialAddr(ctx, addr)

		if conn := waitForResult(); conn != nil {
			return conn, nil
		}
	}

	for running != 0 {
		res := <-results
		if conn := handleResult(res); conn != nil {
			return conn, nil
		}
	}

	// no dial succeeded, return the first error
	return nil, firstError
}

// Interleave v4 and v6 addresses in the given slice.
func interleaveIPV4AndV6(addrs *[]string) {
	v4 := make([]string, 0, len(*addrs))
	v6 := make([]string, 0, len(*addrs))

	for _, addr := range *addrs {
		ip, err := netip.ParseAddr(addr)
		if err != nil {
			klog.ErrorS(err, "Invalid address ignored", "address", addr)
			continue
		}

		if ip.Is4() {
			v4 = append(v4, addr)
		} else {
			v6 = append(v6, addr)
		}
	}

	maxLen := len(v4)
	if len(v6) > maxLen {
		maxLen = len(v6)
	}

	*addrs = (*addrs)[:0]
	for i := 0; i != maxLen; i++ {
		if i < len(v4) {
			*addrs = append(*addrs, v4[i])
		}
		if i < len(v6) {
			*addrs = append(*addrs, v6[i])
		}
	}
}
