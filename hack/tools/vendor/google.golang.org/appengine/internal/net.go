// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

// This file implements a network dialer that limits the number of concurrent connections.
// It is only used for API calls.

import (
	"log"
	"net"
	"runtime"
	"sync"
	"time"
)

var limitSem = make(chan int, 100) // TODO(dsymonds): Use environment variable.

func limitRelease() {
	// non-blocking
	select {
	case <-limitSem:
	default:
		// This should not normally happen.
		log.Print("appengine: unbalanced limitSem release!")
	}
}

func limitDial(network, addr string) (net.Conn, error) {
	limitSem <- 1

	// Dial with a timeout in case the API host is MIA.
	// The connection should normally be very fast.
	conn, err := net.DialTimeout(network, addr, 10*time.Second)
	if err != nil {
		limitRelease()
		return nil, err
	}
	lc := &limitConn{Conn: conn}
	runtime.SetFinalizer(lc, (*limitConn).Close) // shouldn't usually be required
	return lc, nil
}

type limitConn struct {
	close sync.Once
	net.Conn
}

func (lc *limitConn) Close() error {
	defer lc.close.Do(func() {
		limitRelease()
		runtime.SetFinalizer(lc, nil)
	})
	return lc.Conn.Close()
}
