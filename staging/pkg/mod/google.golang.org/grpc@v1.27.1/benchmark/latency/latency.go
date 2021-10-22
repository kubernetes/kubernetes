/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package latency provides wrappers for net.Conn, net.Listener, and
// net.Dialers, designed to interoperate to inject real-world latency into
// network connections.
package latency

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"time"
)

// Dialer is a function matching the signature of net.Dial.
type Dialer func(network, address string) (net.Conn, error)

// TimeoutDialer is a function matching the signature of net.DialTimeout.
type TimeoutDialer func(network, address string, timeout time.Duration) (net.Conn, error)

// ContextDialer is a function matching the signature of
// net.Dialer.DialContext.
type ContextDialer func(ctx context.Context, network, address string) (net.Conn, error)

// Network represents a network with the given bandwidth, latency, and MTU
// (Maximum Transmission Unit) configuration, and can produce wrappers of
// net.Listeners, net.Conn, and various forms of dialing functions.  The
// Listeners and Dialers/Conns on both sides of connections must come from this
// package, but need not be created from the same Network.  Latency is computed
// when sending (in Write), and is injected when receiving (in Read).  This
// allows senders' Write calls to be non-blocking, as in real-world
// applications.
//
// Note: Latency is injected by the sender specifying the absolute time data
// should be available, and the reader delaying until that time arrives to
// provide the data.  This package attempts to counter-act the effects of clock
// drift and existing network latency by measuring the delay between the
// sender's transmission time and the receiver's reception time during startup.
// No attempt is made to measure the existing bandwidth of the connection.
type Network struct {
	Kbps    int           // Kilobits per second; if non-positive, infinite
	Latency time.Duration // One-way latency (sending); if non-positive, no delay
	MTU     int           // Bytes per packet; if non-positive, infinite
}

var (
	//Local simulates local network.
	Local = Network{0, 0, 0}
	//LAN simulates local area network network.
	LAN = Network{100 * 1024, 2 * time.Millisecond, 1500}
	//WAN simulates wide area network.
	WAN = Network{20 * 1024, 30 * time.Millisecond, 1500}
	//Longhaul simulates bad network.
	Longhaul = Network{1000 * 1024, 200 * time.Millisecond, 9000}
)

// Conn returns a net.Conn that wraps c and injects n's latency into that
// connection.  This function also imposes latency for connection creation.
// If n's Latency is lower than the measured latency in c, an error is
// returned.
func (n *Network) Conn(c net.Conn) (net.Conn, error) {
	start := now()
	nc := &conn{Conn: c, network: n, readBuf: new(bytes.Buffer)}
	if err := nc.sync(); err != nil {
		return nil, err
	}
	sleep(start.Add(nc.delay).Sub(now()))
	return nc, nil
}

type conn struct {
	net.Conn
	network *Network

	readBuf     *bytes.Buffer // one packet worth of data received
	lastSendEnd time.Time     // time the previous Write should be fully on the wire
	delay       time.Duration // desired latency - measured latency
}

// header is sent before all data transmitted by the application.
type header struct {
	ReadTime int64 // Time the reader is allowed to read this packet (UnixNano)
	Sz       int32 // Size of the data in the packet
}

func (c *conn) Write(p []byte) (n int, err error) {
	tNow := now()
	if c.lastSendEnd.Before(tNow) {
		c.lastSendEnd = tNow
	}
	for len(p) > 0 {
		pkt := p
		if c.network.MTU > 0 && len(pkt) > c.network.MTU {
			pkt = pkt[:c.network.MTU]
			p = p[c.network.MTU:]
		} else {
			p = nil
		}
		if c.network.Kbps > 0 {
			if congestion := c.lastSendEnd.Sub(tNow) - c.delay; congestion > 0 {
				// The network is full; sleep until this packet can be sent.
				sleep(congestion)
				tNow = tNow.Add(congestion)
			}
		}
		c.lastSendEnd = c.lastSendEnd.Add(c.network.pktTime(len(pkt)))
		hdr := header{ReadTime: c.lastSendEnd.Add(c.delay).UnixNano(), Sz: int32(len(pkt))}
		if err := binary.Write(c.Conn, binary.BigEndian, hdr); err != nil {
			return n, err
		}
		x, err := c.Conn.Write(pkt)
		n += x
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func (c *conn) Read(p []byte) (n int, err error) {
	if c.readBuf.Len() == 0 {
		var hdr header
		if err := binary.Read(c.Conn, binary.BigEndian, &hdr); err != nil {
			return 0, err
		}
		defer func() { sleep(time.Unix(0, hdr.ReadTime).Sub(now())) }()

		if _, err := io.CopyN(c.readBuf, c.Conn, int64(hdr.Sz)); err != nil {
			return 0, err
		}
	}
	// Read from readBuf.
	return c.readBuf.Read(p)
}

// sync does a handshake and then measures the latency on the network in
// coordination with the other side.
func (c *conn) sync() error {
	const (
		pingMsg  = "syncPing"
		warmup   = 10               // minimum number of iterations to measure latency
		giveUp   = 50               // maximum number of iterations to measure latency
		accuracy = time.Millisecond // req'd accuracy to stop early
		goodRun  = 3                // stop early if latency within accuracy this many times
	)

	type syncMsg struct {
		SendT int64 // Time sent.  If zero, stop.
		RecvT int64 // Time received.  If zero, fill in and respond.
	}

	// A trivial handshake
	if err := binary.Write(c.Conn, binary.BigEndian, []byte(pingMsg)); err != nil {
		return err
	}
	var ping [8]byte
	if err := binary.Read(c.Conn, binary.BigEndian, &ping); err != nil {
		return err
	} else if string(ping[:]) != pingMsg {
		return fmt.Errorf("malformed handshake message: %v (want %q)", ping, pingMsg)
	}

	// Both sides are alive and syncing.  Calculate network delay / clock skew.
	att := 0
	good := 0
	var latency time.Duration
	localDone, remoteDone := false, false
	send := true
	for !localDone || !remoteDone {
		if send {
			if err := binary.Write(c.Conn, binary.BigEndian, syncMsg{SendT: now().UnixNano()}); err != nil {
				return err
			}
			att++
			send = false
		}

		// Block until we get a syncMsg
		m := syncMsg{}
		if err := binary.Read(c.Conn, binary.BigEndian, &m); err != nil {
			return err
		}

		if m.RecvT == 0 {
			// Message initiated from other side.
			if m.SendT == 0 {
				remoteDone = true
				continue
			}
			// Send response.
			m.RecvT = now().UnixNano()
			if err := binary.Write(c.Conn, binary.BigEndian, m); err != nil {
				return err
			}
			continue
		}

		lag := time.Duration(m.RecvT - m.SendT)
		latency += lag
		avgLatency := latency / time.Duration(att)
		if e := lag - avgLatency; e > -accuracy && e < accuracy {
			good++
		} else {
			good = 0
		}
		if att < giveUp && (att < warmup || good < goodRun) {
			send = true
			continue
		}
		localDone = true
		latency = avgLatency
		// Tell the other side we're done.
		if err := binary.Write(c.Conn, binary.BigEndian, syncMsg{}); err != nil {
			return err
		}
	}
	if c.network.Latency <= 0 {
		return nil
	}
	c.delay = c.network.Latency - latency
	if c.delay < 0 {
		return fmt.Errorf("measured network latency (%v) higher than desired latency (%v)", latency, c.network.Latency)
	}
	return nil
}

// Listener returns a net.Listener that wraps l and injects n's latency in its
// connections.
func (n *Network) Listener(l net.Listener) net.Listener {
	return &listener{Listener: l, network: n}
}

type listener struct {
	net.Listener
	network *Network
}

func (l *listener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return l.network.Conn(c)
}

// Dialer returns a Dialer that wraps d and injects n's latency in its
// connections.  n's Latency is also injected to the connection's creation.
func (n *Network) Dialer(d Dialer) Dialer {
	return func(network, address string) (net.Conn, error) {
		conn, err := d(network, address)
		if err != nil {
			return nil, err
		}
		return n.Conn(conn)
	}
}

// TimeoutDialer returns a TimeoutDialer that wraps d and injects n's latency
// in its connections.  n's Latency is also injected to the connection's
// creation.
func (n *Network) TimeoutDialer(d TimeoutDialer) TimeoutDialer {
	return func(network, address string, timeout time.Duration) (net.Conn, error) {
		conn, err := d(network, address, timeout)
		if err != nil {
			return nil, err
		}
		return n.Conn(conn)
	}
}

// ContextDialer returns a ContextDialer that wraps d and injects n's latency
// in its connections.  n's Latency is also injected to the connection's
// creation.
func (n *Network) ContextDialer(d ContextDialer) ContextDialer {
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := d(ctx, network, address)
		if err != nil {
			return nil, err
		}
		return n.Conn(conn)
	}
}

// pktTime returns the time it takes to transmit one packet of data of size b
// in bytes.
func (n *Network) pktTime(b int) time.Duration {
	if n.Kbps <= 0 {
		return time.Duration(0)
	}
	return time.Duration(b) * time.Second / time.Duration(n.Kbps*(1024/8))
}

// Wrappers for testing

var now = time.Now
var sleep = time.Sleep
