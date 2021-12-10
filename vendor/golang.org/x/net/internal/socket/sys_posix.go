// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows || zos
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows zos

package socket

import (
	"encoding/binary"
	"errors"
	"net"
	"runtime"
	"strconv"
	"sync"
	"time"
)

// marshalInetAddr writes a in sockaddr format into the buffer b.
// The buffer must be sufficiently large (sizeofSockaddrInet4/6).
// Returns the number of bytes written.
func marshalInetAddr(a net.Addr, b []byte) int {
	switch a := a.(type) {
	case *net.TCPAddr:
		return marshalSockaddr(a.IP, a.Port, a.Zone, b)
	case *net.UDPAddr:
		return marshalSockaddr(a.IP, a.Port, a.Zone, b)
	case *net.IPAddr:
		return marshalSockaddr(a.IP, 0, a.Zone, b)
	default:
		return 0
	}
}

func marshalSockaddr(ip net.IP, port int, zone string, b []byte) int {
	if ip4 := ip.To4(); ip4 != nil {
		switch runtime.GOOS {
		case "android", "illumos", "linux", "solaris", "windows":
			NativeEndian.PutUint16(b[:2], uint16(sysAF_INET))
		default:
			b[0] = sizeofSockaddrInet4
			b[1] = sysAF_INET
		}
		binary.BigEndian.PutUint16(b[2:4], uint16(port))
		copy(b[4:8], ip4)
		return sizeofSockaddrInet4
	}
	if ip6 := ip.To16(); ip6 != nil && ip.To4() == nil {
		switch runtime.GOOS {
		case "android", "illumos", "linux", "solaris", "windows":
			NativeEndian.PutUint16(b[:2], uint16(sysAF_INET6))
		default:
			b[0] = sizeofSockaddrInet6
			b[1] = sysAF_INET6
		}
		binary.BigEndian.PutUint16(b[2:4], uint16(port))
		copy(b[8:24], ip6)
		if zone != "" {
			NativeEndian.PutUint32(b[24:28], uint32(zoneCache.index(zone)))
		}
		return sizeofSockaddrInet6
	}
	return 0
}

func parseInetAddr(b []byte, network string) (net.Addr, error) {
	if len(b) < 2 {
		return nil, errors.New("invalid address")
	}
	var af int
	switch runtime.GOOS {
	case "android", "illumos", "linux", "solaris", "windows":
		af = int(NativeEndian.Uint16(b[:2]))
	default:
		af = int(b[1])
	}
	var ip net.IP
	var zone string
	if af == sysAF_INET {
		if len(b) < sizeofSockaddrInet4 {
			return nil, errors.New("short address")
		}
		ip = make(net.IP, net.IPv4len)
		copy(ip, b[4:8])
	}
	if af == sysAF_INET6 {
		if len(b) < sizeofSockaddrInet6 {
			return nil, errors.New("short address")
		}
		ip = make(net.IP, net.IPv6len)
		copy(ip, b[8:24])
		if id := int(NativeEndian.Uint32(b[24:28])); id > 0 {
			zone = zoneCache.name(id)
		}
	}
	switch network {
	case "tcp", "tcp4", "tcp6":
		return &net.TCPAddr{IP: ip, Port: int(binary.BigEndian.Uint16(b[2:4])), Zone: zone}, nil
	case "udp", "udp4", "udp6":
		return &net.UDPAddr{IP: ip, Port: int(binary.BigEndian.Uint16(b[2:4])), Zone: zone}, nil
	default:
		return &net.IPAddr{IP: ip, Zone: zone}, nil
	}
}

// An ipv6ZoneCache represents a cache holding partial network
// interface information. It is used for reducing the cost of IPv6
// addressing scope zone resolution.
//
// Multiple names sharing the index are managed by first-come
// first-served basis for consistency.
type ipv6ZoneCache struct {
	sync.RWMutex                // guard the following
	lastFetched  time.Time      // last time routing information was fetched
	toIndex      map[string]int // interface name to its index
	toName       map[int]string // interface index to its name
}

var zoneCache = ipv6ZoneCache{
	toIndex: make(map[string]int),
	toName:  make(map[int]string),
}

// update refreshes the network interface information if the cache was last
// updated more than 1 minute ago, or if force is set. It returns whether the
// cache was updated.
func (zc *ipv6ZoneCache) update(ift []net.Interface, force bool) (updated bool) {
	zc.Lock()
	defer zc.Unlock()
	now := time.Now()
	if !force && zc.lastFetched.After(now.Add(-60*time.Second)) {
		return false
	}
	zc.lastFetched = now
	if len(ift) == 0 {
		var err error
		if ift, err = net.Interfaces(); err != nil {
			return false
		}
	}
	zc.toIndex = make(map[string]int, len(ift))
	zc.toName = make(map[int]string, len(ift))
	for _, ifi := range ift {
		zc.toIndex[ifi.Name] = ifi.Index
		if _, ok := zc.toName[ifi.Index]; !ok {
			zc.toName[ifi.Index] = ifi.Name
		}
	}
	return true
}

func (zc *ipv6ZoneCache) name(zone int) string {
	updated := zoneCache.update(nil, false)
	zoneCache.RLock()
	name, ok := zoneCache.toName[zone]
	zoneCache.RUnlock()
	if !ok && !updated {
		zoneCache.update(nil, true)
		zoneCache.RLock()
		name, ok = zoneCache.toName[zone]
		zoneCache.RUnlock()
	}
	if !ok { // last resort
		name = strconv.Itoa(zone)
	}
	return name
}

func (zc *ipv6ZoneCache) index(zone string) int {
	updated := zoneCache.update(nil, false)
	zoneCache.RLock()
	index, ok := zoneCache.toIndex[zone]
	zoneCache.RUnlock()
	if !ok && !updated {
		zoneCache.update(nil, true)
		zoneCache.RLock()
		index, ok = zoneCache.toIndex[zone]
		zoneCache.RUnlock()
	}
	if !ok { // last resort
		index, _ = strconv.Atoi(zone)
	}
	return index
}
