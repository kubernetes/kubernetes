// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import "github.com/miekg/dns"

// exchangeMsg returns a new dns message based on name, type, bufsize and dnssec.
func newExchangeMsg(name string, typ, bufsize uint16, dnssec bool) *dns.Msg {
	m := new(dns.Msg)
	m.SetQuestion(name, typ)
	m.SetEdns0(bufsize, dnssec)
	return m
}

// exchangeWithRetry sends message m to server, but retries on ServerFailure.
func exchangeWithRetry(c *dns.Client, m *dns.Msg, server string) (*dns.Msg, error) {
	r, _, err := c.Exchange(m, server)
	if err == nil && r.Rcode == dns.RcodeServerFailure {
		// redo the query
		r, _, err = c.Exchange(m, server)
	}
	return r, err
}

func (s *server) randomNameserverID(id uint16) int {
	nsid := 0
	if s.config.NSRotate {
		// Use request Id for "random" nameserver selection.
		nsid = int(id) % len(s.config.Nameservers)
	}
	return nsid
}
