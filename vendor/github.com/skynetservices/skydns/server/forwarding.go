// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"fmt"

	"github.com/miekg/dns"
)

// ServeDNSForward forwards a request to a nameservers and returns the response.
func (s *server) ServeDNSForward(w dns.ResponseWriter, req *dns.Msg) *dns.Msg {
	if s.config.NoRec {
		m := s.ServerFailure(req)
		w.WriteMsg(m)
		return m
	}

	if len(s.config.Nameservers) == 0 || dns.CountLabel(req.Question[0].Name) < s.config.Ndots {
		if s.config.Verbose {
			if len(s.config.Nameservers) == 0 {
				logf("can not forward, no nameservers defined")
			} else {
				logf("can not forward, name too short (less than %d labels): `%s'", s.config.Ndots, req.Question[0].Name)
			}
		}
		m := s.ServerFailure(req)
		m.RecursionAvailable = true // this is still true
		w.WriteMsg(m)
		return m
	}

	var (
		r   *dns.Msg
		err error
	)

	nsid := s.randomNameserverID(req.Id)
	try := 0
Redo:
	if isTCP(w) {
		r, err = exchangeWithRetry(s.dnsTCPclient, req, s.config.Nameservers[nsid])
	} else {
		r, err = exchangeWithRetry(s.dnsUDPclient, req, s.config.Nameservers[nsid])
	}
	if err == nil {
		r.Compress = true
		r.Id = req.Id
		w.WriteMsg(r)
		return r
	}
	// Seen an error, this can only mean, "server not reached", try again
	// but only if we have not exausted our nameservers.
	if try < len(s.config.Nameservers) {
		try++
		nsid = (nsid + 1) % len(s.config.Nameservers)
		goto Redo
	}

	logf("failure to forward request %q", err)
	m := s.ServerFailure(req)
	return m
}

// ServeDNSReverse is the handler for DNS requests for the reverse zone. If nothing is found
// locally the request is forwarded to the forwarder for resolution.
func (s *server) ServeDNSReverse(w dns.ResponseWriter, req *dns.Msg) *dns.Msg {
	m := new(dns.Msg)
	m.SetReply(req)
	m.Compress = true
	m.Authoritative = false // Set to false, because I don't know what to do wrt DNSSEC.
	m.RecursionAvailable = true
	var err error
	if m.Answer, err = s.PTRRecords(req.Question[0]); err == nil {
		// TODO(miek): Reverse DNSSEC. We should sign this, but requires a key....and more
		// Probably not worth the hassle?
		if err := w.WriteMsg(m); err != nil {
			logf("failure to return reply %q", err)
		}
		return m
	}
	// Always forward if not found locally.
	return s.ServeDNSForward(w, req)
}

// Lookup looks up name,type using the recursive nameserver defines
// in the server's config. If none defined it returns an error.
func (s *server) Lookup(n string, t, bufsize uint16, dnssec bool) (*dns.Msg, error) {
	if len(s.config.Nameservers) == 0 {
		return nil, fmt.Errorf("no nameservers configured can not lookup name")
	}
	if dns.CountLabel(n) < s.config.Ndots {
		return nil, fmt.Errorf("name has fewer than %d labels", s.config.Ndots)
	}
	m := newExchangeMsg(n, t, bufsize, dnssec)

	nsid := s.randomNameserverID(m.Id)
	try := 0
Redo:
	r, err := exchangeWithRetry(s.dnsUDPclient, m, s.config.Nameservers[nsid])
	if err == nil {
		if r.Rcode != dns.RcodeSuccess {
			return nil, fmt.Errorf("rcode %d is not equal to success", r.Rcode)
		}
		// Reset TTLs to rcache TTL to make some of the other code
		// and the tests not care about TTLs
		for _, rr := range r.Answer {
			rr.Header().Ttl = uint32(s.config.RCacheTtl)
		}
		for _, rr := range r.Extra {
			rr.Header().Ttl = uint32(s.config.RCacheTtl)
		}
		return r, nil
	}
	// Seen an error, this can only mean, "server not reached", try again
	// but only if we have not exausted our nameservers.
	if try < len(s.config.Nameservers) {
		try++
		nsid = (nsid + 1) % len(s.config.Nameservers)
		goto Redo
	}
	return nil, fmt.Errorf("failure to lookup name")
}
