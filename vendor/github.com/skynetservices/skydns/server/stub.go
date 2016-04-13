// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"log"
	"net"
	"strconv"
	"strings"

	"github.com/miekg/dns"
	"github.com/skynetservices/skydns/msg"
)

const ednsStubCode = dns.EDNS0LOCALSTART + 10

// ednsStub is the EDNS0 record we add to stub queries. Queries which have this record are
// not forwarded again.
var ednsStub = func() *dns.OPT {
	o := new(dns.OPT)
	o.Hdr.Name = "."
	o.Hdr.Rrtype = dns.TypeOPT
	e := new(dns.EDNS0_LOCAL)
	e.Code = ednsStubCode
	e.Data = []byte{1}
	o.Option = append(o.Option, e)
	return o
}()

// Look in .../dns/stub/<domain>/xx for msg.Services. Loop through them
// extract <domain> and add them as forwarders (ip:port-combos) for
// the stub zones. Only numeric (i.e. IP address) hosts are used.
func (s *server) UpdateStubZones() {
	stubmap := make(map[string][]string)

	services, err := s.backend.Records("stub.dns."+s.config.Domain, false)
	if err != nil {
		log.Printf("skydns: stub zone update failed: %s", err)
		return
	}
	for _, serv := range services {
		if serv.Port == 0 {
			serv.Port = 53
		}
		ip := net.ParseIP(serv.Host)
		if ip == nil {
			log.Printf("skydns: stub zone non-address %s seen for: %s", serv.Key, serv.Host)
			continue
		}

		domain := msg.Domain(serv.Key)
		// Chop of left most label, because that is used as the nameserver place holder
		// and drop the right most labels that belong to localDomain.
		labels := dns.SplitDomainName(domain)
		domain = dns.Fqdn(strings.Join(labels[1:len(labels)-dns.CountLabel(s.config.localDomain)], "."))

		// If the remaining name equals s.config.LocalDomain we ignore it.
		if domain == s.config.localDomain {
			log.Printf("skydns: not adding stub zone for my own domain")
			continue
		}
		stubmap[domain] = append(stubmap[domain], net.JoinHostPort(serv.Host, strconv.Itoa(serv.Port)))
	}

	s.config.stub = &stubmap
}

// ServeDNSStubForward forwards a request to a nameservers and returns the response.
func (s *server) ServeDNSStubForward(w dns.ResponseWriter, req *dns.Msg, ns []string) {
	StatsStubForwardCount.Inc(1)
	promExternalRequestCount.WithLabelValues("stub").Inc()

	// Check EDNS0 Stub option, if set drop the packet.
	option := req.IsEdns0()
	if option != nil {
		for _, o := range option.Option {
			if o.Option() == ednsStubCode && len(o.(*dns.EDNS0_LOCAL).Data) == 1 &&
				o.(*dns.EDNS0_LOCAL).Data[0] == 1 {
				// Maybe log source IP here?
				log.Printf("skydns: not fowarding stub request to another stub")
				return
			}
		}
	}

	tcp := false
	if _, ok := w.RemoteAddr().(*net.TCPAddr); ok {
		tcp = true
	}
	// Add a custom EDNS0 option to the packet, so we can detect loops
	// when 2 stubs are forwarding to each other.
	if option != nil {
		option.Option = append(option.Option, &dns.EDNS0_LOCAL{ednsStubCode, []byte{1}})
	} else {
		req.Extra = append(req.Extra, ednsStub)
	}

	var (
		r   *dns.Msg
		err error
		try int
	)

	// Use request Id for "random" nameserver selection.
	nsid := int(req.Id) % len(ns)
Redo:
	switch tcp {
	case false:
		r, _, err = s.dnsUDPclient.Exchange(req, ns[nsid])
	case true:
		r, _, err = s.dnsTCPclient.Exchange(req, ns[nsid])
	}
	if err == nil {
		r.Compress = true
		r.Id = req.Id
		w.WriteMsg(r)
		return
	}
	// Seen an error, this can only mean, "server not reached", try again
	// but only if we have not exausted our nameservers.
	if try < len(ns) {
		try++
		nsid = (nsid + 1) % len(ns)
		goto Redo
	}

	log.Printf("skydns: failure to forward stub request %q", err)
	m := new(dns.Msg)
	m.SetReply(req)
	m.SetRcode(req, dns.RcodeServerFailure)
	w.WriteMsg(m)
}
