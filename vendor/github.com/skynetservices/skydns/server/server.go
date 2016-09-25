// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"fmt"
	"math"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/skynetservices/skydns/cache"
	"github.com/skynetservices/skydns/metrics"
	"github.com/skynetservices/skydns/msg"

	etcd "github.com/coreos/etcd/client"
	"github.com/coreos/go-systemd/activation"
	"github.com/miekg/dns"
)

const Version = "2.5.3a"

type server struct {
	backend Backend
	config  *Config

	group        *sync.WaitGroup
	dnsUDPclient *dns.Client // used for forwarding queries
	dnsTCPclient *dns.Client // used for forwarding queries
	scache       *cache.Cache
	rcache       *cache.Cache
}

// New returns a new SkyDNS server.
func New(backend Backend, config *Config) *server {
	return &server{
		backend: backend,
		config:  config,

		group:        new(sync.WaitGroup),
		scache:       cache.New(config.SCache, 0),
		rcache:       cache.New(config.RCache, config.RCacheTtl),
		dnsUDPclient: &dns.Client{Net: "udp", ReadTimeout: config.ReadTimeout, WriteTimeout: config.ReadTimeout, SingleInflight: true},
		dnsTCPclient: &dns.Client{Net: "tcp", ReadTimeout: config.ReadTimeout, WriteTimeout: config.ReadTimeout, SingleInflight: true},
	}
}

// Run is a blocking operation that starts the server listening on the DNS ports.
func (s *server) Run() error {
	mux := dns.NewServeMux()
	mux.Handle(".", s)

	dnsReadyMsg := func(addr, net string) {
		if s.config.DNSSEC == "" {
			logf("ready for queries on %s for %s://%s [rcache %d]", s.config.Domain, net, addr, s.config.RCache)
		} else {
			logf("ready for queries on %s for %s://%s [rcache %d], signing with %s [scache %d]", s.config.Domain, net, addr, s.config.RCache, s.config.DNSSEC, s.config.SCache)
		}
	}

	if s.config.Systemd {
		packetConns, err := activation.PacketConns(false)
		if err != nil {
			return err
		}
		listeners, err := activation.Listeners(true)
		if err != nil {
			return err
		}
		if len(packetConns) == 0 && len(listeners) == 0 {
			return fmt.Errorf("no UDP or TCP sockets supplied by systemd")
		}
		for _, p := range packetConns {
			if u, ok := p.(*net.UDPConn); ok {
				s.group.Add(1)
				go func() {
					defer s.group.Done()
					if err := dns.ActivateAndServe(nil, u, mux); err != nil {
						fatalf("%s", err)
					}
				}()
				dnsReadyMsg(u.LocalAddr().String(), "udp")
			}
		}
		for _, l := range listeners {
			if t, ok := l.(*net.TCPListener); ok {
				s.group.Add(1)
				go func() {
					defer s.group.Done()
					if err := dns.ActivateAndServe(t, nil, mux); err != nil {
						fatalf("%s", err)
					}
				}()
				dnsReadyMsg(t.Addr().String(), "tcp")
			}
		}
	} else {
		s.group.Add(1)
		go func() {
			defer s.group.Done()
			if err := dns.ListenAndServe(s.config.DnsAddr, "tcp", mux); err != nil {
				fatalf("%s", err)
			}
		}()
		dnsReadyMsg(s.config.DnsAddr, "tcp")
		s.group.Add(1)
		go func() {
			defer s.group.Done()
			if err := dns.ListenAndServe(s.config.DnsAddr, "udp", mux); err != nil {
				fatalf("%s", err)
			}
		}()
		dnsReadyMsg(s.config.DnsAddr, "udp")
	}

	s.group.Wait()
	return nil
}

// Stop stops a server.
func (s *server) Stop() {
	// TODO(miek)
	//s.group.Add(-2)
}

// ServeDNS is the handler for DNS requests, responsible for parsing DNS request, possibly forwarding
// it to a real dns server and returning a response.
func (s *server) ServeDNS(w dns.ResponseWriter, req *dns.Msg) {
	m := new(dns.Msg)
	m.SetReply(req)
	m.Authoritative = true
	m.RecursionAvailable = true
	m.Compress = true

	bufsize := uint16(512)
	dnssec := false
	tcp := false
	start := time.Now()

	q := req.Question[0]
	name := strings.ToLower(q.Name)

	if q.Qtype == dns.TypeANY {
		m.Authoritative = false
		m.Rcode = dns.RcodeRefused
		m.RecursionAvailable = false
		m.RecursionDesired = false
		m.Compress = false
		w.WriteMsg(m)

		metrics.ReportRequestCount(m, metrics.Auth)
		metrics.ReportDuration(m, start, metrics.Auth)
		metrics.ReportErrorCount(m, metrics.Auth)

		return
	}

	if o := req.IsEdns0(); o != nil {
		bufsize = o.UDPSize()
		dnssec = o.Do()
	}
	if bufsize < 512 {
		bufsize = 512
	}
	// with TCP we can send 64K
	if tcp = isTCP(w); tcp {
		bufsize = dns.MaxMsgSize - 1
	}

	if s.config.Verbose {
		logf("received DNS Request for %q from %q with type %d", q.Name, w.RemoteAddr(), q.Qtype)
	}

	// Check cache first.
	m1 := s.rcache.Hit(q, dnssec, tcp, m.Id)
	if m1 != nil {
		metrics.ReportRequestCount(req, metrics.Cache)

		if send := s.overflowOrTruncated(w, m1, int(bufsize), metrics.Cache); send {
			return
		}

		// Still round-robin even with hits from the cache.
		// Only shuffle A and AAAA records with each other.
		if q.Qtype == dns.TypeA || q.Qtype == dns.TypeAAAA {
			s.RoundRobin(m1.Answer)
		}

		if err := w.WriteMsg(m1); err != nil {
			logf("failure to return reply %q", err)
		}

		metrics.ReportDuration(m1, start, metrics.Cache)
		metrics.ReportErrorCount(m1, metrics.Cache)
		return
	}

	for zone, ns := range *s.config.stub {
		if strings.HasSuffix(name, "." + zone) || name == zone {
			metrics.ReportRequestCount(req, metrics.Stub)

			resp := s.ServeDNSStubForward(w, req, ns)
			if resp != nil {
				s.rcache.InsertMessage(cache.Key(q, dnssec, tcp), resp)
			}

			metrics.ReportDuration(resp, start, metrics.Stub)
			metrics.ReportErrorCount(resp, metrics.Stub)
			return
		}
	}

	// If the qname is local.ds.skydns.local. and s.config.Local != "", substitute that name.
	if s.config.Local != "" && name == s.config.localDomain {
		name = s.config.Local
	}

	if q.Qtype == dns.TypePTR && strings.HasSuffix(name, ".in-addr.arpa.") || strings.HasSuffix(name, ".ip6.arpa.") {
		metrics.ReportRequestCount(req, metrics.Reverse)

		resp := s.ServeDNSReverse(w, req)
		if resp != nil {
			s.rcache.InsertMessage(cache.Key(q, dnssec, tcp), resp)
		}

		metrics.ReportDuration(resp, start, metrics.Reverse)
		metrics.ReportErrorCount(resp, metrics.Reverse)
		return
	}

	if q.Qclass != dns.ClassCHAOS && !strings.HasSuffix(name, "." +s.config.Domain) && name != s.config.Domain {
		metrics.ReportRequestCount(req, metrics.Rec)

		resp := s.ServeDNSForward(w, req)
		if resp != nil {
			s.rcache.InsertMessage(cache.Key(q, dnssec, tcp), resp)
		}

		metrics.ReportDuration(resp, start, metrics.Rec)
		metrics.ReportErrorCount(resp, metrics.Rec)
		return
	}

	metrics.ReportCacheMiss(metrics.Response)

	defer func() {
		metrics.ReportDuration(m, start, metrics.Auth)
		metrics.ReportErrorCount(m, metrics.Auth)

		if m.Rcode == dns.RcodeServerFailure {
			if err := w.WriteMsg(m); err != nil {
				logf("failure to return reply %q", err)
			}
			return
		}
		// Set TTL to the minimum of the RRset and dedup the message, i.e. remove identical RRs.
		m = s.dedup(m)

		minttl := s.config.Ttl
		if len(m.Answer) > 1 {
			for _, r := range m.Answer {
				if r.Header().Ttl < minttl {
					minttl = r.Header().Ttl
				}
			}
			for _, r := range m.Answer {
				r.Header().Ttl = minttl
			}
		}

		if dnssec {
			if s.config.PubKey != nil {
				m.AuthenticatedData = true
				s.Denial(m)
				s.Sign(m, bufsize)
			}
		}

		if send := s.overflowOrTruncated(w, m, int(bufsize), metrics.Auth); send {
			return
		}

		s.rcache.InsertMessage(cache.Key(q, dnssec, tcp), m)

		if err := w.WriteMsg(m); err != nil {
			logf("failure to return reply %q", err)
		}
	}()

	if name == s.config.Domain {
		if q.Qtype == dns.TypeSOA {
			m.Answer = []dns.RR{s.NewSOA()}
			return
		}
		if q.Qtype == dns.TypeDNSKEY {
			if s.config.PubKey != nil {
				m.Answer = []dns.RR{s.config.PubKey}
				return
			}
		}
	}
	if q.Qclass == dns.ClassCHAOS {
		if q.Qtype == dns.TypeTXT {
			switch name {
			case "authors.bind.":
				fallthrough
			case s.config.Domain:
				hdr := dns.RR_Header{Name: q.Name, Rrtype: dns.TypeTXT, Class: dns.ClassCHAOS, Ttl: 0}
				authors := []string{"Erik St. Martin", "Brian Ketelsen", "Miek Gieben", "Michael Crosby"}
				for _, a := range authors {
					m.Answer = append(m.Answer, &dns.TXT{Hdr: hdr, Txt: []string{a}})
				}
				for j := 0; j < len(authors)*(int(dns.Id())%4+1); j++ {
					q := int(dns.Id()) % len(authors)
					p := int(dns.Id()) % len(authors)
					if q == p {
						p = (p + 1) % len(authors)
					}
					m.Answer[q], m.Answer[p] = m.Answer[p], m.Answer[q]
				}
				return
			case "version.bind.":
				fallthrough
			case "version.server.":
				hdr := dns.RR_Header{Name: q.Name, Rrtype: dns.TypeTXT, Class: dns.ClassCHAOS, Ttl: 0}
				m.Answer = []dns.RR{&dns.TXT{Hdr: hdr, Txt: []string{Version}}}
				return
			case "hostname.bind.":
				fallthrough
			case "id.server.":
				// TODO(miek): machine name to return
				hdr := dns.RR_Header{Name: q.Name, Rrtype: dns.TypeTXT, Class: dns.ClassCHAOS, Ttl: 0}
				m.Answer = []dns.RR{&dns.TXT{Hdr: hdr, Txt: []string{"localhost"}}}
				return
			}
		}
		// still here, fail
		m.SetReply(req)
		m.SetRcode(req, dns.RcodeServerFailure)
		return
	}

	switch q.Qtype {
	case dns.TypeNS:
		if name != s.config.Domain {
			break
		}
		// Lookup s.config.DnsDomain
		records, extra, err := s.NSRecords(q, s.config.dnsDomain)
		if isEtcdNameError(err, s) {
			m = s.NameError(req)
			return
		}
		m.Answer = append(m.Answer, records...)
		m.Extra = append(m.Extra, extra...)
	case dns.TypeA, dns.TypeAAAA:
		records, err := s.AddressRecords(q, name, nil, bufsize, dnssec, false)
		if isEtcdNameError(err, s) {
			m = s.NameError(req)
			return
		}
		m.Answer = append(m.Answer, records...)
	case dns.TypeTXT:
		records, err := s.TXTRecords(q, name)
		if isEtcdNameError(err, s) {
			m = s.NameError(req)
			return
		}
		m.Answer = append(m.Answer, records...)
	case dns.TypeCNAME:
		records, err := s.CNAMERecords(q, name)
		if isEtcdNameError(err, s) {
			m = s.NameError(req)
			return
		}
		m.Answer = append(m.Answer, records...)
	case dns.TypeMX:
		records, extra, err := s.MXRecords(q, name, bufsize, dnssec)
		if isEtcdNameError(err, s) {
			m = s.NameError(req)
			return
		}
		m.Answer = append(m.Answer, records...)
		m.Extra = append(m.Extra, extra...)
	default:
		fallthrough // also catch other types, so that they return NODATA
	case dns.TypeSRV:
		records, extra, err := s.SRVRecords(q, name, bufsize, dnssec)
		if err != nil {
			if isEtcdNameError(err, s) {
				m = s.NameError(req)
				return
			}
			logf("got error from backend: %s", err)
			if q.Qtype == dns.TypeSRV { // Otherwise NODATA
				m = s.ServerFailure(req)
				return
			}
		}
		// if we are here again, check the types, because an answer may only
		// be given for SRV. All other types should return NODATA, the
		// NXDOMAIN part is handled in the above code. TODO(miek): yes this
		// can be done in a more elegant manor.
		if q.Qtype == dns.TypeSRV {
			m.Answer = append(m.Answer, records...)
			m.Extra = append(m.Extra, extra...)
		}
	}

	if len(m.Answer) == 0 { // NODATA response
		m.Ns = []dns.RR{s.NewSOA()}
		m.Ns[0].Header().Ttl = s.config.MinTtl
	}
}

func (s *server) AddressRecords(q dns.Question, name string, previousRecords []dns.RR, bufsize uint16, dnssec, both bool) (records []dns.RR, err error) {
	services, err := s.backend.Records(name, false)
	if err != nil {
		return nil, err
	}

	services = msg.Group(services)

	for _, serv := range services {
		ip := net.ParseIP(serv.Host)
		switch {
		case ip == nil:
			// Try to resolve as CNAME if it's not an IP, but only if we don't create loops.
			if q.Name == dns.Fqdn(serv.Host) {
				logf("CNAME loop detected: %q -> %q", q.Name, q.Name)
				// x CNAME x is a direct loop, don't add those
				continue
			}

			newRecord := serv.NewCNAME(q.Name, dns.Fqdn(serv.Host))
			if len(previousRecords) > 7 {
				logf("CNAME lookup limit of 8 exceeded for %s", newRecord)
				// don't add it, and just continue
				continue
			}
			if s.isDuplicateCNAME(newRecord, previousRecords) {
				logf("CNAME loop detected for record %s", newRecord)
				continue
			}

			nextRecords, err := s.AddressRecords(dns.Question{Name: dns.Fqdn(serv.Host), Qtype: q.Qtype, Qclass: q.Qclass},
				strings.ToLower(dns.Fqdn(serv.Host)), append(previousRecords, newRecord), bufsize, dnssec, both)
			if err == nil {
				// Only have we found something we should add the CNAME and the IP addresses.
				if len(nextRecords) > 0 {
					records = append(records, newRecord)
					records = append(records, nextRecords...)
				}
				continue
			}
			// This means we can not complete the CNAME, try to look else where.
			target := newRecord.Target
			if dns.IsSubDomain(s.config.Domain, target) {
				// We should already have found it
				continue
			}
			m1, e1 := s.Lookup(target, q.Qtype, bufsize, dnssec)
			if e1 != nil {
				logf("incomplete CNAME chain from %q: %s", target, e1)
				continue
			}
			// Len(m1.Answer) > 0 here is well?
			records = append(records, newRecord)
			records = append(records, m1.Answer...)
			continue
		case ip.To4() != nil && (q.Qtype == dns.TypeA || both):
			records = append(records, serv.NewA(q.Name, ip.To4()))
		case ip.To4() == nil && (q.Qtype == dns.TypeAAAA || both):
			records = append(records, serv.NewAAAA(q.Name, ip.To16()))
		}
	}
	s.RoundRobin(records)
	return records, nil
}

// NSRecords returns NS records from etcd.
func (s *server) NSRecords(q dns.Question, name string) (records []dns.RR, extra []dns.RR, err error) {
	services, err := s.backend.Records(name, false)
	if err != nil {
		return nil, nil, err
	}

	services = msg.Group(services)

	for _, serv := range services {
		ip := net.ParseIP(serv.Host)
		switch {
		case ip == nil:
			return nil, nil, fmt.Errorf("NS record must be an IP address")
		case ip.To4() != nil:
			serv.Host = msg.Domain(serv.Key)
			records = append(records, serv.NewNS(q.Name, serv.Host))
			extra = append(extra, serv.NewA(serv.Host, ip.To4()))
		case ip.To4() == nil:
			serv.Host = msg.Domain(serv.Key)
			records = append(records, serv.NewNS(q.Name, serv.Host))
			extra = append(extra, serv.NewAAAA(serv.Host, ip.To16()))
		}
	}
	return records, extra, nil
}

// SRVRecords returns SRV records from etcd.
// If the Target is not a name but an IP address, a name is created.
func (s *server) SRVRecords(q dns.Question, name string, bufsize uint16, dnssec bool) (records []dns.RR, extra []dns.RR, err error) {
	services, err := s.backend.Records(name, false)
	if err != nil {
		return nil, nil, err
	}

	services = msg.Group(services)

	// Looping twice to get the right weight vs priority
	w := make(map[int]int)
	for _, serv := range services {
		weight := 100
		if serv.Weight != 0 {
			weight = serv.Weight
		}
		if _, ok := w[serv.Priority]; !ok {
			w[serv.Priority] = weight
			continue
		}
		w[serv.Priority] += weight
	}
	lookup := make(map[string]bool)
	for _, serv := range services {
		w1 := 100.0 / float64(w[serv.Priority])
		if serv.Weight == 0 {
			w1 *= 100
		} else {
			w1 *= float64(serv.Weight)
		}
		weight := uint16(math.Floor(w1))
		ip := net.ParseIP(serv.Host)
		switch {
		case ip == nil:
			srv := serv.NewSRV(q.Name, weight)
			records = append(records, srv)

			if _, ok := lookup[srv.Target]; ok {
				break
			}

			lookup[srv.Target] = true

			if !dns.IsSubDomain(s.config.Domain, srv.Target) {
				m1, e1 := s.Lookup(srv.Target, dns.TypeA, bufsize, dnssec)
				if e1 == nil {
					extra = append(extra, m1.Answer...)
				}
				m1, e1 = s.Lookup(srv.Target, dns.TypeAAAA, bufsize, dnssec)
				if e1 == nil {
					// If we have seen CNAME's we *assume* that they are already added.
					for _, a := range m1.Answer {
						if _, ok := a.(*dns.CNAME); !ok {
							extra = append(extra, a)
						}
					}
				}
				break
			}
			// Internal name, we should have some info on them, either v4 or v6
			// Clients expect a complete answer, because we are a recursor in their
			// view.
			addr, e1 := s.AddressRecords(dns.Question{srv.Target, dns.ClassINET, dns.TypeA},
				srv.Target, nil, bufsize, dnssec, true)
			if e1 == nil {
				extra = append(extra, addr...)
			}
		case ip.To4() != nil:
			serv.Host = msg.Domain(serv.Key)
			srv := serv.NewSRV(q.Name, weight)

			records = append(records, srv)
			extra = append(extra, serv.NewA(srv.Target, ip.To4()))
		case ip.To4() == nil:
			serv.Host = msg.Domain(serv.Key)
			srv := serv.NewSRV(q.Name, weight)

			records = append(records, srv)
			extra = append(extra, serv.NewAAAA(srv.Target, ip.To16()))
		}
	}
	return records, extra, nil
}

// MXRecords returns MX records from etcd.
// If the Target is not a name but an IP address, a name is created.
func (s *server) MXRecords(q dns.Question, name string, bufsize uint16, dnssec bool) (records []dns.RR, extra []dns.RR, err error) {
	services, err := s.backend.Records(name, false)
	if err != nil {
		return nil, nil, err
	}

	lookup := make(map[string]bool)
	for _, serv := range services {
		if !serv.Mail {
			continue
		}
		ip := net.ParseIP(serv.Host)
		switch {
		case ip == nil:
			mx := serv.NewMX(q.Name)
			records = append(records, mx)
			if _, ok := lookup[mx.Mx]; ok {
				break
			}

			lookup[mx.Mx] = true

			if !dns.IsSubDomain(s.config.Domain, mx.Mx) {
				m1, e1 := s.Lookup(mx.Mx, dns.TypeA, bufsize, dnssec)
				if e1 == nil {
					extra = append(extra, m1.Answer...)
				}
				m1, e1 = s.Lookup(mx.Mx, dns.TypeAAAA, bufsize, dnssec)
				if e1 == nil {
					// If we have seen CNAME's we *assume* that they are already added.
					for _, a := range m1.Answer {
						if _, ok := a.(*dns.CNAME); !ok {
							extra = append(extra, a)
						}
					}
				}
				break
			}
			// Internal name
			addr, e1 := s.AddressRecords(dns.Question{mx.Mx, dns.ClassINET, dns.TypeA},
				mx.Mx, nil, bufsize, dnssec, true)
			if e1 == nil {
				extra = append(extra, addr...)
			}
		case ip.To4() != nil:
			serv.Host = msg.Domain(serv.Key)
			records = append(records, serv.NewMX(q.Name))
			extra = append(extra, serv.NewA(serv.Host, ip.To4()))
		case ip.To4() == nil:
			serv.Host = msg.Domain(serv.Key)
			records = append(records, serv.NewMX(q.Name))
			extra = append(extra, serv.NewAAAA(serv.Host, ip.To16()))
		}
	}
	return records, extra, nil
}

func (s *server) CNAMERecords(q dns.Question, name string) (records []dns.RR, err error) {
	services, err := s.backend.Records(name, true)
	if err != nil {
		return nil, err
	}

	services = msg.Group(services)

	if len(services) > 0 {
		serv := services[0]
		if ip := net.ParseIP(serv.Host); ip == nil {
			records = append(records, serv.NewCNAME(q.Name, dns.Fqdn(serv.Host)))
		}
	}
	return records, nil
}

func (s *server) TXTRecords(q dns.Question, name string) (records []dns.RR, err error) {
	services, err := s.backend.Records(name, false)
	if err != nil {
		return nil, err
	}

	services = msg.Group(services)

	for _, serv := range services {
		if serv.Text == "" {
			continue
		}
		records = append(records, serv.NewTXT(q.Name))
	}
	return records, nil
}

func (s *server) PTRRecords(q dns.Question) (records []dns.RR, err error) {
	name := strings.ToLower(q.Name)
	serv, err := s.backend.ReverseRecord(name)
	if err != nil {
		return nil, err
	}

	records = append(records, serv.NewPTR(q.Name, serv.Ttl))
	return records, nil
}

// SOA returns a SOA record for this SkyDNS instance.
func (s *server) NewSOA() dns.RR {
	return &dns.SOA{Hdr: dns.RR_Header{Name: s.config.Domain, Rrtype: dns.TypeSOA, Class: dns.ClassINET, Ttl: s.config.Ttl},
		Ns:      appendDomain("ns.dns", s.config.Domain),
		Mbox:    s.config.Hostmaster,
		Serial:  uint32(time.Now().Truncate(time.Hour).Unix()),
		Refresh: 28800,
		Retry:   7200,
		Expire:  604800,
		Minttl:  s.config.MinTtl,
	}
}

func (s *server) isDuplicateCNAME(r *dns.CNAME, records []dns.RR) bool {
	for _, rec := range records {
		if v, ok := rec.(*dns.CNAME); ok {
			if v.Target == r.Target {
				return true
			}
		}
	}
	return false
}

func (s *server) NameError(req *dns.Msg) *dns.Msg {
	m := new(dns.Msg)
	m.SetRcode(req, dns.RcodeNameError)
	m.Ns = []dns.RR{s.NewSOA()}
	m.Ns[0].Header().Ttl = s.config.MinTtl
	return m
}

func (s *server) ServerFailure(req *dns.Msg) *dns.Msg {
	m := new(dns.Msg)
	m.SetRcode(req, dns.RcodeServerFailure)
	return m
}

func (s *server) RoundRobin(rrs []dns.RR) {
	if !s.config.RoundRobin {
		return
	}
	// If we have more than 1 CNAME don't touch the packet, because some stub resolver (=glibc)
	// can't deal with the returned packet if the CNAMEs need to be accesses in the reverse order.
	cname := 0
	for _, r := range rrs {
		if r.Header().Rrtype == dns.TypeCNAME {
			cname++
			if cname > 1 {
				return
			}
		}
	}

	switch l := len(rrs); l {
	case 2:
		if dns.Id()%2 == 0 {
			rrs[0], rrs[1] = rrs[1], rrs[0]
		}
	default:
		for j := 0; j < l*(int(dns.Id())%4+1); j++ {
			q := int(dns.Id()) % l
			p := int(dns.Id()) % l
			if q == p {
				p = (p + 1) % l
			}
			rrs[q], rrs[p] = rrs[p], rrs[q]
		}
	}

}

// dedup will de-duplicate a message on a per section basis.
// Multiple identical (same name, class, type and rdata) RRs will be coalesced into one.
func (s *server) dedup(m *dns.Msg) *dns.Msg {
	// Answer section
	ma := make(map[string]dns.RR)
	for _, a := range m.Answer {
		// Or use Pack()... Think this function also could be placed in go dns.
		s1 := a.Header().Name
		s1 += strconv.Itoa(int(a.Header().Class))
		s1 += strconv.Itoa(int(a.Header().Rrtype))
		// there can only be one CNAME for an ownername
		if a.Header().Rrtype == dns.TypeCNAME {
			if _, ok := ma[s1]; ok {
				// already exist, randomly overwrite if roundrobin is true
				// Note: even with roundrobin *off* this depends on the
				// order we get the names.
				if s.config.RoundRobin && dns.Id()%2 == 0 {
					ma[s1] = a
					continue
				}
			}
			ma[s1] = a
			continue
		}
		for i := 1; i <= dns.NumField(a); i++ {
			s1 += dns.Field(a, i)
		}
		ma[s1] = a
	}
	// Only is our map is smaller than the #RR in the answer section we should reset the RRs
	// in the section it self
	if len(ma) < len(m.Answer) {
		i := 0
		for _, v := range ma {
			m.Answer[i] = v
			i++
		}
		m.Answer = m.Answer[:len(ma)]
	}

	// Additional section
	me := make(map[string]dns.RR)
	for _, e := range m.Extra {
		s1 := e.Header().Name
		s1 += strconv.Itoa(int(e.Header().Class))
		s1 += strconv.Itoa(int(e.Header().Rrtype))
		// there can only be one CNAME for an ownername
		if e.Header().Rrtype == dns.TypeCNAME {
			if _, ok := me[s1]; ok {
				// already exist, randomly overwrite if roundrobin is true
				if s.config.RoundRobin && dns.Id()%2 == 0 {
					me[s1] = e
					continue
				}
			}
			me[s1] = e
			continue
		}
		for i := 1; i <= dns.NumField(e); i++ {
			s1 += dns.Field(e, i)
		}
		me[s1] = e
	}

	if len(me) < len(m.Extra) {
		i := 0
		for _, v := range me {
			m.Extra[i] = v
			i++
		}
		m.Extra = m.Extra[:len(me)]
	}

	return m
}

// overflowOrTruncated writes back an error to the client if the message does not fit.
// It updates prometheus metrics. If something has been written to the client, true
// will be returned.
func (s *server) overflowOrTruncated(w dns.ResponseWriter, m *dns.Msg, bufsize int, sy metrics.System) bool {
	switch isTCP(w) {
	case true:
		if _, overflow := Fit(m, dns.MaxMsgSize, true); overflow {
			metrics.ReportErrorCount(m, sy)
			msgFail := s.ServerFailure(m)
			w.WriteMsg(msgFail)
			return true
		}
	case false:
		// Overflow with udp always results in TC.
		Fit(m, bufsize, false)
		metrics.ReportErrorCount(m, sy)
		if m.Truncated {
			w.WriteMsg(m)
			return true
		}
	}
	return false
}

// isTCP returns true if the client is connecting over TCP.
func isTCP(w dns.ResponseWriter) bool {
	_, ok := w.RemoteAddr().(*net.TCPAddr)
	return ok
}

// etcNameError return a NameError to the client if the error
// returned from etcd has ErrorCode == 100.
func isEtcdNameError(err error, s *server) bool {
	if e, ok := err.(etcd.Error); ok && e.Code == etcd.ErrorCodeKeyNotFound {
		return true
	}
	if err != nil {
		logf("error from backend: %s", err)
	}
	return false
}
