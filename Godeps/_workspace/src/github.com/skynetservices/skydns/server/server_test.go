// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

// etcd needs to be running on http://127.0.0.1:4001
// running standalone tests fails, because metrics need to be enabled. TODO(miek)
// See `if !metricsDone {` in TestMsgOverflow, should be added to more? TODO(miek)

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/miekg/dns"
	backendetcd "github.com/skynetservices/skydns/backends/etcd"
	"github.com/skynetservices/skydns/cache"
	"github.com/skynetservices/skydns/msg"
)

// Keep global port counter that increments with 10 for each
// new call to newTestServer. The dns server is started on port 'Port'.
var Port = 9400
var StrPort = "9400" // string equivalent of Port

func addService(t *testing.T, s *server, k string, ttl uint64, m *msg.Service) {
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	path, _ := msg.PathWithWildcard(k)

	_, err = s.backend.(*backendetcd.Backend).Client().Create(path, string(b), ttl)
	if err != nil {
		// TODO(miek): allow for existing keys...
		t.Fatal(err)
	}
}

func delService(t *testing.T, s *server, k string) {
	path, _ := msg.PathWithWildcard(k)
	_, err := s.backend.(*backendetcd.Backend).Client().Delete(path, false)
	if err != nil {
		t.Fatal(err)
	}
}

func newTestServer(t *testing.T, c bool) *server {
	Port += 10
	StrPort = strconv.Itoa(Port)
	s := new(server)
	client := etcd.NewClient([]string{"http://127.0.0.1:4001"})

	// TODO(miek): why don't I use NewServer??
	s.group = new(sync.WaitGroup)
	s.scache = cache.New(100, 0)
	s.rcache = cache.New(100, 0)
	if c {
		s.rcache = cache.New(100, 60) // 100 items, 60s ttl
	}
	s.config = new(Config)
	s.config.Domain = "skydns.test."
	s.config.DnsAddr = "127.0.0.1:" + StrPort
	s.config.Nameservers = []string{"8.8.4.4:53"}
	SetDefaults(s.config)
	s.config.Local = "104.server1.development.region1.skydns.test."
	s.config.Priority = 10
	s.config.RCacheTtl = RCacheTtl
	s.config.Ttl = 3600
	s.config.Ndots = 2

	s.dnsUDPclient = &dns.Client{Net: "udp", ReadTimeout: 2 * s.config.ReadTimeout, WriteTimeout: 2 * s.config.ReadTimeout, SingleInflight: true}
	s.dnsTCPclient = &dns.Client{Net: "tcp", ReadTimeout: 2 * s.config.ReadTimeout, WriteTimeout: 2 * s.config.ReadTimeout, SingleInflight: true}

	s.backend = backendetcd.NewBackend(client, &backendetcd.Config{
		Ttl:      s.config.Ttl,
		Priority: s.config.Priority,
	})

	go s.Run()
	// Yeah, yeah, should do a proper fix.
	time.Sleep(500 * time.Millisecond)
	return s
}

func newTestServerDNSSEC(t *testing.T, cache bool) *server {
	var err error
	s := newTestServer(t, cache)
	s.config.PubKey = newDNSKEY("skydns.test. IN DNSKEY 256 3 5 AwEAAaXfO+DOBMJsQ5H4TfiabwSpqE4cGL0Qlvh5hrQumrjr9eNSdIOjIHJJKCe56qBU5mH+iBlXP29SVf6UiiMjIrAPDVhClLeWFe0PC+XlWseAyRgiLHdQ8r95+AfkhO5aZgnCwYf9FGGSaT0+CRYN+PyDbXBTLK5FN+j5b6bb7z+d")
	s.config.KeyTag = s.config.PubKey.KeyTag()
	s.config.PrivKey, err = s.config.PubKey.ReadPrivateKey(strings.NewReader(`Private-key-format: v1.3
Algorithm: 5 (RSASHA1)
Modulus: pd874M4EwmxDkfhN+JpvBKmoThwYvRCW+HmGtC6auOv141J0g6MgckkoJ7nqoFTmYf6IGVc/b1JV/pSKIyMisA8NWEKUt5YV7Q8L5eVax4DJGCIsd1Dyv3n4B+SE7lpmCcLBh/0UYZJpPT4JFg34/INtcFMsrkU36PlvptvvP50=
PublicExponent: AQAB
PrivateExponent: C6e08GXphbPPx6j36ZkIZf552gs1XcuVoB4B7hU8P/Qske2QTFOhCwbC8I+qwdtVWNtmuskbpvnVGw9a6X8lh7Z09RIgzO/pI1qau7kyZcuObDOjPw42exmjqISFPIlS1wKA8tw+yVzvZ19vwRk1q6Rne+C1romaUOTkpA6UXsE=
Prime1: 2mgJ0yr+9vz85abrWBWnB8Gfa1jOw/ccEg8ZToM9GLWI34Qoa0D8Dxm8VJjr1tixXY5zHoWEqRXciTtY3omQDQ==
Prime2: wmxLpp9rTzU4OREEVwF43b/TxSUBlUq6W83n2XP8YrCm1nS480w4HCUuXfON1ncGYHUuq+v4rF+6UVI3PZT50Q==
Exponent1: wkdTngUcIiau67YMmSFBoFOq9Lldy9HvpVzK/R0e5vDsnS8ZKTb4QJJ7BaG2ADpno7pISvkoJaRttaEWD3a8rQ==
Exponent2: YrC8OglEXIGkV3tm2494vf9ozPL6+cBkFsPPg9dXbvVCyyuW0pGHDeplvfUqs4nZp87z8PsoUL+LAUqdldnwcQ==
Coefficient: mMFr4+rDY5V24HZU3Oa5NEb55iQ56ZNa182GnNhWqX7UqWjcUUGjnkCy40BqeFAQ7lp52xKHvP5Zon56mwuQRw==
`), "stdin")
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestDNSForward(t *testing.T) {
	s := newTestServer(t, false)
	defer s.Stop()

	c := new(dns.Client)
	m := new(dns.Msg)
	m.SetQuestion("www.example.com.", dns.TypeA)
	resp, _, err := c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		// try twice
		resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(resp.Answer) == 0 || resp.Rcode != dns.RcodeSuccess {
		t.Fatal("answer expected to have A records or rcode not equal to RcodeSuccess")
	}
	// TCP
	c.Net = "tcp"
	resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Answer) == 0 || resp.Rcode != dns.RcodeSuccess {
		t.Fatal("answer expected to have A records or rcode not equal to RcodeSuccess")
	}
	// disable recursion and check
	s.config.NoRec = true

	m.SetQuestion("www.example.com.", dns.TypeA)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Rcode != dns.RcodeServerFailure {
		t.Fatal("answer expected to have rcode equal to RcodeFailure")
	}
}

func TestDNSStubForward(t *testing.T) {
	s := newTestServer(t, false)
	defer s.Stop()

	c := new(dns.Client)
	m := new(dns.Msg)

	stubEx := &msg.Service{
		// IP address of a.iana-servers.net.
		Host: "199.43.132.53", Key: "a.example.com.stub.dns.skydns.test.",
	}
	stubBroken := &msg.Service{
		Host: "127.0.0.1", Port: 5454, Key: "b.example.org.stub.dns.skydns.test.",
	}
	stubLoop := &msg.Service{
		Host: "127.0.0.1", Port: Port, Key: "b.example.net.stub.dns.skydns.test.",
	}
	addService(t, s, stubEx.Key, 0, stubEx)
	defer delService(t, s, stubEx.Key)
	addService(t, s, stubBroken.Key, 0, stubBroken)
	defer delService(t, s, stubBroken.Key)
	addService(t, s, stubLoop.Key, 0, stubLoop)
	defer delService(t, s, stubLoop.Key)

	s.UpdateStubZones()

	m.SetQuestion("www.example.com.", dns.TypeA)
	resp, _, err := c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		// try twice
		resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(resp.Answer) == 0 || resp.Rcode != dns.RcodeSuccess {
		t.Fatal("answer expected to have A records or rcode not equal to RcodeSuccess")
	}
	// The main diff. here is that we expect the AA bit to be set, because we directly
	// queried the authoritative servers.
	if resp.Authoritative != true {
		t.Fatal("answer expected to have AA bit set")
	}

	// This should fail.
	m.SetQuestion("www.example.org.", dns.TypeA)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
	if len(resp.Answer) != 0 || resp.Rcode != dns.RcodeServerFailure {
		t.Fatal("answer expected to fail for example.org")
	}

	// This should really fail with a timeout.
	m.SetQuestion("www.example.net.", dns.TypeA)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
	if err == nil {
		t.Fatal("answer expected to fail for example.net")
	} else {
		t.Logf("succesfully failing %s", err)
	}

	// Packet with EDNS0
	m.SetEdns0(4096, true)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
	if err == nil {
		t.Fatal("answer expected to fail for example.net")
	} else {
		t.Logf("succesfully failing %s", err)
	}

	// Now start another SkyDNS instance on a different port,
	// add a stubservice for it and check if the forwarding is
	// actually working.
	oldStrPort := StrPort

	s1 := newTestServer(t, false)
	defer s1.Stop()
	s1.config.Domain = "skydns.com."

	// Add forwarding IP for internal.skydns.com. Use Port to point to server s.
	stubForward := &msg.Service{
		Host: "127.0.0.1", Port: Port, Key: "b.internal.skydns.com.stub.dns.skydns.test.",
	}
	addService(t, s, stubForward.Key, 0, stubForward)
	defer delService(t, s, stubForward.Key)
	s.UpdateStubZones()

	// Add an answer for this in our "new" server.
	stubReply := &msg.Service{
		Host: "127.1.1.1", Key: "www.internal.skydns.com.",
	}
	addService(t, s1, stubReply.Key, 0, stubReply)
	defer delService(t, s1, stubReply.Key)

	m = new(dns.Msg)
	m.SetQuestion("www.internal.skydns.com.", dns.TypeA)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+oldStrPort)
	if err != nil {
		t.Fatalf("failed to forward %s", err)
	}
	if resp.Answer[0].(*dns.A).A.String() != "127.1.1.1" {
		t.Fatalf("failed to get correct reply")
	}

	// Adding an in baliwick internal domain forward.
	s2 := newTestServer(t, false)
	defer s2.Stop()
	s2.config.Domain = "internal.skydns.net."

	// Add forwarding IP for internal.skydns.net. Use Port to point to server s.
	stubForward1 := &msg.Service{
		Host: "127.0.0.1", Port: Port, Key: "b.internal.skydns.net.stub.dns.skydns.test.",
	}
	addService(t, s, stubForward1.Key, 0, stubForward1)
	defer delService(t, s, stubForward1.Key)
	s.UpdateStubZones()

	// Add an answer for this in our "new" server.
	stubReply1 := &msg.Service{
		Host: "127.10.10.10", Key: "www.internal.skydns.net.",
	}
	addService(t, s2, stubReply1.Key, 0, stubReply1)
	defer delService(t, s2, stubReply1.Key)

	m = new(dns.Msg)
	m.SetQuestion("www.internal.skydns.net.", dns.TypeA)
	resp, _, err = c.Exchange(m, "127.0.0.1:"+oldStrPort)
	if err != nil {
		t.Fatalf("failed to forward %s", err)
	}
	if resp.Answer[0].(*dns.A).A.String() != "127.10.10.10" {
		t.Fatalf("failed to get correct reply")
	}
}

func TestDNSTtlRRset(t *testing.T) {
	s := newTestServerDNSSEC(t, false)
	defer s.Stop()

	ttl := uint32(60)
	for _, serv := range services {
		addService(t, s, serv.Key, uint64(ttl), serv)
		defer delService(t, s, serv.Key)
		ttl += 60
	}
	c := new(dns.Client)
	tc := dnsTestCases[9]
	t.Logf("%v\n", tc)
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)
	if tc.dnssec == true {
		m.SetEdns0(4096, true)
	}
	resp, _, err := c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		t.Fatalf("failing: %s: %s\n", m.String(), err.Error())
	}
	t.Logf("%s\n", resp)
	ttl = 360
	for i, a := range resp.Answer {
		if a.Header().Ttl != ttl {
			t.Errorf("Answer %d should have a Header TTL of %d, but has %d", i, ttl, a.Header().Ttl)
		}
	}
}

type rrSet []dns.RR

func (p rrSet) Len() int           { return len(p) }
func (p rrSet) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p rrSet) Less(i, j int) bool { return p[i].String() < p[j].String() }

func TestDNS(t *testing.T) {
	s := newTestServerDNSSEC(t, false)
	defer s.Stop()

	for _, serv := range services {
		addService(t, s, serv.Key, 0, serv)
		defer delService(t, s, serv.Key)
	}
	c := new(dns.Client)
	for _, tc := range dnsTestCases {
		m := new(dns.Msg)
		m.SetQuestion(tc.Qname, tc.Qtype)
		if tc.dnssec {
			m.SetEdns0(4096, true)
		}
		if tc.chaos {
			m.Question[0].Qclass = dns.ClassCHAOS
		}
		resp, _, err := c.Exchange(m, "127.0.0.1:"+StrPort)
		if err != nil {
			// try twice, be more resilent against remote lookups
			// timing out.
			resp, _, err = c.Exchange(m, "127.0.0.1:"+StrPort)
			if err != nil {
				t.Fatalf("failing: %s: %s\n", m.String(), err.Error())
			}
		}
		sort.Sort(rrSet(resp.Answer))
		sort.Sort(rrSet(resp.Ns))
		sort.Sort(rrSet(resp.Extra))
		fatal := false
		defer func() {
			if fatal {
				t.Logf("question: %s\n", m.Question[0].String())
				t.Logf("%s\n", resp)
			}
		}()
		if resp.Rcode != tc.Rcode {
			fatal = true
			t.Fatalf("rcode is %q, expected %q", dns.RcodeToString[resp.Rcode], dns.RcodeToString[tc.Rcode])
		}
		if len(resp.Answer) != len(tc.Answer) {
			fatal = true
			t.Fatalf("answer for %q contained %d results, %d expected", tc.Qname, len(resp.Answer), len(tc.Answer))
		}
		for i, a := range resp.Answer {
			if a.Header().Name != tc.Answer[i].Header().Name {
				fatal = true
				t.Fatalf("answer %d should have a Header Name of %q, but has %q", i, tc.Answer[i].Header().Name, a.Header().Name)
			}
			if a.Header().Ttl != tc.Answer[i].Header().Ttl {
				fatal = true
				t.Fatalf("Answer %d should have a Header TTL of %d, but has %d", i, tc.Answer[i].Header().Ttl, a.Header().Ttl)
			}
			if a.Header().Rrtype != tc.Answer[i].Header().Rrtype {
				fatal = true
				t.Fatalf("answer %d should have a header response type of %d, but has %d", i, tc.Answer[i].Header().Rrtype, a.Header().Rrtype)
			}
			switch x := a.(type) {
			case *dns.SRV:
				if x.Priority != tc.Answer[i].(*dns.SRV).Priority {
					fatal = true
					t.Fatalf("answer %d should have a Priority of %d, but has %d", i, tc.Answer[i].(*dns.SRV).Priority, x.Priority)
				}
				if x.Weight != tc.Answer[i].(*dns.SRV).Weight {
					fatal = true
					t.Fatalf("answer %d should have a Weight of %d, but has %d", i, tc.Answer[i].(*dns.SRV).Weight, x.Weight)
				}
				if x.Port != tc.Answer[i].(*dns.SRV).Port {
					fatal = true
					t.Fatalf("answer %d should have a Port of %d, but has %d", i, tc.Answer[i].(*dns.SRV).Port, x.Port)
				}
				if x.Target != tc.Answer[i].(*dns.SRV).Target {
					fatal = true
					t.Fatalf("answer %d should have a Target of %q, but has %q", i, tc.Answer[i].(*dns.SRV).Target, x.Target)
				}
			case *dns.A:
				if x.A.String() != tc.Answer[i].(*dns.A).A.String() {
					fatal = true
					t.Fatalf("answer %d should have a Address of %q, but has %q", i, tc.Answer[i].(*dns.A).A.String(), x.A.String())
				}
			case *dns.AAAA:
				if x.AAAA.String() != tc.Answer[i].(*dns.AAAA).AAAA.String() {
					fatal = true
					t.Fatalf("answer %d should have a Address of %q, but has %q", i, tc.Answer[i].(*dns.AAAA).AAAA.String(), x.AAAA.String())
				}
			case *dns.TXT:
				for j, txt := range x.Txt {
					if txt != tc.Answer[i].(*dns.TXT).Txt[j] {
						fatal = true
						t.Fatalf("answer %d should have a Txt of %q, but has %q", i, tc.Answer[i].(*dns.TXT).Txt[j], txt)
					}
				}
			case *dns.DNSKEY:
				tt := tc.Answer[i].(*dns.DNSKEY)
				if x.Flags != tt.Flags {
					fatal = true
					t.Fatalf("DNSKEY flags should be %q, but is %q", x.Flags, tt.Flags)
				}
				if x.Protocol != tt.Protocol {
					fatal = true
					t.Fatalf("DNSKEY protocol should be %q, but is %q", x.Protocol, tt.Protocol)
				}
				if x.Algorithm != tt.Algorithm {
					fatal = true
					t.Fatalf("DNSKEY algorithm should be %q, but is %q", x.Algorithm, tt.Algorithm)
				}
			case *dns.RRSIG:
				tt := tc.Answer[i].(*dns.RRSIG)
				if x.TypeCovered != tt.TypeCovered {
					fatal = true
					t.Fatalf("RRSIG type-covered should be %d, but is %d", x.TypeCovered, tt.TypeCovered)
				}
				if x.Algorithm != tt.Algorithm {
					fatal = true
					t.Fatalf("RRSIG algorithm should be %d, but is %d", x.Algorithm, tt.Algorithm)
				}
				if x.Labels != tt.Labels {
					fatal = true
					t.Fatalf("RRSIG label should be %d, but is %d", x.Labels, tt.Labels)
				}
				if x.OrigTtl != tt.OrigTtl {
					fatal = true
					t.Fatalf("RRSIG orig-ttl should be %d, but is %d", x.OrigTtl, tt.OrigTtl)
				}
				if x.KeyTag != tt.KeyTag {
					fatal = true
					t.Fatalf("RRSIG key-tag should be %d, but is %d", x.KeyTag, tt.KeyTag)
				}
				if x.SignerName != tt.SignerName {
					fatal = true
					t.Fatalf("RRSIG signer-name should be %q, but is %q", x.SignerName, tt.SignerName)
				}
			case *dns.SOA:
				tt := tc.Answer[i].(*dns.SOA)
				if x.Ns != tt.Ns {
					fatal = true
					t.Fatalf("SOA nameserver should be %q, but is %q", x.Ns, tt.Ns)
				}
			case *dns.PTR:
				tt := tc.Answer[i].(*dns.PTR)
				if x.Ptr != tt.Ptr {
					fatal = true
					t.Fatalf("PTR ptr should be %q, but is %q", x.Ptr, tt.Ptr)
				}
			case *dns.CNAME:
				tt := tc.Answer[i].(*dns.CNAME)
				if x.Target != tt.Target {
					fatal = true
					t.Fatalf("CNAME target should be %q, but is %q", x.Target, tt.Target)
				}
			case *dns.MX:
				tt := tc.Answer[i].(*dns.MX)
				if x.Mx != tt.Mx {
					t.Fatalf("MX Mx should be %q, but is %q", x.Mx, tt.Mx)
				}
				if x.Preference != tt.Preference {
					t.Fatalf("MX Preference should be %q, but is %q", x.Preference, tt.Preference)
				}
			}
		}
		if len(resp.Ns) != len(tc.Ns) {
			fatal = true
			t.Fatalf("authority for %q contained %d results, %d expected", tc.Qname, len(resp.Ns), len(tc.Ns))
		}
		for i, n := range resp.Ns {
			switch x := n.(type) {
			case *dns.SOA:
				tt := tc.Ns[i].(*dns.SOA)
				if x.Ns != tt.Ns {
					fatal = true
					t.Fatalf("SOA nameserver should be %q, but is %q", x.Ns, tt.Ns)
				}
			case *dns.NS:
				tt := tc.Ns[i].(*dns.NS)
				if x.Ns != tt.Ns {
					fatal = true
					t.Fatalf("NS nameserver should be %q, but is %q", x.Ns, tt.Ns)
				}
			case *dns.NSEC3:
				tt := tc.Ns[i].(*dns.NSEC3)
				if x.NextDomain != tt.NextDomain {
					fatal = true
					t.Fatalf("NSEC3 nextdomain should be %q, but is %q", x.NextDomain, tt.NextDomain)
				}
				if x.Hdr.Name != tt.Hdr.Name {
					fatal = true
					t.Fatalf("NSEC3 ownername should be %q, but is %q", x.Hdr.Name, tt.Hdr.Name)
				}
				for j, y := range x.TypeBitMap {
					if y != tt.TypeBitMap[j] {
						fatal = true
						t.Fatalf("NSEC3 bitmap should have %q, but is %q", dns.TypeToString[y], dns.TypeToString[tt.TypeBitMap[j]])
					}
				}
			}
		}
		if len(resp.Extra) != len(tc.Extra) {
			fatal = true
			t.Fatalf("additional for %q contained %d results, %d expected", tc.Qname, len(resp.Extra), len(tc.Extra))
		}
		for i, e := range resp.Extra {
			switch x := e.(type) {
			case *dns.A:
				if x.A.String() != tc.Extra[i].(*dns.A).A.String() {
					fatal = true
					t.Fatalf("extra %d should have a address of %q, but has %q", i, tc.Extra[i].(*dns.A).A.String(), x.A.String())
				}
			case *dns.AAAA:
				if x.AAAA.String() != tc.Extra[i].(*dns.AAAA).AAAA.String() {
					fatal = true
					t.Fatalf("extra %d should have a address of %q, but has %q", i, tc.Extra[i].(*dns.AAAA).AAAA.String(), x.AAAA.String())
				}
			case *dns.CNAME:
				tt := tc.Extra[i].(*dns.CNAME)
				if x.Target != tt.Target {
					// Super super gross hack.
					if x.Target == "a.ipaddr.skydns.test." && tt.Target == "b.ipaddr.skydns.test." {
						// These records are randomly choosen, either one is OK.
						continue
					}
					fatal = true
					t.Fatalf("CNAME target should be %q, but is %q", x.Target, tt.Target)
				}
			}
		}
	}
}

type dnsTestCase struct {
	Qname  string
	Qtype  uint16
	dnssec bool
	chaos  bool
	Rcode  int
	Answer []dns.RR
	Ns     []dns.RR
	Extra  []dns.RR
}

// Note the key is encoded as dns name, while in "reality" it is a Etcd path.
var services = []*msg.Service{
	{Host: "server1", Port: 8080, Key: "100.server1.development.region1.skydns.test."},
	{Host: "server2", Port: 80, Key: "101.server2.production.region1.skydns.test."},
	{Host: "server4", Port: 80, Priority: 333, Key: "102.server4.development.region6.skydns.test."},
	{Host: "server3", Key: "103.server4.development.region2.skydns.test."},
	{Host: "172.16.1.1", Key: "a.ipaddr.skydns.test."},
	{Host: "172.16.1.2", Key: "b.ipaddr.skydns.test."},
	{Host: "ipaddr.skydns.test", Key: "1.backend.in.skydns.test."},
	{Host: "10.0.0.1", Key: "104.server1.development.region1.skydns.test."},
	{Host: "2001::8:8:8:8", Key: "105.server3.production.region2.skydns.test."},
	{Host: "104.server1.development.region1.skydns.test", Key: "1.cname.skydns.test."},
	{Host: "100.server1.development.region1.skydns.test", Key: "2.cname.skydns.test."},
	{Host: "www.miek.nl", Key: "external1.cname.skydns.test."},
	{Host: "www.miek.nl", Key: "ext1.cname2.skydns.test."},
	{Host: "www.miek.nl", Key: "ext2.cname2.skydns.test."},
	{Host: "wwwwwww.miek.nl", Key: "external2.cname.skydns.test."},
	{Host: "4.cname.skydns.test", Key: "3.cname.skydns.test."},
	{Host: "3.cname.skydns.test", Key: "4.cname.skydns.test."},
	{Host: "10.0.0.2", Key: "ttl.skydns.test.", Ttl: 360},
	{Host: "reverse.example.com", Key: "1.0.0.10.in-addr.arpa."}, // 10.0.0.1
	{Host: "server1", Weight: 130, Key: "100.server1.region5.skydns.test."},
	{Host: "server2", Weight: 80, Key: "101.server2.region5.skydns.test."},
	{Host: "server3", Weight: 150, Key: "103.server3.region5.skydns.test."},
	{Host: "server4", Priority: 30, Key: "104.server4.region5.skydns.test."},
	{Host: "172.16.1.1", Key: "a.ipaddr2.skydns.test."},
	{Host: "2001::8:8:8:8", Key: "b.ipaddr2.skydns.test."},
	{Host: "ipaddr2.skydns.test", Key: "both.v4v6.test.skydns.test."},

	// A name: bar.skydns.test with 2 ports open and points to one ip: 192.168.0.1
	{Host: "192.168.0.1", Port: 80, Key: "x.bar.skydns.test.", TargetStrip: 1},
	{Host: "bar.skydns.local", Port: 443, Key: "y.bar.skydns.test.", TargetStrip: 0},

	// nameserver
	{Host: "10.0.0.2", Key: "a.ns.dns.skydns.test."},
	{Host: "10.0.0.3", Key: "b.ns.dns.skydns.test."},
	// txt
	{Text: "abc", Key: "a1.txt.skydns.test."},
	{Text: "abc abc", Key: "a2.txt.skydns.test."},
	// duplicate ip address
	{Host: "10.11.11.10", Key: "http.multiport.http.skydns.test.", Port: 80},
	{Host: "10.11.11.10", Key: "https.multiport.http.skydns.test.", Port: 443},

	// uppercase name
	{Host: "127.0.0.1", Key: "upper.skydns.test.", Port: 443},

	// mx
	{Host: "mx.skydns.test", Priority: 50, Mail: true, Key: "a.mail.skydns.test."},
	{Host: "mx.miek.nl", Priority: 50, Mail: true, Key: "b.mail.skydns.test."},
	{Host: "a.ipaddr.skydns.test", Priority: 30, Mail: true, Key: "a.mx.skydns.test."},

	// Double CNAME, see issue #168
	{Host: "mx2.skydns.test", Priority: 50, Mail: true, Key: "a.mail2.skydns.test."},
	{Host: "a.ipaddr.skydns.test", Mail: true, Key: "a.mx2.skydns.test."},
	// Sometimes we *do* get back a.ipaddr.skydns.test, making this test flaky.
	{Host: "b.ipaddr.skydns.test", Mail: true, Key: "b.mx2.skydns.test."},

	// groups
	{Host: "127.0.0.1", Key: "a.dom.skydns.test.", Group: "g1"},
	{Host: "127.0.0.2", Key: "b.sub.dom.skydns.test.", Group: "g1"},

	{Host: "127.0.0.1", Key: "a.dom2.skydns.test.", Group: "g1"},
	{Host: "127.0.0.2", Key: "b.sub.dom2.skydns.test.", Group: ""},

	{Host: "127.0.0.1", Key: "a.dom1.skydns.test.", Group: "g1"},
	{Host: "127.0.0.2", Key: "b.sub.dom1.skydns.test.", Group: "g2"},
}

var dnsTestCases = []dnsTestCase{
	// Full Name Test
	{
		Qname: "100.server1.development.region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{newSRV("100.server1.development.region1.skydns.test. 3600 SRV 10 100 8080 server1.")},
	},
	// SOA Record Test
	{
		Qname: "skydns.test.", Qtype: dns.TypeSOA,
		Answer: []dns.RR{newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0")},
	},
	// NS Record Test
	{
		Qname: "skydns.test.", Qtype: dns.TypeNS,
		Answer: []dns.RR{
			newNS("skydns.test. 3600 NS a.ns.dns.skydns.test."),
			newNS("skydns.test. 3600 NS b.ns.dns.skydns.test."),
		},
		Extra: []dns.RR{
			newA("a.ns.dns.skydns.test. 3600 A 10.0.0.2"),
			newA("b.ns.dns.skydns.test. 3600 A 10.0.0.3"),
		},
	},
	// A Record For NS Record Test
	{
		Qname: "ns.dns.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("ns.dns.skydns.test. 3600 A 10.0.0.2"),
			newA("ns.dns.skydns.test. 3600 A 10.0.0.3"),
		},
	},
	// A Record Test
	{
		Qname: "104.server1.development.region1.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1")},
	},
	// Multiple A Record Test
	{
		Qname: "ipaddr.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("ipaddr.skydns.test. 3600 A 172.16.1.1"),
			newA("ipaddr.skydns.test. 3600 A 172.16.1.2"),
		},
	},
	// A Record Test with SRV
	{
		Qname: "104.server1.development.region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{newSRV("104.server1.development.region1.skydns.test. 3600 SRV 10 100 0 104.server1.development.region1.skydns.test.")},
		Extra:  []dns.RR{newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1")},
	},
	// AAAAA Record Test
	{
		Qname: "105.server3.production.region2.skydns.test.", Qtype: dns.TypeAAAA,
		Answer: []dns.RR{newAAAA("105.server3.production.region2.skydns.test. 3600 AAAA 2001::8:8:8:8")},
	},
	// Multi SRV with the same target, should be dedupped.
	{
		Qname: "*.cname2.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("*.cname2.skydns.test. 3600 IN SRV 10 100 0 www.miek.nl."),
		},
		Extra: []dns.RR{
			newA("a.miek.nl. 3600 IN A 176.58.119.54"),
			newAAAA("a.miek.nl. 3600 IN AAAA 2a01:7e00::f03c:91ff:feae:e74c"),
			newCNAME("www.miek.nl. 3600 IN CNAME a.miek.nl."),
		},
	},
	// TTL Test
	{
		// This test is referenced by number from DNSTtlRRset
		Qname: "ttl.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("ttl.skydns.test. 360 A 10.0.0.2")},
	},
	// CNAME Test
	{
		Qname: "1.cname.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newCNAME("1.cname.skydns.test. 3600 CNAME 104.server1.development.region1.skydns.test."),
			newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1"),
		},
	},
	// Direct CNAME Test
	{
		Qname: "1.cname.skydns.test.", Qtype: dns.TypeCNAME,
		Answer: []dns.RR{
			newCNAME("1.cname.skydns.test. 3600 CNAME 104.server1.development.region1.skydns.test."),
		},
	},
	// CNAME (unresolvable internal name)
	{
		Qname: "2.cname.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{},
		Ns:     []dns.RR{newSOA("skydns.test. 60 SOA ns.dns.skydns.test. hostmaster.skydns.test. 1407441600 28800 7200 604800 60")},
	},
	// CNAME loop detection
	{
		Qname: "3.cname.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{},
		Ns:     []dns.RR{newSOA("skydns.test. 60 SOA ns.dns.skydns.test. hostmaster.skydns.test. 1407441600 28800 7200 604800 60")},
	},
	// CNAME (resolvable external name)
	{
		Qname: "external1.cname.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("a.miek.nl. 60 IN A 176.58.119.54"),
			newCNAME("external1.cname.skydns.test. 60 IN CNAME www.miek.nl."),
			newCNAME("www.miek.nl. 60 IN CNAME a.miek.nl."),
		},
	},
	// CNAME (unresolvable external name)
	{
		Qname: "external2.cname.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{},
		Ns:     []dns.RR{newSOA("skydns.test. 60 SOA ns.dns.skydns.test. hostmaster.skydns.test. 1407441600 28800 7200 604800 60")},
	},
	// Priority Test
	{
		Qname: "region6.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{newSRV("region6.skydns.test. 3600 SRV 333 100 80 server4.")},
	},
	// Subdomain Test
	{
		Qname: "region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("region1.skydns.test. 3600 SRV 10 33 0 104.server1.development.region1.skydns.test."),
			newSRV("region1.skydns.test. 3600 SRV 10 33 80 server2"),
			newSRV("region1.skydns.test. 3600 SRV 10 33 8080 server1.")},
		Extra: []dns.RR{newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1")},
	},
	// Subdomain Weight Test
	{
		Qname: "region5.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("region5.skydns.test. 3600 SRV 10 22 0 server2."),
			newSRV("region5.skydns.test. 3600 SRV 10 36 0 server1."),
			newSRV("region5.skydns.test. 3600 SRV 10 41 0 server3."),
			newSRV("region5.skydns.test. 3600 SRV 30 100 0 server4.")},
	},
	// Wildcard Test
	{
		Qname: "*.region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("*.region1.skydns.test. 3600 SRV 10 33 0 104.server1.development.region1.skydns.test."),
			newSRV("*.region1.skydns.test. 3600 SRV 10 33 80 server2"),
			newSRV("*.region1.skydns.test. 3600 SRV 10 33 8080 server1.")},
		Extra: []dns.RR{newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1")},
	},
	// Wildcard Test
	{
		Qname: "production.*.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("production.*.skydns.test. 3600 IN SRV 10 50 0 105.server3.production.region2.skydns.test."),
			newSRV("production.*.skydns.test. 3600 IN SRV 10 50 80 server2.")},
		Extra: []dns.RR{newAAAA("105.server3.production.region2.skydns.test. 3600 IN AAAA 2001::8:8:8:8")},
	},
	// Wildcard Test
	{
		Qname: "production.any.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("production.any.skydns.test. 3600 IN SRV 10 50 0 105.server3.production.region2.skydns.test."),
			newSRV("production.any.skydns.test. 3600 IN SRV 10 50 80 server2.")},
		Extra: []dns.RR{newAAAA("105.server3.production.region2.skydns.test. 3600 IN AAAA 2001::8:8:8:8")},
	},
	// NXDOMAIN Test
	{
		Qname: "doesnotexist.skydns.test.", Qtype: dns.TypeA,
		Rcode: dns.RcodeNameError,
		Ns: []dns.RR{
			newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0"),
		},
	},
	// NODATA Test
	{
		Qname: "104.server1.development.region1.skydns.test.", Qtype: dns.TypeTXT,
		Ns: []dns.RR{newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0")},
	},
	// NODATA Test 2
	{
		Qname: "100.server1.development.region1.skydns.test.", Qtype: dns.TypeA,
		Rcode: dns.RcodeSuccess,
		Ns:    []dns.RR{newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0")},
	},
	// CNAME Test that targets multiple A records (hits a directory in etcd)
	{
		Qname: "1.backend.in.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newCNAME("1.backend.in.skydns.test. IN CNAME ipaddr.skydns.test."),
			newA("ipaddr.skydns.test. IN A 172.16.1.1"),
			newA("ipaddr.skydns.test. IN A 172.16.1.2"),
		},
	},
	// Query a etcd directory key
	{
		Qname: "backend.in.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newCNAME("backend.in.skydns.test. IN CNAME ipaddr.skydns.test."),
			newA("ipaddr.skydns.test. IN A 172.16.1.1"),
			newA("ipaddr.skydns.test. IN A 172.16.1.2"),
		},
	},
	// Txt
	{
		Qname: "a1.txt.skydns.test.", Qtype: dns.TypeTXT,
		Answer: []dns.RR{
			newTXT("a1.txt.skydns.test. IN TXT \"abc\""),
		},
	},
	{
		Qname: "a2.txt.skydns.test.", Qtype: dns.TypeTXT,
		Answer: []dns.RR{
			newTXT("a2.txt.skydns.test. IN TXT \"abc abc\""),
		},
	},
	{
		Qname: "txt.skydns.test.", Qtype: dns.TypeTXT,
		Answer: []dns.RR{
			newTXT("txt.skydns.test. IN TXT \"abc abc\""),
			newTXT("txt.skydns.test. IN TXT \"abc\""),
		},
	},

	// DNSSEC

	// DNSKEY Test
	{
		dnssec: true,
		Qname:  "skydns.test.", Qtype: dns.TypeDNSKEY,
		Answer: []dns.RR{
			newDNSKEY("skydns.test. 3600 DNSKEY 256 3 5 deadbeaf"),
			newRRSIG("skydns.test. 3600 RRSIG DNSKEY 5 2 3600 0 0 51945 skydns.test. deadbeaf"),
		},
		Extra: []dns.RR{new(dns.OPT)},
	},
	// Signed Response Test
	{
		dnssec: true,
		Qname:  "104.server1.development.region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newRRSIG("104.server1.development.region1.skydns.test. 3600 RRSIG SRV 5 6 3600 0 0 51945 skydns.test. deadbeaf"),
			newSRV("104.server1.development.region1.skydns.test. 3600 SRV 10 100 0 104.server1.development.region1.skydns.test.")},
		Extra: []dns.RR{
			newRRSIG("104.server1.developmen.region1.skydns.test. 3600 RRSIG A 5 6 3600 0 0 51945 skydns.test. deadbeaf"),
			newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1"),
			new(dns.OPT),
		},
	},
	// Signed Response Test, ask twice to check cache
	{
		dnssec: true,
		Qname:  "104.server1.development.region1.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newRRSIG("104.server1.development.region1.skydns.test. 3600 RRSIG SRV 5 6 3600 0 0 51945 skydns.test. deadbeaf"),
			newSRV("104.server1.development.region1.skydns.test. 3600 SRV 10 100 0 104.server1.development.region1.skydns.test.")},
		Extra: []dns.RR{
			newRRSIG("104.server1.developmen.region1.skydns.test. 3600 RRSIG A 5 6 3600 0 0 51945 skydns.test. deadbeaf"),
			newA("104.server1.development.region1.skydns.test. 3600 A 10.0.0.1"),
			new(dns.OPT),
		},
	},
	// NXDOMAIN Test
	{
		dnssec: true,
		Qname:  "doesnotexist.skydns.test.", Qtype: dns.TypeA,
		Rcode: dns.RcodeNameError,
		Ns: []dns.RR{
			newNSEC3("44ohaq2njb0idnvolt9ggthvsk1e1uv8.skydns.test.	60 NSEC3 1 0 0 - 44OHAQ2NJB0IDNVOLT9GGTHVSK1E1UVA"),
			newRRSIG("44ohaq2njb0idnvolt9ggthvsk1e1uv8.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newNSEC3("ah4v7g5qoiri26armrb3bldqi1sng6a2.skydns.test.	60 NSEC3 1 0 0 - AH4V7G5QOIRI26ARMRB3BLDQI1SNG6A3 A AAAA SRV RRSIG"),
			newRRSIG("ah4v7g5qoiri26armrb3bldqi1sng6a2.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newNSEC3("lksd858f4cldl7emdord75k5jeks49p8.skydns.test.	60 NSEC3 1 0 0 - LKSD858F4CLDL7EMDORD75K5JEKS49PA"),
			newRRSIG("lksd858f4cldl7emdord75k5jeks49p8.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newRRSIG("skydns.test.	60 RRSIG SOA 5 2 3600 20140814205559 20140807175559 51945 skydns.test. deadbeaf"),
			newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0"),
		},
		Extra: []dns.RR{new(dns.OPT)},
	},
	// NXDOMAIN Test, cache test
	{
		dnssec: true,
		Qname:  "doesnotexist.skydns.test.", Qtype: dns.TypeA,
		Rcode: dns.RcodeNameError,
		Ns: []dns.RR{
			newNSEC3("44ohaq2njb0idnvolt9ggthvsk1e1uv8.skydns.test.	60 NSEC3 1 0 0 - 44OHAQ2NJB0IDNVOLT9GGTHVSK1E1UVA"),
			newRRSIG("44ohaq2njb0idnvolt9ggthvsk1e1uv8.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newNSEC3("ah4v7g5qoiri26armrb3bldqi1sng6a2.skydns.test.	60 NSEC3 1 0 0 - AH4V7G5QOIRI26ARMRB3BLDQI1SNG6A3 A AAAA SRV RRSIG"),
			newRRSIG("ah4v7g5qoiri26armrb3bldqi1sng6a2.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newNSEC3("lksd858f4cldl7emdord75k5jeks49p8.skydns.test.	60 NSEC3 1 0 0 - LKSD858F4CLDL7EMDORD75K5JEKS49PA"),
			newRRSIG("lksd858f4cldl7emdord75k5jeks49p8.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814205559 20140807175559 51945 skydns.test. deadbeef"),
			newRRSIG("skydns.test.	60 RRSIG SOA 5 2 3600 20140814205559 20140807175559 51945 skydns.test. deadbeaf"),
			newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0"),
		},
		Extra: []dns.RR{new(dns.OPT)},
	},
	// NODATA Test
	{
		dnssec: true,
		Qname:  "104.server1.development.region1.skydns.test.", Qtype: dns.TypeTXT,
		Rcode: dns.RcodeSuccess,
		Ns: []dns.RR{
			newNSEC3("E76CLEL5E7TQHRTFLTBVH0645NEKFJV9.skydns.test.	60 NSEC3 1 0 0 - E76CLEL5E7TQHRTFLTBVH0645NEKFJVA A AAAA SRV RRSIG"),
			newRRSIG("E76CLEL5E7TQHRTFLTBVH0645NEKFJV9.skydns.test.	60 RRSIG NSEC3 5 3 3600 20140814211641 20140807181641 51945 skydns.test. deadbeef"),
			newRRSIG("skydns.test.	60 RRSIG SOA 5 2 3600 20140814211641 20140807181641 51945 skydns.test. deadbeef"),
			newSOA("skydns.test.	60 SOA ns.dns.skydns.test. hostmaster.skydns.test. 1407445200 28800 7200 604800 60"),
		},
		Extra: []dns.RR{new(dns.OPT)},
	},
	// Reverse v4 local answer
	{
		Qname: "1.0.0.10.in-addr.arpa.", Qtype: dns.TypePTR,
		Answer: []dns.RR{newPTR("1.0.0.10.in-addr.arpa. 3600 PTR reverse.example.com.")},
	},
	// Reverse v6 local answer

	// Reverse forwarding answer, TODO(miek) does not work
	//	{
	//		Qname: "1.0.16.172.in-addr.arpa.", Qtype: dns.TypePTR,
	//		Rcode: dns.RcodeNameError,
	//		Ns:    []dns.RR{newSOA("16.172.in-addr.arpa. 10800 SOA localhost. nobody.invalid. 0 0 0 0 0")},
	//	},

	// Reverse no answer

	// Local data query
	{
		Qname: "local.dns.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("local.dns.skydns.test. 3600 A 10.0.0.1")},
	},
	// Author test
	{
		Qname: "skydns.test.", Qtype: dns.TypeTXT,
		chaos: true,
		Answer: []dns.RR{
			newTXT("skydns.test. 0 TXT \"Brian Ketelsen\""),
			newTXT("skydns.test. 0 TXT \"Erik St. Martin\""),
			newTXT("skydns.test. 0 TXT \"Michael Crosby\""),
			newTXT("skydns.test. 0 TXT \"Miek Gieben\""),
		},
	},
	// Author test 2
	{
		Qname: "authors.bind.", Qtype: dns.TypeTXT,
		chaos: true,
		Answer: []dns.RR{
			newTXT("authors.bind. 0 TXT \"Brian Ketelsen\""),
			newTXT("authors.bind. 0 TXT \"Erik St. Martin\""),
			newTXT("authors.bind. 0 TXT \"Michael Crosby\""),
			newTXT("authors.bind. 0 TXT \"Miek Gieben\""),
		},
	},
	// Author test, caps test
	{
		Qname: "AUTHOrs.BIND.", Qtype: dns.TypeTXT,
		chaos: true,
		Answer: []dns.RR{
			newTXT("AUTHOrs.BIND. 0 TXT \"Brian Ketelsen\""),
			newTXT("AUTHOrs.BIND. 0 TXT \"Erik St. Martin\""),
			newTXT("AUTHOrs.BIND. 0 TXT \"Michael Crosby\""),
			newTXT("AUTHOrs.BIND. 0 TXT \"Miek Gieben\""),
		},
	},
	// Author test 3, no answer.
	{
		Qname: "local.dns.skydns.test.", Qtype: dns.TypeA,
		Rcode: dns.RcodeServerFailure,
		chaos: true,
	},
	// HINFO Test, should be nodata for the apex
	{
		Qname: "skydns.test.", Qtype: dns.TypeHINFO,
		Ns: []dns.RR{newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0")},
	},
	// One IP, two ports open, ask for the IP only.
	{
		Qname: "bar.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("bar.skydns.test. 3600 A 192.168.0.1"),
		},
	},
	// Then ask for the SRV records.
	{
		Qname: "bar.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{
			newSRV("bar.skydns.test. 3600 SRV 10 50 443 bar.skydns.local."),
			// Issue 144 says x.bar.skydns.test should be bar.skydns.test
			newSRV("bar.skydns.test. 3600 SRV 10 50 80 bar.skydns.test."),
		},
		Extra: []dns.RR{
			newA("bar.skydns.test. 3600 A 192.168.0.1"),
		},
	},
	// Duplicate IP address test
	{
		Qname: "multiport.http.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("multiport.http.skydns.test. IN A 10.11.11.10")},
	},

	// Casing test
	{
		Qname: "uppeR.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("uppeR.skydns.test. IN A 127.0.0.1")},
	},
	{
		Qname: "upper.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{newA("upper.skydns.test. IN A 127.0.0.1")},
	},

	// SRV record with name that is internally resolvable.
	{
		Qname: "1.cname.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{newSRV("1.cname.skydns.test. IN SRV 10 100 0 104.server1.development.region1.skydns.test.")},
		Extra:  []dns.RR{newA("104.server1.development.region1.skydns.test. IN A 10.0.0.1")},
	},
	// SRV record with name that is internally resolvable. Get v4 and v6 records.
	{
		Qname: "both.v4v6.test.skydns.test.", Qtype: dns.TypeSRV,
		Answer: []dns.RR{newSRV("both.v4v6.test.skydns.test. IN SRV 10 100 0 ipaddr2.skydns.test.")},
		Extra: []dns.RR{
			newA("ipaddr2.skydns.test. IN A	172.16.1.1"),
			newAAAA("ipaddr2.skydns.test. IN AAAA 2001::8:8:8:8"),
		},
	},
	// MX Tests
	{
		// NODATA as this is not an Mail: true record.
		Qname: "100.server1.development.region1.skydns.test.", Qtype: dns.TypeMX,
		Ns: []dns.RR{
			newSOA("skydns.test. 3600 SOA ns.dns.skydns.test. hostmaster.skydns.test. 0 0 0 0 0"),
		},
	},
	{
		Qname: "b.mail.skydns.test.", Qtype: dns.TypeMX,
		Answer: []dns.RR{newMX("b.mail.skydns.test. IN MX 50 mx.miek.nl.")},
	},
	{
		// See issue #168
		Qname: "a.mail.skydns.test.", Qtype: dns.TypeMX,
		Answer: []dns.RR{newMX("a.mail.skydns.test. IN MX 50 mx.skydns.test.")},
		Extra: []dns.RR{
			newA("a.ipaddr.skydns.test. IN A 172.16.1.1"),
			newCNAME("mx.skydns.tests. IN CNAME a.ipaddr.skydns.test."),
		},
	},
	{
		Qname: "mx.skydns.test.", Qtype: dns.TypeMX,
		Answer: []dns.RR{
			newMX("mx.skydns.test. IN MX 30 a.ipaddr.skydns.test."),
		},
		Extra: []dns.RR{
			newA("a.ipaddr.skydns.test. A 172.16.1.1"),
		},
	},
	// Double CNAMEs in the additional
	{
		Qname: "a.mail2.skydns.test.", Qtype: dns.TypeMX,
		Answer: []dns.RR{
			newMX("a.mail2.skydns.test. IN MX 50 mx2.skydns.test."),
		},
		Extra: []dns.RR{
			newA("a.ipaddr.skydns.test. A 172.16.1.1"),
			newA("b.ipaddr.skydns.test. A 172.16.1.2"),
			// only one CNAME can be here, if we round-robin we randomly choose
			// without it, pick the first
			newCNAME("mx2.skydns.test. CNAME b.ipaddr.skydns.test."),
		},
	},
	// Groups
	{
		// hits the group 'g1' and only includes those records
		Qname: "dom.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("dom.skydns.test. IN A 127.0.0.1"),
			newA("dom.skydns.test. IN A 127.0.0.2"),
		},
	},
	{
		// One has group, the other has not...  Include the non-group always.
		Qname: "dom2.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("dom2.skydns.test. IN A 127.0.0.1"),
			newA("dom2.skydns.test. IN A 127.0.0.2"),
		},
	},
	{
		// The groups differ.
		Qname: "dom1.skydns.test.", Qtype: dns.TypeA,
		Answer: []dns.RR{
			newA("dom1.skydns.test. IN A 127.0.0.1"),
		},
	},
}

func newA(rr string) *dns.A           { r, _ := dns.NewRR(rr); return r.(*dns.A) }
func newAAAA(rr string) *dns.AAAA     { r, _ := dns.NewRR(rr); return r.(*dns.AAAA) }
func newCNAME(rr string) *dns.CNAME   { r, _ := dns.NewRR(rr); return r.(*dns.CNAME) }
func newSRV(rr string) *dns.SRV       { r, _ := dns.NewRR(rr); return r.(*dns.SRV) }
func newSOA(rr string) *dns.SOA       { r, _ := dns.NewRR(rr); return r.(*dns.SOA) }
func newNS(rr string) *dns.NS         { r, _ := dns.NewRR(rr); return r.(*dns.NS) }
func newDNSKEY(rr string) *dns.DNSKEY { r, _ := dns.NewRR(rr); return r.(*dns.DNSKEY) }
func newRRSIG(rr string) *dns.RRSIG   { r, _ := dns.NewRR(rr); return r.(*dns.RRSIG) }
func newNSEC3(rr string) *dns.NSEC3   { r, _ := dns.NewRR(rr); return r.(*dns.NSEC3) }
func newPTR(rr string) *dns.PTR       { r, _ := dns.NewRR(rr); return r.(*dns.PTR) }
func newTXT(rr string) *dns.TXT       { r, _ := dns.NewRR(rr); return r.(*dns.TXT) }
func newMX(rr string) *dns.MX         { r, _ := dns.NewRR(rr); return r.(*dns.MX) }

func TestDedup(t *testing.T) {
	m := new(dns.Msg)
	m.Answer = []dns.RR{
		newA("svc.ns.kubernetes.local. IN A 3.3.3.3"),
		newA("svc.ns.kubernetes.local. IN A 2.2.2.2"),
		newA("svc.ns.kubernetes.local. IN A 3.3.3.3"),
		newA("svc.ns.kubernetes.local. IN A 2.2.2.2"),
		newA("svc.ns.kubernetes.local. IN A 1.1.1.1"),
		newA("svc.ns.kubernetes.local. IN A 1.1.1.1"),
	}
	s := &server{}
	m = s.dedup(m)
	sort.Sort(rrSet(m.Answer))
	if len(m.Answer) != 3 {
		t.Fatalf("failing dedup: should have collapsed it to 3 records")
	}
	if dns.Field(m.Answer[0], 1) != "1.1.1.1" || dns.Field(m.Answer[1], 1) != "2.2.2.2" ||
		dns.Field(m.Answer[2], 1) != "3.3.3.3" {
		t.Fatalf("failing dedup: %s", m)
	}
}

func TestCacheTruncated(t *testing.T) {
	s := newTestServer(t, true)
	m := &dns.Msg{}
	m.SetQuestion("skydns.test.", dns.TypeSRV)
	m.Truncated = true
	s.rcache.InsertMessage(cache.QuestionKey(m.Question[0], false), m)

	// Now asking for this should result in a non-truncated answer.
	resp, _ := dns.Exchange(m, "127.0.0.1:"+StrPort)
	if resp.Truncated {
		t.Fatal("truncated bit should be false")
	}
}

func TestMsgOverflow(t *testing.T) {
	if testing.Short() {
                t.Skip("skipping test in short mode.")
        }

	s := newTestServer(t, false)
	defer s.Stop()

	c := new(dns.Client)
	m := new(dns.Msg)

	// TODO(miek): rethink how to enable metrics in tests.
	if !metricsDone {
		Metrics()
	}

	for i := 0; i < 2000; i++ {
		is := strconv.Itoa(i)
		m := &msg.Service{
			Host: "2001::" + is, Key: "machine" + is + ".machines.skydns.test.",
		}
		addService(t, s, m.Key, 0, m)
		defer delService(t, s, m.Key)
	}
	m.SetQuestion("machines.skydns.test.", dns.TypeSRV)
	resp, _, err := c.Exchange(m, "127.0.0.1:"+StrPort)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", resp)

	if resp.Rcode != dns.RcodeServerFailure {
		t.Fatalf("expecting server failure, got %d", resp.Rcode)
	}
}

func BenchmarkDNSSingleCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, true)
	defer s.Stop()

	serv := services[0]
	addService(t, s, serv.Key, 0, serv)
	defer delService(t, s, serv.Key)

	c := new(dns.Client)
	tc := dnsTestCases[0]
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}

func BenchmarkDNSWildcardCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, true)
	defer s.Stop()

	for _, serv := range services {
		m := &msg.Service{Host: serv.Host, Port: serv.Port}
		addService(t, s, serv.Key, 0, m)
		defer delService(t, s, serv.Key)
	}

	c := new(dns.Client)
	tc := dnsTestCases[8] // Wildcard Test
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}

func BenchmarkDNSSECSingleCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, true)
	defer s.Stop()

	serv := services[0]
	addService(t, s, serv.Key, 0, serv)
	defer delService(t, s, serv.Key)

	c := new(dns.Client)
	tc := dnsTestCases[0]
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)
	m.SetEdns0(4096, true)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}

func BenchmarkDNSSingleNoCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, false)
	defer s.Stop()

	serv := services[0]
	addService(t, s, serv.Key, 0, serv)
	defer delService(t, s, serv.Key)

	c := new(dns.Client)
	tc := dnsTestCases[0]
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}

func BenchmarkDNSWildcardNoCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, false)
	defer s.Stop()

	for _, serv := range services {
		m := &msg.Service{Host: serv.Host, Port: serv.Port}
		addService(t, s, serv.Key, 0, m)
		defer delService(t, s, serv.Key)
	}

	c := new(dns.Client)
	tc := dnsTestCases[8] // Wildcard Test
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}

func BenchmarkDNSSECSingleNoCache(b *testing.B) {
	b.StopTimer()
	t := new(testing.T)
	s := newTestServerDNSSEC(t, false)
	defer s.Stop()

	serv := services[0]
	addService(t, s, serv.Key, 0, serv)
	defer delService(t, s, serv.Key)

	c := new(dns.Client)
	tc := dnsTestCases[0]
	m := new(dns.Msg)
	m.SetQuestion(tc.Qname, tc.Qtype)
	m.SetEdns0(4096, true)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c.Exchange(m, "127.0.0.1:"+StrPort)
	}
}
