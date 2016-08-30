package agent

import (
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/miekg/dns"
)

func makeDNSServer(t *testing.T) (string, *DNSServer) {
	return makeDNSServerConfig(t, nil, nil)
}

func makeDNSServerConfig(
	t *testing.T,
	agentFn func(c *Config),
	dnsFn func(*DNSConfig)) (string, *DNSServer) {
	// Create the configs and apply the functions
	agentConf := nextConfig()
	if agentFn != nil {
		agentFn(agentConf)
	}
	dnsConf := &DNSConfig{}
	if dnsFn != nil {
		dnsFn(dnsConf)
	}

	// Add in the recursor if any
	if r := agentConf.DNSRecursor; r != "" {
		agentConf.DNSRecursors = append(agentConf.DNSRecursors, r)
	}

	// Start the server
	addr, _ := agentConf.ClientListener(agentConf.Addresses.DNS, agentConf.Ports.DNS)
	dir, agent := makeAgent(t, agentConf)
	server, err := NewDNSServer(agent, dnsConf, agent.logOutput,
		agentConf.Domain, addr.String(), agentConf.DNSRecursors)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	return dir, server
}

// makeRecursor creates a generic DNS server which always returns
// the provided reply. This is useful for mocking a DNS recursor with
// an expected result.
func makeRecursor(t *testing.T, answer []dns.RR) *dns.Server {
	dnsConf := nextConfig()
	dnsAddr := fmt.Sprintf("%s:%d", dnsConf.Addresses.DNS, dnsConf.Ports.DNS)
	mux := dns.NewServeMux()
	mux.HandleFunc(".", func(resp dns.ResponseWriter, msg *dns.Msg) {
		ans := &dns.Msg{Answer: answer[:]}
		ans.SetReply(msg)
		if err := resp.WriteMsg(ans); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	server := &dns.Server{
		Addr:    dnsAddr,
		Net:     "udp",
		Handler: mux,
	}
	go server.ListenAndServe()
	return server
}

// dnsCNAME returns a DNS CNAME record struct
func dnsCNAME(src, dest string) *dns.CNAME {
	return &dns.CNAME{
		Hdr: dns.RR_Header{
			Name:   dns.Fqdn(src),
			Rrtype: dns.TypeCNAME,
			Class:  dns.ClassINET,
		},
		Target: dns.Fqdn(dest),
	}
}

// dnsA returns a DNS A record struct
func dnsA(src, dest string) *dns.A {
	return &dns.A{
		Hdr: dns.RR_Header{
			Name:   dns.Fqdn(src),
			Rrtype: dns.TypeA,
			Class:  dns.ClassINET,
		},
		A: net.ParseIP(dest),
	}
}

func TestRecursorAddr(t *testing.T) {
	addr, err := recursorAddr("8.8.8.8")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if addr != "8.8.8.8:53" {
		t.Fatalf("bad: %v", addr)
	}
}

func TestDNS_NodeLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		TaggedAddresses: map[string]string{
			"wan": "127.0.0.2",
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("foo.node.consul.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aRec, ok := in.Answer[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	// Re-do the query, but specify the DC
	m = new(dns.Msg)
	m.SetQuestion("foo.node.dc1.consul.", dns.TypeANY)

	c = new(dns.Client)
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aRec, ok = in.Answer[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	// lookup a non-existing node, we should receive a SOA
	m = new(dns.Msg)
	m.SetQuestion("nofoo.node.dc1.consul.", dns.TypeANY)

	c = new(dns.Client)
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Ns) != 1 {
		t.Fatalf("Bad: %#v %#v", in, len(in.Answer))
	}

	soaRec, ok := in.Ns[0].(*dns.SOA)
	if !ok {
		t.Fatalf("Bad: %#v", in.Ns[0])
	}
	if soaRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Ns[0])
	}
}

func TestDNS_CaseInsensitiveNodeLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "Foo",
		Address:    "127.0.0.1",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("fOO.node.dc1.consul.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("empty lookup: %#v", in)
	}
}

func TestDNS_NodeLookup_PeriodName(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node with period in name
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo.bar",
		Address:    "127.0.0.1",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("foo.bar.node.consul.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aRec, ok := in.Answer[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
}

func TestDNS_NodeLookup_AAAA(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "::4242:4242",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("bar.node.consul.", dns.TypeAAAA)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aRec, ok := in.Answer[0].(*dns.AAAA)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.AAAA.String() != "::4242:4242" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
}

func TestDNS_NodeLookup_CNAME(t *testing.T) {
	recursor := makeRecursor(t, []dns.RR{
		dnsCNAME("www.google.com", "google.com"),
		dnsA("google.com", "1.2.3.4"),
	})
	defer recursor.Shutdown()

	dir, srv := makeDNSServerConfig(t, func(c *Config) {
		c.DNSRecursor = recursor.Addr
	}, nil)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "google",
		Address:    "www.google.com",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("google.node.consul.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should have the service record, CNAME record + A record
	if len(in.Answer) != 3 {
		t.Fatalf("Bad: %#v", in)
	}

	cnRec, ok := in.Answer[0].(*dns.CNAME)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if cnRec.Target != "www.google.com." {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if cnRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
}

func TestDNS_ReverseLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo2",
		Address:    "127.0.0.2",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("2.0.0.127.in-addr.arpa.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	ptrRec, ok := in.Answer[0].(*dns.PTR)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if ptrRec.Ptr != "foo2.node.dc1.consul." {
		t.Fatalf("Bad: %#v", ptrRec)
	}
}

func TestDNS_ReverseLookup_CustomDomain(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()
	srv.domain = dns.Fqdn("custom")

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo2",
		Address:    "127.0.0.2",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("2.0.0.127.in-addr.arpa.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	ptrRec, ok := in.Answer[0].(*dns.PTR)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if ptrRec.Ptr != "foo2.node.dc1.custom." {
		t.Fatalf("Bad: %#v", ptrRec)
	}
}

func TestDNS_ReverseLookup_IPV6(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "::4242:4242",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("2.4.2.4.2.4.2.4.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	ptrRec, ok := in.Answer[0].(*dns.PTR)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if ptrRec.Ptr != "bar.node.dc1.consul." {
		t.Fatalf("Bad: %#v", ptrRec)
	}
}

func TestDNS_ServiceLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node with a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		srvRec, ok := in.Answer[0].(*dns.SRV)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if srvRec.Port != 12345 {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Target != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}

		aRec, ok := in.Extra[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Name != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
	}

	// Lookup a non-existing service/query, we should receive an SOA.
	questions = []string{
		"nodb.service.consul.",
		"nope.query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Ns) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		soaRec, ok := in.Ns[0].(*dns.SOA)
		if !ok {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
		if soaRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
	}
}

func TestDNS_ServiceLookup_ServiceAddress(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node with a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Address: "127.0.0.2",
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		srvRec, ok := in.Answer[0].(*dns.SRV)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if srvRec.Port != 12345 {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Target != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}

		aRec, ok := in.Extra[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Name != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.A.String() != "127.0.0.2" {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
	}
}

func TestDNS_ServiceLookup_WanAddress(t *testing.T) {
	dir1, srv1 := makeDNSServerConfig(t,
		func(c *Config) {
			c.Datacenter = "dc1"
			c.TranslateWanAddrs = true
		}, nil)
	defer os.RemoveAll(dir1)
	defer srv1.Shutdown()

	dir2, srv2 := makeDNSServerConfig(t, func(c *Config) {
		c.Datacenter = "dc2"
		c.TranslateWanAddrs = true
	}, nil)
	defer os.RemoveAll(dir2)
	defer srv2.Shutdown()

	testutil.WaitForLeader(t, srv1.agent.RPC, "dc1")
	testutil.WaitForLeader(t, srv2.agent.RPC, "dc2")

	// Join WAN cluster
	addr := fmt.Sprintf("127.0.0.1:%d",
		srv1.agent.config.Ports.SerfWan)
	if _, err := srv2.agent.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(
		func() (bool, error) {
			return len(srv1.agent.WANMembers()) > 1, nil
		},
		func(err error) {
			t.Fatalf("Failed waiting for WAN join: %v", err)
		})

	// Register a remote node with a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc2",
			Node:       "foo",
			Address:    "127.0.0.1",
			TaggedAddresses: map[string]string{
				"wan": "127.0.0.2",
			},
			Service: &structs.NodeService{
				Service: "db",
			},
		}

		var out struct{}
		if err := srv2.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc2",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv2.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the SRV record via service and prepared query.
	questions := []string{
		"db.service.dc2.consul.",
		id + ".query.dc2.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv1.agent.config.ClientListener("", srv1.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		aRec, ok := in.Extra[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Name != "foo.node.dc2.consul." {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.A.String() != "127.0.0.2" {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
	}

	// Also check the A record directly
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeA)

		c := new(dns.Client)
		addr, _ := srv1.agent.config.ClientListener("", srv1.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		aRec, ok := in.Answer[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if aRec.Hdr.Name != question {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if aRec.A.String() != "127.0.0.2" {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
	}

	// Now query from the same DC and make sure we get the local address
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv2.agent.config.ClientListener("", srv2.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		aRec, ok := in.Extra[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Name != "foo.node.dc2.consul." {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
	}

	// Also check the A record directly from DC2
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeA)

		c := new(dns.Client)
		addr, _ := srv2.agent.config.ClientListener("", srv2.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		aRec, ok := in.Answer[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if aRec.Hdr.Name != question {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
	}
}

func TestDNS_CaseInsensitiveServiceLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node with a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "Db",
				Tags:    []string{"Master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query, as well as a name.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "somequery",
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Try some variations to make sure case doesn't matter.
	questions := []string{
		"master.db.service.consul.",
		"mASTER.dB.service.consul.",
		"MASTER.dB.service.consul.",
		"db.service.consul.",
		"DB.service.consul.",
		"Db.service.consul.",
		"somequery.query.consul.",
		"SomeQuery.query.consul.",
		"SOMEQUERY.query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("empty lookup: %#v", in)
		}
	}
}

func TestDNS_ServiceLookup_TagPeriod(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"v1.master"},
			Port:    12345,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("v1.master.db.service.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Port != 12345 {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Target != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", srvRec)
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Name != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_ServiceLookup_PreparedQueryNamePeriod(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node with a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register a prepared query with a period in the name.
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "some.query.we.like",
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}

		var id string
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	m := new(dns.Msg)
	m.SetQuestion("some.query.we.like.query.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Port != 12345 {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Target != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", srvRec)
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Name != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_ServiceLookup_Dedup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a single node with multiple instances of a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				ID:      "db2",
				Service: "db",
				Tags:    []string{"slave"},
				Port:    12345,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				ID:      "db3",
				Service: "db",
				Tags:    []string{"slave"},
				Port:    12346,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query, make sure only
	// one IP is returned.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		aRec, ok := in.Answer[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
	}
}

func TestDNS_ServiceLookup_Dedup_SRV(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a single node with multiple instances of a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				ID:      "db2",
				Service: "db",
				Tags:    []string{"slave"},
				Port:    12345,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				ID:      "db3",
				Service: "db",
				Tags:    []string{"slave"},
				Port:    12346,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query, make sure only
	// one IP is returned and two unique ports are returned.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 2 {
			t.Fatalf("Bad: %#v", in)
		}

		srvRec, ok := in.Answer[0].(*dns.SRV)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if srvRec.Port != 12345 && srvRec.Port != 12346 {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Target != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", srvRec)
		}

		srvRec, ok = in.Answer[1].(*dns.SRV)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[1])
		}
		if srvRec.Port != 12346 && srvRec.Port != 12345 {
			t.Fatalf("Bad: %#v", srvRec)
		}
		if srvRec.Port == in.Answer[0].(*dns.SRV).Port {
			t.Fatalf("should be a different port")
		}
		if srvRec.Target != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", srvRec)
		}

		aRec, ok := in.Extra[0].(*dns.A)
		if !ok {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.Hdr.Name != "foo.node.dc1.consul." {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Extra[0])
		}
	}
}

func TestDNS_Recurse(t *testing.T) {
	recursor := makeRecursor(t, []dns.RR{dnsA("apple.com", "1.2.3.4")})
	defer recursor.Shutdown()

	dir, srv := makeDNSServerConfig(t, func(c *Config) {
		c.DNSRecursor = recursor.Addr
	}, nil)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	m := new(dns.Msg)
	m.SetQuestion("apple.com.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) == 0 {
		t.Fatalf("Bad: %#v", in)
	}
	if in.Rcode != dns.RcodeSuccess {
		t.Fatalf("Bad: %#v", in)
	}
}

func TestDNS_ServiceLookup_FilterCritical(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register nodes with health checks in various states.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID: "serf",
				Name:    "serf",
				Status:  structs.HealthCritical,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args2 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "bar",
			Address:    "127.0.0.2",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID: "serf",
				Name:    "serf",
				Status:  structs.HealthCritical,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args2, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args3 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "bar",
			Address:    "127.0.0.2",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthCritical,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args3, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args4 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "baz",
			Address:    "127.0.0.3",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args4, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args5 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "quux",
			Address:    "127.0.0.4",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthWarning,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args5, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Only 4 and 5 are not failing, so we should get 2 answers
		if len(in.Answer) != 2 {
			t.Fatalf("Bad: %#v", in)
		}

		ips := make(map[string]bool)
		for _, resp := range in.Answer {
			aRec := resp.(*dns.A)
			ips[aRec.A.String()] = true
		}

		if !ips["127.0.0.3"] {
			t.Fatalf("Bad: %#v should contain 127.0.0.3 (state healthy)", in)
		}
		if !ips["127.0.0.4"] {
			t.Fatalf("Bad: %#v should contain 127.0.0.4 (state warning)", in)
		}
	}
}

func TestDNS_ServiceLookup_OnlyFailing(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register nodes with all health checks in a critical state.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID: "serf",
				Name:    "serf",
				Status:  structs.HealthCritical,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args2 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "bar",
			Address:    "127.0.0.2",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID: "serf",
				Name:    "serf",
				Status:  structs.HealthCritical,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args2, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args3 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "bar",
			Address:    "127.0.0.2",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthCritical,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args3, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// All 3 are failing, so we should get 0 answers and an NXDOMAIN response
		if len(in.Answer) != 0 {
			t.Fatalf("Bad: %#v", in)
		}

		if in.Rcode != dns.RcodeNameError {
			t.Fatalf("Bad: %#v", in)
		}
	}
}

func TestDNS_ServiceLookup_OnlyPassing(t *testing.T) {
	dir, srv := makeDNSServerConfig(t, nil, func(c *DNSConfig) {
		c.OnlyPassing = true
	})
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register nodes with health checks in various states.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthPassing,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args2 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "bar",
			Address:    "127.0.0.2",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthWarning,
			},
		}

		if err := srv.agent.RPC("Catalog.Register", args2, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args3 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "baz",
			Address:    "127.0.0.3",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthCritical,
			},
		}

		if err := srv.agent.RPC("Catalog.Register", args3, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args4 := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "quux",
			Address:    "127.0.0.4",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
			Check: &structs.HealthCheck{
				CheckID:   "db",
				Name:      "db",
				ServiceID: "db",
				Status:    structs.HealthUnknown,
			},
		}

		if err := srv.agent.RPC("Catalog.Register", args4, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service:     "db",
					OnlyPassing: true,
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"db.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Only 1 is passing, so we should only get 1 answer
		if len(in.Answer) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		resp := in.Answer[0]
		aRec := resp.(*dns.A)

		if aRec.A.String() != "127.0.0.1" {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
	}
}

func TestDNS_ServiceLookup_Randomize(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a large set of nodes.
	for i := 0; i < 3*maxServiceResponses; i++ {
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       fmt.Sprintf("foo%d", i),
			Address:    fmt.Sprintf("127.0.0.%d", i+1),
			Service: &structs.NodeService{
				Service: "web",
				Port:    8000,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "web",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query. Ensure the
	// response is randomized each time.
	questions := []string{
		"web.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		uniques := map[string]struct{}{}
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		for i := 0; i < 10; i++ {
			m := new(dns.Msg)
			m.SetQuestion(question, dns.TypeANY)

			c := new(dns.Client)
			in, _, err := c.Exchange(m, addr.String())
			if err != nil {
				t.Fatalf("err: %v", err)
			}

			// Response length should be truncated and we should get
			// an A record for each response.
			if len(in.Answer) != maxServiceResponses {
				t.Fatalf("Bad: %#v", len(in.Answer))
			}

			// Collect all the names.
			var names []string
			for _, rec := range in.Answer {
				switch v := rec.(type) {
				case *dns.SRV:
					names = append(names, v.Target)
				case *dns.A:
					names = append(names, v.A.String())
				}
			}
			nameS := strings.Join(names, "|")

			// Tally the results.
			uniques[nameS] = struct{}{}
		}

		// Give some wiggle room. Since the responses are randomized and
		// there is a finite number of combinations, requiring 0
		// duplicates every test run eventually gives us failures.
		if len(uniques) < 2 {
			t.Fatalf("unique response ratio too low: %d/10\n%v", len(uniques), uniques)
		}
	}
}

func TestDNS_ServiceLookup_Truncate(t *testing.T) {
	dir, srv := makeDNSServerConfig(t, nil, func(c *DNSConfig) {
		c.EnableTruncate = true
	})
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register nodes a large number of nodes.
	for i := 0; i < 3*maxServiceResponses; i++ {
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       fmt.Sprintf("foo%d", i),
			Address:    fmt.Sprintf("127.0.0.%d", i+1),
			Service: &structs.NodeService{
				Service: "web",
				Port:    8000,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "web",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query. Ensure the
	// response is truncated each time.
	questions := []string{
		"web.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		c := new(dns.Client)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil && err != dns.ErrTruncated {
			t.Fatalf("err: %v", err)
		}

		// Check for the truncate bit
		if !in.Truncated {
			t.Fatalf("should have truncate bit")
		}
	}
}

func TestDNS_ServiceLookup_LargeResponses(t *testing.T) {
	dir, srv := makeDNSServerConfig(t, nil, func(c *DNSConfig) {
		c.EnableTruncate = true
	})
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	longServiceName := "this-is-a-very-very-very-very-very-long-name-for-a-service"

	// Register 3 nodes
	for i := 0; i < 3; i++ {
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       fmt.Sprintf("foo%d", i),
			Address:    fmt.Sprintf("127.0.0.%d", i+1),
			Service: &structs.NodeService{
				Service: longServiceName,
				Tags:    []string{"master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: longServiceName,
				Service: structs.ServiceQuery{
					Service: longServiceName,
					Tags:    []string{"master"},
				},
			},
		}
		var id string
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"_" + longServiceName + "._master.service.consul.",
		longServiceName + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil && err != dns.ErrTruncated {
			t.Fatalf("err: %v", err)
		}

		// Make sure the response size is RFC 1035-compliant for UDP messages
		if in.Len() > 512 {
			t.Fatalf("Bad: %#v", in.Len())
		}

		// We should only have two answers now
		if len(in.Answer) != 2 {
			t.Fatalf("Bad: %#v", len(in.Answer))
		}

		// Check for the truncate bit
		if !in.Truncated {
			t.Fatalf("should have truncate bit")
		}
	}
}

func TestDNS_ServiceLookup_MaxResponses(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a large number of nodes.
	for i := 0; i < 6*maxServiceResponses; i++ {
		nodeAddress := fmt.Sprintf("127.0.0.%d", i+1)
		if i > 3 {
			nodeAddress = fmt.Sprintf("fe80::%d", i+1)
		}
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       fmt.Sprintf("foo%d", i),
			Address:    nodeAddress,
			Service: &structs.NodeService{
				Service: "web",
				Port:    8000,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "web",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"web.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		c := new(dns.Client)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 3 {
			t.Fatalf("should receive 3 answers for ANY")
		}

		m.SetQuestion(question, dns.TypeA)
		in, _, err = c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 3 {
			t.Fatalf("should receive 3 answers for A")
		}

		m.SetQuestion(question, dns.TypeAAAA)
		in, _, err = c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Answer) != 3 {
			t.Fatalf("should receive 3 answers for AAAA")
		}
	}
}

func TestDNS_ServiceLookup_CNAME(t *testing.T) {
	recursor := makeRecursor(t, []dns.RR{
		dnsCNAME("www.google.com", "google.com"),
		dnsA("google.com", "1.2.3.4"),
	})
	defer recursor.Shutdown()

	dir, srv := makeDNSServerConfig(t, func(c *Config) {
		c.DNSRecursor = recursor.Addr
	}, nil)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node with a name for an address.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "google",
			Address:    "www.google.com",
			Service: &structs.NodeService{
				Service: "search",
				Port:    80,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register an equivalent prepared query.
	var id string
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Service: structs.ServiceQuery{
					Service: "search",
				},
			},
		}
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Look up the service directly and via prepared query.
	questions := []string{
		"search.service.consul.",
		id + ".query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeANY)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		// Service CNAME, google CNAME, google A record
		if len(in.Answer) != 3 {
			t.Fatalf("Bad: %#v", in)
		}

		// Should have service CNAME
		cnRec, ok := in.Answer[0].(*dns.CNAME)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}
		if cnRec.Target != "www.google.com." {
			t.Fatalf("Bad: %#v", in.Answer[0])
		}

		// Should have google CNAME
		cnRec, ok = in.Answer[1].(*dns.CNAME)
		if !ok {
			t.Fatalf("Bad: %#v", in.Answer[1])
		}
		if cnRec.Target != "google.com." {
			t.Fatalf("Bad: %#v", in.Answer[1])
		}

		// Check we recursively resolve
		if _, ok := in.Answer[2].(*dns.A); !ok {
			t.Fatalf("Bad: %#v", in.Answer[2])
		}
	}
}

func TestDNS_NodeLookup_TTL(t *testing.T) {
	recursor := makeRecursor(t, []dns.RR{
		dnsCNAME("www.google.com", "google.com"),
		dnsA("google.com", "1.2.3.4"),
	})
	defer recursor.Shutdown()

	dir, srv := makeDNSServerConfig(t, func(c *Config) {
		c.DNSRecursor = recursor.Addr
	}, func(c *DNSConfig) {
		c.NodeTTL = 10 * time.Second
		c.AllowStale = true
		c.MaxStale = time.Second
	})
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("foo.node.consul.", dns.TypeANY)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aRec, ok := in.Answer[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	// Register node with IPv6
	args = &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "::4242:4242",
	}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check an IPv6 record
	m = new(dns.Msg)
	m.SetQuestion("bar.node.consul.", dns.TypeANY)

	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	aaaaRec, ok := in.Answer[0].(*dns.AAAA)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aaaaRec.AAAA.String() != "::4242:4242" {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if aaaaRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	// Register node with CNAME
	args = &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "google",
		Address:    "www.google.com",
	}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m = new(dns.Msg)
	m.SetQuestion("google.node.consul.", dns.TypeANY)

	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should have the CNAME record + a few A records
	if len(in.Answer) < 2 {
		t.Fatalf("Bad: %#v", in)
	}

	cnRec, ok := in.Answer[0].(*dns.CNAME)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if cnRec.Target != "www.google.com." {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if cnRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
}

func TestDNS_ServiceLookup_TTL(t *testing.T) {
	confFn := func(c *DNSConfig) {
		c.ServiceTTL = map[string]time.Duration{
			"db": 10 * time.Second,
			"*":  5 * time.Second,
		}
		c.AllowStale = true
		c.MaxStale = time.Second
	}
	dir, srv := makeDNSServerConfig(t, nil, confFn)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node with 2 services
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    12345,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	args = &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "api",
			Port:    2222,
		},
	}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("db.service.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}

	m = new(dns.Msg)
	m.SetQuestion("api.service.consul.", dns.TypeSRV)
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok = in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Hdr.Ttl != 5 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok = in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 5 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_PreparedQuery_TTL(t *testing.T) {
	confFn := func(c *DNSConfig) {
		c.ServiceTTL = map[string]time.Duration{
			"db": 10 * time.Second,
			"*":  5 * time.Second,
		}
		c.AllowStale = true
		c.MaxStale = time.Second
	}
	dir, srv := makeDNSServerConfig(t, nil, confFn)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a node and a service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "db",
				Tags:    []string{"master"},
				Port:    12345,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foo",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "api",
				Port:    2222,
			},
		}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register prepared queries with and without a TTL set for "db", as
	// well as one for "api".
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "db-ttl",
				Service: structs.ServiceQuery{
					Service: "db",
				},
				DNS: structs.QueryDNSOptions{
					TTL: "18s",
				},
			},
		}

		var id string
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "db-nottl",
				Service: structs.ServiceQuery{
					Service: "db",
				},
			},
		}

		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "api-nottl",
				Service: structs.ServiceQuery{
					Service: "api",
				},
			},
		}

		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Make sure the TTL is set when requested, and overrides the agent-
	// specific config since the query takes precedence.
	m := new(dns.Msg)
	m.SetQuestion("db-ttl.query.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Hdr.Ttl != 18 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 18 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}

	// And the TTL should take the service-specific value from the agent's
	// config otherwise.
	m = new(dns.Msg)
	m.SetQuestion("db-nottl.query.consul.", dns.TypeSRV)
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok = in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok = in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 10 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}

	// If there's no query TTL and no service-specific value then the wild
	// card value should be used.
	m = new(dns.Msg)
	m.SetQuestion("api-nottl.query.consul.", dns.TypeSRV)
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok = in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Hdr.Ttl != 5 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok = in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 5 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_ServiceLookup_SRV_RFC(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    12345,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("_db._master.service.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Port != 12345 {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Target != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Name != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_ServiceLookup_SRV_RFC_TCP_Default(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register node
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    12345,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	m := new(dns.Msg)
	m.SetQuestion("_db._tcp.service.consul.", dns.TypeSRV)

	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	srvRec, ok := in.Answer[0].(*dns.SRV)
	if !ok {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}
	if srvRec.Port != 12345 {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Target != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", srvRec)
	}
	if srvRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Answer[0])
	}

	aRec, ok := in.Extra[0].(*dns.A)
	if !ok {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Name != "foo.node.dc1.consul." {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.A.String() != "127.0.0.1" {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
	if aRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Extra[0])
	}
}

func TestDNS_ServiceLookup_FilterACL(t *testing.T) {
	confFn := func(c *Config) {
		c.ACLMasterToken = "root"
		c.ACLDatacenter = "dc1"
		c.ACLDownPolicy = "deny"
		c.ACLDefaultPolicy = "deny"
	}
	dir, srv := makeDNSServerConfig(t, confFn, nil)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a service
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "foo",
			Port:    12345,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Set up the DNS query
	c := new(dns.Client)
	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
	m := new(dns.Msg)
	m.SetQuestion("foo.service.consul.", dns.TypeA)

	// Query with the root token. Should get results.
	srv.agent.config.ACLToken = "root"
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(in.Answer) != 1 {
		t.Fatalf("Bad: %#v", in)
	}

	// Query with a non-root token without access. Should get nothing.
	srv.agent.config.ACLToken = "anonymous"
	in, _, err = c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(in.Answer) != 0 {
		t.Fatalf("Bad: %#v", in)
	}
}

func TestDNS_NonExistingLookup(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)

	// lookup a non-existing node, we should receive a SOA
	m := new(dns.Msg)
	m.SetQuestion("nonexisting.consul.", dns.TypeANY)

	c := new(dns.Client)
	in, _, err := c.Exchange(m, addr.String())
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(in.Ns) != 1 {
		t.Fatalf("Bad: %#v %#v", in, len(in.Answer))
	}

	soaRec, ok := in.Ns[0].(*dns.SOA)
	if !ok {
		t.Fatalf("Bad: %#v", in.Ns[0])
	}
	if soaRec.Hdr.Ttl != 0 {
		t.Fatalf("Bad: %#v", in.Ns[0])
	}
}

func TestDNS_NonExistingLookupEmptyAorAAAA(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Register a v6-only service and a v4-only service.
	{
		args := &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foov6",
			Address:    "fe80::1",
			Service: &structs.NodeService{
				Service: "webv6",
				Port:    8000,
			},
		}

		var out struct{}
		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       "foov4",
			Address:    "127.0.0.1",
			Service: &structs.NodeService{
				Service: "webv4",
				Port:    8000,
			},
		}

		if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Register equivalent prepared queries.
	{
		args := &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "webv4",
				Service: structs.ServiceQuery{
					Service: "webv4",
				},
			},
		}

		var id string
		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}

		args = &structs.PreparedQueryRequest{
			Datacenter: "dc1",
			Op:         structs.PreparedQueryCreate,
			Query: &structs.PreparedQuery{
				Name: "webv6",
				Service: structs.ServiceQuery{
					Service: "webv6",
				},
			},
		}

		if err := srv.agent.RPC("PreparedQuery.Apply", args, &id); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Check for ipv6 records on ipv4-only service directly and via the
	// prepared query.
	questions := []string{
		"webv4.service.consul.",
		"webv4.query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeAAAA)

		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		c := new(dns.Client)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Ns) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		soaRec, ok := in.Ns[0].(*dns.SOA)
		if !ok {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
		if soaRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}

		if in.Rcode != dns.RcodeSuccess {
			t.Fatalf("Bad: %#v", in)
		}
	}

	// Check for ipv4 records on ipv6-only service directly and via the
	// prepared query.
	questions = []string{
		"webv6.service.consul.",
		"webv6.query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeA)

		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		c := new(dns.Client)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Ns) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		soaRec, ok := in.Ns[0].(*dns.SOA)
		if !ok {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
		if soaRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}

		if in.Rcode != dns.RcodeSuccess {
			t.Fatalf("Bad: %#v", in)
		}
	}
}

func TestDNS_PreparedQuery_AllowStale(t *testing.T) {
	confFn := func(c *DNSConfig) {
		c.AllowStale = true
		c.MaxStale = time.Second
	}
	dir, srv := makeDNSServerConfig(t, nil, confFn)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	m := MockPreparedQuery{}
	if err := srv.agent.InjectEndpoint("PreparedQuery", &m); err != nil {
		t.Fatalf("err: %v", err)
	}

	m.executeFn = func(args *structs.PreparedQueryExecuteRequest, reply *structs.PreparedQueryExecuteResponse) error {
		// Return a response that's perpetually too stale.
		reply.LastContact = 2 * time.Second
		return nil
	}

	// Make sure that the lookup terminates and results in an SOA since
	// the query doesn't exist.
	{
		m := new(dns.Msg)
		m.SetQuestion("nope.query.consul.", dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Ns) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		soaRec, ok := in.Ns[0].(*dns.SOA)
		if !ok {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
		if soaRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
	}
}

func TestDNS_InvalidQueries(t *testing.T) {
	dir, srv := makeDNSServer(t)
	defer os.RemoveAll(dir)
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Try invalid forms of queries that should hit the special invalid case
	// of our query parser.
	questions := []string{
		"consul.",
		"node.consul.",
		"service.consul.",
		"query.consul.",
	}
	for _, question := range questions {
		m := new(dns.Msg)
		m.SetQuestion(question, dns.TypeSRV)

		c := new(dns.Client)
		addr, _ := srv.agent.config.ClientListener("", srv.agent.config.Ports.DNS)
		in, _, err := c.Exchange(m, addr.String())
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		if len(in.Ns) != 1 {
			t.Fatalf("Bad: %#v", in)
		}

		soaRec, ok := in.Ns[0].(*dns.SOA)
		if !ok {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
		if soaRec.Hdr.Ttl != 0 {
			t.Fatalf("Bad: %#v", in.Ns[0])
		}
	}
}
