package dns

import (
	"crypto/tls"
	"fmt"
	"net"
	"strconv"
	"testing"
	"time"
)

func TestClientSync(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeSOA)

	c := new(Client)
	r, _, err := c.Exchange(m, addrstr)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}
	if r != nil && r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
	// And now with plain Exchange().
	r, err = Exchange(m, addrstr)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}
	if r == nil || r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
}

func TestClientTLSSync(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	cert, err := tls.X509KeyPair(CertPEMBlock, KeyPEMBlock)
	if err != nil {
		t.Fatalf("unable to build certificate: %v", err)
	}

	config := tls.Config{
		Certificates: []tls.Certificate{cert},
	}

	s, addrstr, err := RunLocalTLSServer("127.0.0.1:0", &config)
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeSOA)

	c := new(Client)
	c.Net = "tcp-tls"
	c.TLSConfig = &tls.Config{
		InsecureSkipVerify: true,
	}

	r, _, err := c.Exchange(m, addrstr)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}
	if r != nil && r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
}

func TestClientSyncBadId(t *testing.T) {
	HandleFunc("miek.nl.", HelloServerBadId)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeSOA)

	c := new(Client)
	if _, _, err := c.Exchange(m, addrstr); err != ErrId {
		t.Errorf("did not find a bad Id")
	}
	// And now with plain Exchange().
	if _, err := Exchange(m, addrstr); err != ErrId {
		t.Errorf("did not find a bad Id")
	}
}

func TestClientEDNS0(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeDNSKEY)

	m.SetEdns0(2048, true)

	c := new(Client)
	r, _, err := c.Exchange(m, addrstr)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}

	if r != nil && r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
}

// Validates the transmission and parsing of local EDNS0 options.
func TestClientEDNS0Local(t *testing.T) {
	optStr1 := "1979:0x0707"
	optStr2 := strconv.Itoa(EDNS0LOCALSTART) + ":0x0601"

	handler := func(w ResponseWriter, req *Msg) {
		m := new(Msg)
		m.SetReply(req)

		m.Extra = make([]RR, 1, 2)
		m.Extra[0] = &TXT{Hdr: RR_Header{Name: m.Question[0].Name, Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}, Txt: []string{"Hello local edns"}}

		// If the local options are what we expect, then reflect them back.
		ec1 := req.Extra[0].(*OPT).Option[0].(*EDNS0_LOCAL).String()
		ec2 := req.Extra[0].(*OPT).Option[1].(*EDNS0_LOCAL).String()
		if ec1 == optStr1 && ec2 == optStr2 {
			m.Extra = append(m.Extra, req.Extra[0])
		}

		w.WriteMsg(m)
	}

	HandleFunc("miek.nl.", handler)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to run test server: %s", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeTXT)

	// Add two local edns options to the query.
	ec1 := &EDNS0_LOCAL{Code: 1979, Data: []byte{7, 7}}
	ec2 := &EDNS0_LOCAL{Code: EDNS0LOCALSTART, Data: []byte{6, 1}}
	o := &OPT{Hdr: RR_Header{Name: ".", Rrtype: TypeOPT}, Option: []EDNS0{ec1, ec2}}
	m.Extra = append(m.Extra, o)

	c := new(Client)
	r, _, err := c.Exchange(m, addrstr)
	if err != nil {
		t.Errorf("failed to exchange: %s", err)
	}

	if r != nil && r.Rcode != RcodeSuccess {
		t.Error("failed to get a valid answer")
		t.Logf("%v\n", r)
	}

	txt := r.Extra[0].(*TXT).Txt[0]
	if txt != "Hello local edns" {
		t.Error("Unexpected result for miek.nl", txt, "!= Hello local edns")
	}

	// Validate the local options in the reply.
	got := r.Extra[1].(*OPT).Option[0].(*EDNS0_LOCAL).String()
	if got != optStr1 {
		t.Errorf("failed to get local edns0 answer; got %s, expected %s", got, optStr1)
		t.Logf("%v\n", r)
	}

	got = r.Extra[1].(*OPT).Option[1].(*EDNS0_LOCAL).String()
	if got != optStr2 {
		t.Errorf("failed to get local edns0 answer; got %s, expected %s", got, optStr2)
		t.Logf("%v\n", r)
	}
}

// ExampleTsigSecret_updateLeaseTSIG shows how to update a lease signed with TSIG
func ExampleTsigSecret_updateLeaseTSIG() {
	m := new(Msg)
	m.SetUpdate("t.local.ip6.io.")
	rr, _ := NewRR("t.local.ip6.io. 30 A 127.0.0.1")
	rrs := make([]RR, 1)
	rrs[0] = rr
	m.Insert(rrs)

	leaseRr := new(OPT)
	leaseRr.Hdr.Name = "."
	leaseRr.Hdr.Rrtype = TypeOPT
	e := new(EDNS0_UL)
	e.Code = EDNS0UL
	e.Lease = 120
	leaseRr.Option = append(leaseRr.Option, e)
	m.Extra = append(m.Extra, leaseRr)

	c := new(Client)
	m.SetTsig("polvi.", HmacMD5, 300, time.Now().Unix())
	c.TsigSecret = map[string]string{"polvi.": "pRZgBrBvI4NAHZYhxmhs/Q=="}

	_, _, err := c.Exchange(m, "127.0.0.1:53")
	if err != nil {
		panic(err)
	}
}

func TestClientConn(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	// This uses TCP just to make it slightly different than TestClientSync
	s, addrstr, err := RunLocalTCPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeSOA)

	cn, err := Dial("tcp", addrstr)
	if err != nil {
		t.Errorf("failed to dial %s: %v", addrstr, err)
	}

	err = cn.WriteMsg(m)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}
	r, err := cn.ReadMsg()
	if r == nil || r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}

	err = cn.WriteMsg(m)
	if err != nil {
		t.Errorf("failed to exchange: %v", err)
	}
	h := new(Header)
	buf, err := cn.ReadMsgHeader(h)
	if buf == nil {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
	if int(h.Bits&0xF) != RcodeSuccess {
		t.Errorf("failed to get an valid answer in ReadMsgHeader\n%v", r)
	}
	if h.Ancount != 0 || h.Qdcount != 1 || h.Nscount != 0 || h.Arcount != 1 {
		t.Errorf("expected to have question and additional in response; got something else: %+v", h)
	}
	if err = r.Unpack(buf); err != nil {
		t.Errorf("unable to unpack message fully: %v", err)
	}
}

func TestTruncatedMsg(t *testing.T) {
	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeSRV)
	cnt := 10
	for i := 0; i < cnt; i++ {
		r := &SRV{
			Hdr:    RR_Header{Name: m.Question[0].Name, Rrtype: TypeSRV, Class: ClassINET, Ttl: 0},
			Port:   uint16(i + 8000),
			Target: "target.miek.nl.",
		}
		m.Answer = append(m.Answer, r)

		re := &A{
			Hdr: RR_Header{Name: m.Question[0].Name, Rrtype: TypeA, Class: ClassINET, Ttl: 0},
			A:   net.ParseIP(fmt.Sprintf("127.0.0.%d", i)).To4(),
		}
		m.Extra = append(m.Extra, re)
	}
	buf, err := m.Pack()
	if err != nil {
		t.Errorf("failed to pack: %v", err)
	}

	r := new(Msg)
	if err = r.Unpack(buf); err != nil {
		t.Errorf("unable to unpack message: %v", err)
	}
	if len(r.Answer) != cnt {
		t.Errorf("answer count after regular unpack doesn't match: %d", len(r.Answer))
	}
	if len(r.Extra) != cnt {
		t.Errorf("extra count after regular unpack doesn't match: %d", len(r.Extra))
	}

	m.Truncated = true
	buf, err = m.Pack()
	if err != nil {
		t.Errorf("failed to pack truncated: %v", err)
	}

	r = new(Msg)
	if err = r.Unpack(buf); err != nil && err != ErrTruncated {
		t.Errorf("unable to unpack truncated message: %v", err)
	}
	if !r.Truncated {
		t.Errorf("truncated message wasn't unpacked as truncated")
	}
	if len(r.Answer) != cnt {
		t.Errorf("answer count after truncated unpack doesn't match: %d", len(r.Answer))
	}
	if len(r.Extra) != cnt {
		t.Errorf("extra count after truncated unpack doesn't match: %d", len(r.Extra))
	}

	// Now we want to remove almost all of the extra records
	// We're going to loop over the extra to get the count of the size of all
	// of them
	off := 0
	buf1 := make([]byte, m.Len())
	for i := 0; i < len(m.Extra); i++ {
		off, err = PackRR(m.Extra[i], buf1, off, nil, m.Compress)
		if err != nil {
			t.Errorf("failed to pack extra: %v", err)
		}
	}

	// Remove all of the extra bytes but 10 bytes from the end of buf
	off -= 10
	buf1 = buf[:len(buf)-off]

	r = new(Msg)
	if err = r.Unpack(buf1); err != nil && err != ErrTruncated {
		t.Errorf("unable to unpack cutoff message: %v", err)
	}
	if !r.Truncated {
		t.Error("truncated cutoff message wasn't unpacked as truncated")
	}
	if len(r.Answer) != cnt {
		t.Errorf("answer count after cutoff unpack doesn't match: %d", len(r.Answer))
	}
	if len(r.Extra) != 0 {
		t.Errorf("extra count after cutoff unpack is not zero: %d", len(r.Extra))
	}

	// Now we want to remove almost all of the answer records too
	buf1 = make([]byte, m.Len())
	as := 0
	for i := 0; i < len(m.Extra); i++ {
		off1 := off
		off, err = PackRR(m.Extra[i], buf1, off, nil, m.Compress)
		as = off - off1
		if err != nil {
			t.Errorf("failed to pack extra: %v", err)
		}
	}

	// Keep exactly one answer left
	// This should still cause Answer to be nil
	off -= as
	buf1 = buf[:len(buf)-off]

	r = new(Msg)
	if err = r.Unpack(buf1); err != nil && err != ErrTruncated {
		t.Errorf("unable to unpack cutoff message: %v", err)
	}
	if !r.Truncated {
		t.Error("truncated cutoff message wasn't unpacked as truncated")
	}
	if len(r.Answer) != 0 {
		t.Errorf("answer count after second cutoff unpack is not zero: %d", len(r.Answer))
	}

	// Now leave only 1 byte of the question
	// Since the header is always 12 bytes, we just need to keep 13
	buf1 = buf[:13]

	r = new(Msg)
	err = r.Unpack(buf1)
	if err == nil || err == ErrTruncated {
		t.Errorf("error should not be ErrTruncated from question cutoff unpack: %v", err)
	}

	// Finally, if we only have the header, we should still return an error
	buf1 = buf[:12]

	r = new(Msg)
	if err = r.Unpack(buf1); err == nil || err != ErrTruncated {
		t.Errorf("error not ErrTruncated from header-only unpack: %v", err)
	}
}

func TestTimeout(t *testing.T) {
	// Set up a dummy UDP server that won't respond
	addr, err := net.ResolveUDPAddr("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("unable to resolve local udp address: %v", err)
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		t.Fatalf("unable to run test server: %v", err)
	}
	defer conn.Close()
	addrstr := conn.LocalAddr().String()

	// Message to send
	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeTXT)

	// Use a channel + timeout to ensure we don't get stuck if the
	// Client Timeout is not working properly
	done := make(chan struct{})

	timeout := time.Millisecond
	allowable := timeout + (10 * time.Millisecond)
	abortAfter := timeout + (100 * time.Millisecond)

	start := time.Now()

	go func() {
		c := &Client{Timeout: timeout}
		_, _, err := c.Exchange(m, addrstr)
		if err == nil {
			t.Error("no timeout using Client")
		}
		done <- struct{}{}
	}()

	select {
	case <-done:
	case <-time.After(abortAfter):
	}

	length := time.Since(start)

	if length > allowable {
		t.Errorf("exchange took longer (%v) than specified Timeout (%v)", length, timeout)
	}
}
