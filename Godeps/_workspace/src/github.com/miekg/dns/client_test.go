package dns

import (
	"strconv"
	"testing"
	"time"
)

func TestClientSync(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Unable to run test server: %v", err)
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
	if r != nil && r.Rcode != RcodeSuccess {
		t.Errorf("failed to get an valid answer\n%v", r)
	}
}

func TestClientEDNS0(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Unable to run test server: %v", err)
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
		t.Fatalf("Unable to run test server: %s", err)
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
	r, _, e := c.Exchange(m, addrstr)
	if e != nil {
		t.Logf("failed to exchange: %s", e.Error())
		t.Fail()
	}

	if r != nil && r.Rcode != RcodeSuccess {
		t.Log("failed to get a valid answer")
		t.Fail()
		t.Logf("%v\n", r)
	}

	txt := r.Extra[0].(*TXT).Txt[0]
	if txt != "Hello local edns" {
		t.Log("Unexpected result for miek.nl", txt, "!= Hello local edns")
		t.Fail()
	}

	// Validate the local options in the reply.
	got := r.Extra[1].(*OPT).Option[0].(*EDNS0_LOCAL).String()
	if got != optStr1 {
		t.Log("failed to get local edns0 answer; got %s, expected %s", got, optStr1)
		t.Fail()
		t.Logf("%v\n", r)
	}

	got = r.Extra[1].(*OPT).Option[1].(*EDNS0_LOCAL).String()
	if got != optStr2 {
		t.Log("failed to get local edns0 answer; got %s, expected %s", got, optStr2)
		t.Fail()
		t.Logf("%v\n", r)
	}
}

func TestSingleSingleInflight(t *testing.T) {
	HandleFunc("miek.nl.", HelloServer)
	defer HandleRemove("miek.nl.")

	s, addrstr, err := RunLocalUDPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Unable to run test server: %v", err)
	}
	defer s.Shutdown()

	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeDNSKEY)

	c := new(Client)
	c.SingleInflight = true
	nr := 10
	ch := make(chan time.Duration)
	for i := 0; i < nr; i++ {
		go func() {
			_, rtt, _ := c.Exchange(m, addrstr)
			ch <- rtt
		}()
	}
	i := 0
	var first time.Duration
	// With inflight *all* rtt are identical, and by doing actual lookups
	// the changes that this is a coincidence is small.
Loop:
	for {
		select {
		case rtt := <-ch:
			if i == 0 {
				first = rtt
			} else {
				if first != rtt {
					t.Errorf("all rtts should be equal.  got %d want %d", rtt, first)
				}
			}
			i++
			if i == 10 {
				break Loop
			}
		}
	}
}

// ExampleUpdateLeaseTSIG shows how to update a lease signed with TSIG.
func ExampleUpdateLeaseTSIG(t *testing.T) {
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
		t.Error(err)
	}
}
