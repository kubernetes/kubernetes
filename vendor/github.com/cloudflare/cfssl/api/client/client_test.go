package client

import (
	"github.com/cloudflare/cfssl/auth"
	"net"
	"testing"
)

var (
	testProvider auth.Provider
	testKey      = "0123456789ABCDEF0123456789ABCDEF"
	testAD       = []byte{1, 2, 3, 4} // IP address 1.2.3.4
)

func TestNewServer(t *testing.T) {
	s := NewServer("1.1.1.1:::123456789")
	if s != nil {
		t.Fatalf("fatal error, server created with too many colons %v", s)
	}

	s2 := NewServer("1.1.1.1:[]")
	if s != nil {
		t.Fatalf("%v", s2)

	}

	_, port, _ := net.SplitHostPort("")
	if port != "" {
		t.Fatalf("%v", port)

	}

	s = NewServer("127.0.0.1:8888")
	hosts := s.Hosts()
	if len(hosts) != 1 || hosts[0] != "http://127.0.0.1:8888" {
		t.Fatalf("expected [http://127.0.0.1:8888], but have %v", hosts)
	}

	s = NewServer("http://1.1.1.1:9999")
	hosts = s.Hosts()
	if len(hosts) != 1 || hosts[0] != "http://1.1.1.1:9999" {
		t.Fatalf("expected [http://1.1.1.1:9999], but have %v", hosts)
	}

	s = NewServer("https://1.1.1.1:8080")
	hosts = s.Hosts()
	if len(hosts) != 1 || hosts[0] != "https://1.1.1.1:8080" {
		t.Fatalf("expected [https://1.1.1.1:8080], but have %v", hosts)
	}
}

func TestInvalidPort(t *testing.T) {
	s := NewServer("1.1.1.1:99999999999999999999999999999")
	if s != nil {
		t.Fatalf("%v", s)
	}
}

func TestAuthSign(t *testing.T) {
	s := NewServer(".X")
	testProvider, _ = auth.New(testKey, nil)
	testRequest := []byte(`testing 1 2 3`)
	as, err := s.AuthSign(testRequest, testAD, testProvider)
	if as != nil || err == nil {
		t.Fatal("expected error with auth sign function")
	}
}

func TestDefaultAuthSign(t *testing.T) {
	testProvider, _ = auth.New(testKey, nil)
	s := NewAuthServer(".X", testProvider)
	testRequest := []byte(`testing 1 2 3`)
	as, err := s.Sign(testRequest)
	if as != nil || err == nil {
		t.Fatal("expected error with auth sign function")
	}
}

func TestSign(t *testing.T) {
	s := NewServer(".X")
	sign, err := s.Sign([]byte{5, 5, 5, 5})
	if sign != nil || err == nil {
		t.Fatalf("expected error with sign function")
	}
}

func TestNewServerGroup(t *testing.T) {
	s := NewServer("cfssl1.local:8888, cfssl2.local:8888, http://cfssl3.local:8888, http://cfssl4.local:8888")

	ogl, ok := s.(*orderedListGroup)
	if !ok {
		t.Fatalf("expected NewServer to return an ordered group list with a list of servers, instead got a %T = %+v", ogl, ogl)
	}

	if len(ogl.remotes) != 4 {
		t.Fatalf("expected the remote to have four servers, but it has %d", len(ogl.remotes))
	}

	hosts := ogl.Hosts()
	if len(hosts) != 4 {
		t.Fatalf("expected 2 hosts in the group, but have %d", len(hosts))
	}

	if hosts[0] != "http://cfssl1.local:8888" {
		t.Fatalf("expected to see http://cfssl1.local:8888, but saw %s",
			hosts[0])
	}

	if hosts[1] != "http://cfssl2.local:8888" {
		t.Fatalf("expected to see http://cfssl2.local:8888, but saw %s",
			hosts[1])
	}

	if hosts[2] != "http://cfssl3.local:8888" {
		t.Fatalf("expected to see http://cfssl1.local:8888, but saw %s",
			hosts[2])
	}

	if hosts[3] != "http://cfssl4.local:8888" {
		t.Fatalf("expected to see http://cfssl2.local:8888, but saw %s",
			hosts[3])
	}
}

func TestNewTLSServerGroup(t *testing.T) {
	s := NewServer("https://cfssl1.local:8888, https://cfssl2.local:8888")

	ogl, ok := s.(*orderedListGroup)
	if !ok {
		t.Fatalf("expected NewServer to return an ordered group list with a list of servers, instead got a %T = %+v", ogl, ogl)
	}

	if len(ogl.remotes) != 2 {
		t.Fatalf("expected the remote to have two servers, but it has %d", len(ogl.remotes))
	}

	hosts := ogl.Hosts()
	if len(hosts) != 2 {
		t.Fatalf("expected 2 hosts in the group, but have %d", len(hosts))
	}

	if hosts[0] != "https://cfssl1.local:8888" {
		t.Fatalf("expected to see https://cfssl1.local:8888, but saw %s",
			hosts[0])
	}

	if hosts[1] != "https://cfssl2.local:8888" {
		t.Fatalf("expected to see https://cfssl2.local:8888, but saw %s",
			hosts[1])
	}
}

func TestNewOGLGroup(t *testing.T) {
	strategy := StrategyFromString("ordered_list")
	if strategy == StrategyInvalid {
		t.Fatal("expected StrategyOrderedList as selected strategy but have StrategyInvalid")
	}

	if strategy != StrategyOrderedList {
		t.Fatalf("expected StrategyOrderedList (%d) but have %d", StrategyOrderedList, strategy)
	}

	rem, err := NewGroup([]string{"ca1.local,", "ca2.local"}, strategy)
	if err != nil {
		t.Fatalf("%v", err)
	}

	ogl, ok := rem.(*orderedListGroup)
	if !ok {
		t.Fatalf("expected to get an orderedListGroup but got %T", rem)
	}

	if len(ogl.remotes) != 2 {
		t.Fatalf("expected two remotes in the ordered group list but have %d", len(ogl.remotes))
	}
}
