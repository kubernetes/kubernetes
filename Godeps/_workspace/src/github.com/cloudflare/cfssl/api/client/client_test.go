package client

import (
	"net"
	"testing"

	"github.com/cloudflare/cfssl/auth"
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
}

func TestInvalidPort(t *testing.T) {
	s := NewServer("1.1.1.1:99999999999999999999999999999")
	if s != nil {
		t.Fatalf("%v", s)
	}
}

func TestAuthSign(t *testing.T) {
	s := NewServer("1.1")
	testProvider, _ = auth.New(testKey, nil)
	testRequest := []byte(`testing 1 2 3`)
	as, _ := s.AuthSign(testRequest, testAD, testProvider)
	if as != nil {
		t.Fatal("fatal error with auth sign function")
	}
}

func TestSign(t *testing.T) {
	s := NewServer("1.1")
	sign, _ := s.Sign([]byte{5, 5, 5, 5})
	if sign != nil {
		t.Fatalf("%v", sign)
	}
}

func TestNewServerGroup(t *testing.T) {
	s := NewServer("cfssl1.local:8888, cfssl2.local:8888")

	ogl, ok := s.(*orderedListGroup)
	if !ok {
		t.Fatalf("expected NewServer to return an ordered group list with a list of servers, instead got a %T = %+v", ogl, ogl)
	}

	if len(ogl.remotes) != 2 {
		t.Fatalf("expected the remote to have two servers, but it has %d", len(ogl.remotes))
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
