package whitelist

import (
	"encoding/json"
	"net"
	"testing"
)

func TestMarshalNet(t *testing.T) {
	tv := map[string]*BasicNet{
		"test-a": NewBasicNet(),
		"test-b": NewBasicNet(),
	}

	_, n, err := net.ParseCIDR("192.168.3.0/24")
	if err != nil {
		t.Fatalf("%v", err)
	}
	tv["test-a"].Add(n)

	_, n, err = net.ParseCIDR("192.168.7.0/24")
	if err != nil {
		t.Fatalf("%v", err)
	}
	tv["test-a"].Add(n)

	out, err := json.Marshal(tv)
	if err != nil {
		t.Fatalf("%v", err)
	}

	var tvPrime map[string]*BasicNet
	err = json.Unmarshal(out, &tvPrime)
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func TestMarshalNetFail(t *testing.T) {
	wl := NewBasicNet()
	badInput := `192.168.3.1/24,127.0.0.1/32`
	if err := wl.UnmarshalJSON([]byte(badInput)); err == nil {
		t.Fatal("Expected failure unmarshaling bad JSON input.")
	}

	badInput = `"192.168.3.1,127.0.0.256"`
	if err := wl.UnmarshalJSON([]byte(badInput)); err == nil {
		t.Fatal("Expected failure unmarshaling bad JSON input.")
	}
}

var testNet *BasicNet

func testAddNet(wl NetACL, ns string, t *testing.T) {
	_, n, err := net.ParseCIDR(ns)
	if err != nil {
		t.Fatalf("%v", err)
	}

	wl.Add(n)
}

func testDelNet(wl NetACL, ns string, t *testing.T) {
	_, n, err := net.ParseCIDR(ns)
	if err != nil {
		t.Fatalf("%v", err)
	}

	wl.Remove(n)
}

func TestAdd(t *testing.T) {
	// call this to make sure it doesn't panic, and to make sure
	// these code paths are executed.
	testNet = NewBasicNet()
	testNet.Add(nil)

	testAddNet(testNet, "192.168.3.0/24", t)
}

func TestRemove(t *testing.T) {
	testNet.Remove(nil)
	testDelNet(testNet, "192.168.1.1/32", t)
	testDelNet(testNet, "192.168.3.0/24", t)
}

func TestFailPermitted(t *testing.T) {
	var ip = []byte{0, 0}
	if testNet.Permitted(ip) {
		t.Fatal("Expected failure checking invalid IP address.")
	}
}
