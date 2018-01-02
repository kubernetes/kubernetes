package whitelist

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net"
	"net/http"
	"testing"
	"time"
)

type StringLookup struct{}

func (lu StringLookup) Address(args ...interface{}) (net.IP, error) {
	if len(args) != 1 {
		return nil, errors.New("whitelist: lookup requires a string")
	}

	var s string
	switch arg := args[0].(type) {
	case string:
		s = arg
	default:
		return nil, errors.New("whitelist: lookup requires a string")
	}

	ip := net.ParseIP(s)
	if ip == nil {
		return nil, errors.New("whitelist: no address found")
	}
	return ip, nil
}

var slu StringLookup

func checkIPString(wl ACL, addr string, t *testing.T) bool {
	ip, err := slu.Address(addr)
	if err != nil {
		t.Fatalf("%v", err)
	}

	return wl.Permitted(ip)
}

func addIPString(wl HostACL, addr string, t *testing.T) {
	ip, err := slu.Address(addr)
	if err != nil {
		t.Fatalf("%v", err)
	}

	wl.Add(ip)
}

func delIPString(wl HostACL, addr string, t *testing.T) {
	ip, err := slu.Address(addr)
	if err != nil {
		t.Fatalf("%v", err)
	}

	wl.Remove(ip)
}

func TestBasicWhitelist(t *testing.T) {
	wl := NewBasic()

	if checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have denied address")
	}

	addIPString(wl, "127.0.0.1", t)
	if !checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have permitted address")
	}

	delIPString(wl, "127.0.0.1", t)
	if checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have denied address")
	}

	addIPString(wl, "::1", t)
	if checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have denied address")
	}

	wl.Add(nil)
	wl.Remove(nil)
	wl.Permitted(nil)
}

func TestStubWhitelist(t *testing.T) {
	wl := NewHostStub()

	if !checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have permitted address")
	}

	addIPString(wl, "127.0.0.1", t)
	if !checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have permitted address")
	}

	delIPString(wl, "127.0.0.1", t)
	if !checkIPString(wl, "127.0.0.1", t) {
		t.Fatal("whitelist should have permitted address")
	}
}

func TestMarshalHost(t *testing.T) {
	tv := map[string]*Basic{
		"test-a": NewBasic(),
		"test-b": NewBasic(),
	}

	ip := net.ParseIP("192.168.3.1")
	tv["test-a"].Add(ip)

	ip = net.ParseIP("192.168.3.2")
	tv["test-a"].Add(ip)

	if len(tv["test-a"].whitelist) != 2 {
		t.Fatalf("Expected whitelist to have 2 addresses, but have %d", len(tv["test-a"].whitelist))
	}

	out, err := json.Marshal(tv)
	if err != nil {
		t.Fatalf("%v", err)
	}

	var tvPrime map[string]*Basic
	err = json.Unmarshal(out, &tvPrime)
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func TestMarshalHostFail(t *testing.T) {
	wl := NewBasic()
	badInput := `192.168.3.1/24,127.0.0.1/32`
	if err := wl.UnmarshalJSON([]byte(badInput)); err == nil {
		t.Fatal("Expected failure unmarshaling bad JSON input.")
	}

	badInput = `"192.168.3.1/32,127.0.0.252/32"`
	if err := wl.UnmarshalJSON([]byte(badInput)); err == nil {
		t.Fatal("Expected failure unmarshaling bad JSON input.")
	}
}

var shutdown = make(chan struct{}, 1)
var proceed = make(chan struct{}, 0)

func setupTestServer(t *testing.T, wl ACL) {
	ln, err := net.Listen("tcp", "127.0.0.1:4141")
	if err != nil {
		t.Fatalf("%v", err)
	}
	proceed <- struct{}{}
	for {
		select {
		case _, ok := <-shutdown:
			if !ok {
				return
			}
		default:
			conn, err := ln.Accept()
			if err != nil {
				t.Fatalf("%v", err)
			}
			go handleTestConnection(conn, wl, t)
		}
	}
}

func handleTestConnection(conn net.Conn, wl ACL, t *testing.T) {
	defer conn.Close()
	ip, err := NetConnLookup(conn)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if wl.Permitted(ip) {
		conn.Write([]byte("OK"))
	} else {
		conn.Write([]byte("NO"))
	}
}

func TestNetConn(t *testing.T) {
	wl := NewBasic()
	defer close(shutdown)

	go setupTestServer(t, wl)
	<-proceed

	conn, err := net.Dial("tcp", "127.0.0.1:4141")
	if err != nil {
		t.Fatalf("%v", err)
	}
	body, err := ioutil.ReadAll(conn)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if string(body) != "NO" {
		t.Fatalf("Expected NO, but received %s", body)
	}
	conn.Close()

	addIPString(wl, "127.0.0.1", t)
	conn, err = net.Dial("tcp", "127.0.0.1:4141")
	if err != nil {
		t.Fatalf("%v", err)
	}
	body, err = ioutil.ReadAll(conn)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if string(body) != "OK" {
		t.Fatalf("Expected OK, but received %s", body)
	}
	conn.Close()

}

func TestBasicDumpLoad(t *testing.T) {
	wl := NewBasic()
	addIPString(wl, "127.0.0.1", t)
	addIPString(wl, "10.0.1.15", t)
	addIPString(wl, "192.168.1.5", t)

	out := DumpBasic(wl)
	loaded, err := LoadBasic(out)
	if err != nil {
		t.Fatalf("%v", err)
	}

	dumped := DumpBasic(loaded)
	if !bytes.Equal(out, dumped) {
		t.Fatalf("dump -> load failed")
	}
}

func TestBasicFailedLoad(t *testing.T) {
	dump := []byte("192.168.1.5\n192.168.2.3\n192.168.2\n192.168.3.1")
	if _, err := LoadBasic(dump); err == nil {
		t.Fatalf("LoadBasic should fail on invalid IP address")
	}
}

func TestNetConnChecks(t *testing.T) {
	if _, err := NetConnLookup(nil); err == nil {
		t.Fatal("Address should fail with an invalid argument")
	}
}

func TestHTTPRequestLookup(t *testing.T) {
	if _, err := HTTPRequestLookup(nil); err == nil {
		t.Fatal("Address should fail with an invalid argument")
	}

	req := new(http.Request)
	if _, err := HTTPRequestLookup(req); err == nil {
		t.Fatal("Address should fail with an invalid argument")
	}

}

type stubConn struct {
	Fake   bool
	Global bool
}

func (conn *stubConn) Read(b []byte) (n int, err error) {
	return 0, nil
}

func (conn *stubConn) Write(b []byte) (n int, err error) {
	return 0, nil
}

func (conn *stubConn) Close() error {
	return nil
}

func (conn *stubConn) LocalAddr() net.Addr {
	return nil
}

func (conn *stubConn) RemoteAddr() net.Addr {
	if !conn.Fake {
		return nil
	}
	return &net.IPAddr{
		IP: net.IP{},
	}
}

func (conn *stubConn) SetDeadline(t time.Time) error {
	return nil
}

func (conn *stubConn) SetReadDeadline(t time.Time) error {
	return nil
}

func (conn *stubConn) SetWriteDeadline(t time.Time) error {
	return nil
}

func TestStubConn(t *testing.T) {
	var conn = new(stubConn)
	_, err := NetConnLookup(conn)
	if err == nil {
		t.Fatal("Address should fail to return an address")
	}

	conn.Fake = true
	_, err = NetConnLookup(conn)
	if err == nil {
		t.Fatal("Address should fail to return an address")
	}

}

func TestValidIP(t *testing.T) {
	ip4 := []byte{127, 0, 0, 1}
	ip6 := make([]byte, 16)
	ip6[15] = 1

	if !validIP(ip4) || !validIP(ip6) {
		t.Fatal("Failed to validate an IPv4 or an IPv6 address")
	}
}
