// Package whitelist implements IP whitelisting for various types
// of connections. Two types of access control lists (ACLs) are
// supported: host-based and network-based.
package whitelist

import (
	"errors"
	"log"
	"net"
	"sort"
	"strings"
	"sync"
)

// An ACL stores a list of permitted IP addresses, and handles
// concurrency as needed.
type ACL interface {
	// Permitted takes an IP address, and returns true if the
	// IP address is whitelisted (e.g. permitted access).
	Permitted(net.IP) bool
}

// A HostACL stores a list of permitted hosts.
type HostACL interface {
	ACL

	// Add takes an IP address and adds it to the whitelist so
	// that it is now permitted.
	Add(net.IP)

	// Remove takes an IP address and drops it from the whitelist
	// so that it is no longer permitted.
	Remove(net.IP)
}

// validIP takes an IP address (which is implemented as a byte slice)
// and ensures that it is a possible address. Right now, this means
// just doing length checks.
func validIP(ip net.IP) bool {
	if len(ip) == 4 {
		return true
	}

	if len(ip) == 16 {
		return true
	}

	return false
}

// Basic implements a basic map-backed whitelister that uses an
// RWMutex for conccurency. IPv4 addresses are treated differently
// than an IPv6 address; namely, the IPv4 localhost will not match
// the IPv6 localhost.
type Basic struct {
	lock      *sync.Mutex
	whitelist map[string]bool
}

// Permitted returns true if the IP has been whitelisted.
func (wl *Basic) Permitted(ip net.IP) bool {
	if !validIP(ip) {
		return false
	}

	wl.lock.Lock()
	permitted := wl.whitelist[ip.String()]
	wl.lock.Unlock()
	return permitted
}

// Add whitelists an IP.
func (wl *Basic) Add(ip net.IP) {
	if !validIP(ip) {
		return
	}

	wl.lock.Lock()
	defer wl.lock.Unlock()
	wl.whitelist[ip.String()] = true
}

// Remove clears the IP from the whitelist.
func (wl *Basic) Remove(ip net.IP) {
	if !validIP(ip) {
		return
	}

	wl.lock.Lock()
	defer wl.lock.Unlock()
	delete(wl.whitelist, ip.String())
}

// NewBasic returns a new initialised basic whitelist.
func NewBasic() *Basic {
	return &Basic{
		lock:      new(sync.Mutex),
		whitelist: map[string]bool{},
	}
}

// MarshalJSON serialises a host whitelist to a comma-separated list of
// hosts, implementing the json.Marshaler interface.
func (wl *Basic) MarshalJSON() ([]byte, error) {
	wl.lock.Lock()
	defer wl.lock.Unlock()
	var ss = make([]string, 0, len(wl.whitelist))
	for ip := range wl.whitelist {
		ss = append(ss, ip)
	}

	out := []byte(`"` + strings.Join(ss, ",") + `"`)
	return out, nil
}

// UnmarshalJSON implements the json.Unmarshaler interface for host
// whitelists, taking a comma-separated string of hosts.
func (wl *Basic) UnmarshalJSON(in []byte) error {
	if in[0] != '"' || in[len(in)-1] != '"' {
		return errors.New("whitelist: invalid whitelist")
	}

	if wl.lock == nil {
		wl.lock = new(sync.Mutex)
	}

	wl.lock.Lock()
	defer wl.lock.Unlock()

	netString := strings.TrimSpace(string(in[1 : len(in)-1]))
	nets := strings.Split(netString, ",")

	wl.whitelist = map[string]bool{}
	for i := range nets {
		addr := strings.TrimSpace(nets[i])
		if addr == "" {
			continue
		}

		ip := net.ParseIP(addr)
		if ip == nil {
			wl.whitelist = nil
			return errors.New("whitelist: invalid IP address " + addr)
		}
		wl.whitelist[addr] = true
	}

	return nil
}

// DumpBasic returns a whitelist as a byte slice where each IP is on
// its own line.
func DumpBasic(wl *Basic) []byte {
	wl.lock.Lock()
	defer wl.lock.Unlock()

	var addrs = make([]string, 0, len(wl.whitelist))
	for ip := range wl.whitelist {
		addrs = append(addrs, ip)
	}

	sort.Strings(addrs)

	addrList := strings.Join(addrs, "\n")
	return []byte(addrList)
}

// LoadBasic loads a whitelist from a byteslice.
func LoadBasic(in []byte) (*Basic, error) {
	wl := NewBasic()
	addrs := strings.Split(string(in), "\n")

	for _, addr := range addrs {
		ip := net.ParseIP(addr)
		if ip == nil {
			return nil, errors.New("whitelist: invalid address")
		}
		wl.Add(ip)
	}
	return wl, nil
}

// HostStub allows host whitelisting to be added into a system's flow
// without doing anything yet. All operations result in warning log
// messages being printed to stderr. There is no mechanism for
// squelching these messages short of modifying the log package's
// default logger.
type HostStub struct{}

// Permitted always returns true, but prints a warning message alerting
// that whitelisting is stubbed.
func (wl HostStub) Permitted(ip net.IP) bool {
	log.Printf("WARNING: whitelist check for %s but whitelisting is stubbed", ip)
	return true
}

// Add prints a warning message about whitelisting being stubbed.
func (wl HostStub) Add(ip net.IP) {
	log.Printf("WARNING: IP %s added to whitelist but whitelisting is stubbed", ip)
}

// Remove prints a warning message about whitelisting being stubbed.
func (wl HostStub) Remove(ip net.IP) {
	log.Printf("WARNING: IP %s removed from whitelist but whitelisting is stubbed", ip)
}

// NewHostStub returns a new stubbed host whitelister.
func NewHostStub() HostStub {
	log.Println("WARNING: whitelisting is being stubbed")
	return HostStub{}
}
