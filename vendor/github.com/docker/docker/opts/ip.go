package opts

import (
	"fmt"
	"net"
)

// IPOpt holds an IP. It is used to store values from CLI flags.
type IPOpt struct {
	*net.IP
}

// NewIPOpt creates a new IPOpt from a reference net.IP and a
// string representation of an IP. If the string is not a valid
// IP it will fallback to the specified reference.
func NewIPOpt(ref *net.IP, defaultVal string) *IPOpt {
	o := &IPOpt{
		IP: ref,
	}
	o.Set(defaultVal)
	return o
}

// Set sets an IPv4 or IPv6 address from a given string. If the given
// string is not parsable as an IP address it returns an error.
func (o *IPOpt) Set(val string) error {
	ip := net.ParseIP(val)
	if ip == nil {
		return fmt.Errorf("%s is not an ip address", val)
	}
	*o.IP = ip
	return nil
}

// String returns the IP address stored in the IPOpt. If stored IP is a
// nil pointer, it returns an empty string.
func (o *IPOpt) String() string {
	if *o.IP == nil {
		return ""
	}
	return o.IP.String()
}

// Type returns the type of the option
func (o *IPOpt) Type() string {
	return "ip"
}
