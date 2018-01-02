package flagutil

import (
	"errors"
	"fmt"
	"net"
	"strings"
)

// IPv4Flag parses a string into a net.IP after asserting that it
// is an IPv4 address. This type implements the flag.Value interface.
type IPv4Flag struct {
	val net.IP
}

func (f *IPv4Flag) IP() net.IP {
	return f.val
}

func (f *IPv4Flag) Set(v string) error {
	ip := net.ParseIP(v)
	if ip == nil || ip.To4() == nil {
		return errors.New("not an IPv4 address")
	}
	f.val = ip
	return nil
}

func (f *IPv4Flag) String() string {
	return f.val.String()
}

// StringSliceFlag parses a comma-delimited list of strings into
// a []string. This type implements the flag.Value interface.
type StringSliceFlag []string

func (ss *StringSliceFlag) String() string {
	return fmt.Sprintf("%+v", *ss)
}

func (ss *StringSliceFlag) Set(v string) error {
	*ss = strings.Split(v, ",")
	return nil
}
