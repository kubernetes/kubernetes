package pflag

import (
	"fmt"
	"net"
	"strings"
)

// -- net.IP value
type ipValue net.IP

func newIPValue(val net.IP, p *net.IP) *ipValue {
	*p = val
	return (*ipValue)(p)
}

func (i *ipValue) String() string { return net.IP(*i).String() }
func (i *ipValue) Set(s string) error {
	if s == "" {
		return nil
	}
	ip := net.ParseIP(strings.TrimSpace(s))
	if ip == nil {
		return fmt.Errorf("failed to parse IP: %q", s)
	}
	*i = ipValue(ip)
	return nil
}

func (i *ipValue) Type() string {
	return "ip"
}

func ipConv(sval string) (interface{}, error) {
	ip := net.ParseIP(sval)
	if ip != nil {
		return ip, nil
	}
	return nil, fmt.Errorf("invalid string being converted to IP address: %s", sval)
}

// GetIP return the net.IP value of a flag with the given name
func (f *FlagSet) GetIP(name string) (net.IP, error) {
	val, err := f.getFlagType(name, "ip", ipConv)
	if err != nil {
		return nil, err
	}
	return val.(net.IP), nil
}

// IPVar defines an net.IP flag with specified name, default value, and usage string.
// The argument p points to an net.IP variable in which to store the value of the flag.
func (f *FlagSet) IPVar(p *net.IP, name string, value net.IP, usage string) {
	f.VarP(newIPValue(value, p), name, "", usage)
}

// IPVarP is like IPVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IPVarP(p *net.IP, name, shorthand string, value net.IP, usage string) {
	f.VarP(newIPValue(value, p), name, shorthand, usage)
}

// IPVar defines an net.IP flag with specified name, default value, and usage string.
// The argument p points to an net.IP variable in which to store the value of the flag.
func IPVar(p *net.IP, name string, value net.IP, usage string) {
	CommandLine.VarP(newIPValue(value, p), name, "", usage)
}

// IPVarP is like IPVar, but accepts a shorthand letter that can be used after a single dash.
func IPVarP(p *net.IP, name, shorthand string, value net.IP, usage string) {
	CommandLine.VarP(newIPValue(value, p), name, shorthand, usage)
}

// IP defines an net.IP flag with specified name, default value, and usage string.
// The return value is the address of an net.IP variable that stores the value of the flag.
func (f *FlagSet) IP(name string, value net.IP, usage string) *net.IP {
	p := new(net.IP)
	f.IPVarP(p, name, "", value, usage)
	return p
}

// IPP is like IP, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IPP(name, shorthand string, value net.IP, usage string) *net.IP {
	p := new(net.IP)
	f.IPVarP(p, name, shorthand, value, usage)
	return p
}

// IP defines an net.IP flag with specified name, default value, and usage string.
// The return value is the address of an net.IP variable that stores the value of the flag.
func IP(name string, value net.IP, usage string) *net.IP {
	return CommandLine.IPP(name, "", value, usage)
}

// IPP is like IP, but accepts a shorthand letter that can be used after a single dash.
func IPP(name, shorthand string, value net.IP, usage string) *net.IP {
	return CommandLine.IPP(name, shorthand, value, usage)
}
