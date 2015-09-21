package pflag

import (
	"fmt"
	"net"
)

// -- net.IP value
type ipValue net.IP

func newIPValue(val net.IP, p *net.IP) *ipValue {
	*p = val
	return (*ipValue)(p)
}

func (i *ipValue) String() string { return net.IP(*i).String() }
func (i *ipValue) Set(s string) error {
	ip := net.ParseIP(s)
	if ip == nil {
		return fmt.Errorf("failed to parse IP: %q", s)
	}
	*i = ipValue(ip)
	return nil
}
func (i *ipValue) Get() interface{} {
	return net.IP(*i)
}

func (i *ipValue) Type() string {
	return "ip"
}

// IPVar defines an net.IP flag with specified name, default value, and usage string.
// The argument p points to an net.IP variable in which to store the value of the flag.
func (f *FlagSet) IPVar(p *net.IP, name string, value net.IP, usage string) {
	f.VarP(newIPValue(value, p), name, "", usage)
}

// Like IPVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IPVarP(p *net.IP, name, shorthand string, value net.IP, usage string) {
	f.VarP(newIPValue(value, p), name, shorthand, usage)
}

// IPVar defines an net.IP flag with specified name, default value, and usage string.
// The argument p points to an net.IP variable in which to store the value of the flag.
func IPVar(p *net.IP, name string, value net.IP, usage string) {
	CommandLine.VarP(newIPValue(value, p), name, "", usage)
}

// Like IPVar, but accepts a shorthand letter that can be used after a single dash.
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

// Like IP, but accepts a shorthand letter that can be used after a single dash.
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

// Like IP, but accepts a shorthand letter that can be used after a single dash.
func IPP(name, shorthand string, value net.IP, usage string) *net.IP {
	return CommandLine.IPP(name, shorthand, value, usage)
}
