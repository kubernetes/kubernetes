package pflag

import (
	"fmt"
	"net"
	"strings"
)

// IPNet adapts net.IPNet for use as a flag.
type IPNetValue net.IPNet

func (ipnet IPNetValue) String() string {
	n := net.IPNet(ipnet)
	return n.String()
}

func (ipnet *IPNetValue) Set(value string) error {
	_, n, err := net.ParseCIDR(strings.TrimSpace(value))
	if err != nil {
		return err
	}
	*ipnet = IPNetValue(*n)
	return nil
}

func (*IPNetValue) Type() string {
	return "ipNet"
}

var _ = strings.TrimSpace

func newIPNetValue(val net.IPNet, p *net.IPNet) *IPNetValue {
	*p = val
	return (*IPNetValue)(p)
}

func ipNetConv(sval string) (interface{}, error) {
	_, n, err := net.ParseCIDR(strings.TrimSpace(sval))
	if err == nil {
		return *n, nil
	}
	return nil, fmt.Errorf("invalid string being converted to IPNet: %s", sval)
}

// GetIPNet return the net.IPNet value of a flag with the given name
func (f *FlagSet) GetIPNet(name string) (net.IPNet, error) {
	val, err := f.getFlagType(name, "ipNet", ipNetConv)
	if err != nil {
		return net.IPNet{}, err
	}
	return val.(net.IPNet), nil
}

// IPNetVar defines an net.IPNet flag with specified name, default value, and usage string.
// The argument p points to an net.IPNet variable in which to store the value of the flag.
func (f *FlagSet) IPNetVar(p *net.IPNet, name string, value net.IPNet, usage string) {
	f.VarP(newIPNetValue(value, p), name, "", usage)
}

// Like IPNetVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IPNetVarP(p *net.IPNet, name, shorthand string, value net.IPNet, usage string) {
	f.VarP(newIPNetValue(value, p), name, shorthand, usage)
}

// IPNetVar defines an net.IPNet flag with specified name, default value, and usage string.
// The argument p points to an net.IPNet variable in which to store the value of the flag.
func IPNetVar(p *net.IPNet, name string, value net.IPNet, usage string) {
	CommandLine.VarP(newIPNetValue(value, p), name, "", usage)
}

// Like IPNetVar, but accepts a shorthand letter that can be used after a single dash.
func IPNetVarP(p *net.IPNet, name, shorthand string, value net.IPNet, usage string) {
	CommandLine.VarP(newIPNetValue(value, p), name, shorthand, usage)
}

// IPNet defines an net.IPNet flag with specified name, default value, and usage string.
// The return value is the address of an net.IPNet variable that stores the value of the flag.
func (f *FlagSet) IPNet(name string, value net.IPNet, usage string) *net.IPNet {
	p := new(net.IPNet)
	f.IPNetVarP(p, name, "", value, usage)
	return p
}

// Like IPNet, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IPNetP(name, shorthand string, value net.IPNet, usage string) *net.IPNet {
	p := new(net.IPNet)
	f.IPNetVarP(p, name, shorthand, value, usage)
	return p
}

// IPNet defines an net.IPNet flag with specified name, default value, and usage string.
// The return value is the address of an net.IPNet variable that stores the value of the flag.
func IPNet(name string, value net.IPNet, usage string) *net.IPNet {
	return CommandLine.IPNetP(name, "", value, usage)
}

// Like IPNet, but accepts a shorthand letter that can be used after a single dash.
func IPNetP(name, shorthand string, value net.IPNet, usage string) *net.IPNet {
	return CommandLine.IPNetP(name, shorthand, value, usage)
}
