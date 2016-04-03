package opts

import (
	"fmt"
	"net"
)

// IpOpt type that hold an IP
type IpOpt struct {
	*net.IP
}

func NewIpOpt(ref *net.IP, defaultVal string) *IpOpt {
	o := &IpOpt{
		IP: ref,
	}
	o.Set(defaultVal)
	return o
}

func (o *IpOpt) Set(val string) error {
	ip := net.ParseIP(val)
	if ip == nil {
		return fmt.Errorf("%s is not an ip address", val)
	}
	*o.IP = ip
	return nil
}

func (o *IpOpt) String() string {
	if *o.IP == nil {
		return ""
	}
	return o.IP.String()
}
