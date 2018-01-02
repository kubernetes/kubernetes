package driverapi

import (
	"encoding/json"
	"fmt"
	"net"

	"github.com/docker/libnetwork/types"
)

// MarshalJSON encodes IPAMData into json message
func (i *IPAMData) MarshalJSON() ([]byte, error) {
	m := map[string]interface{}{}
	m["AddressSpace"] = i.AddressSpace
	if i.Pool != nil {
		m["Pool"] = i.Pool.String()
	}
	if i.Gateway != nil {
		m["Gateway"] = i.Gateway.String()
	}
	if i.AuxAddresses != nil {
		am := make(map[string]string, len(i.AuxAddresses))
		for k, v := range i.AuxAddresses {
			am[k] = v.String()
		}
		m["AuxAddresses"] = am
	}
	return json.Marshal(m)
}

// UnmarshalJSON decodes a json message into IPAMData
func (i *IPAMData) UnmarshalJSON(data []byte) error {
	var (
		m   map[string]interface{}
		err error
	)
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	i.AddressSpace = m["AddressSpace"].(string)
	if v, ok := m["Pool"]; ok {
		if i.Pool, err = types.ParseCIDR(v.(string)); err != nil {
			return err
		}
	}
	if v, ok := m["Gateway"]; ok {
		if i.Gateway, err = types.ParseCIDR(v.(string)); err != nil {
			return err
		}
	}
	if v, ok := m["AuxAddresses"]; ok {
		b, _ := json.Marshal(v)
		var am map[string]string
		if err = json.Unmarshal(b, &am); err != nil {
			return err
		}
		i.AuxAddresses = make(map[string]*net.IPNet, len(am))
		for k, v := range am {
			if i.AuxAddresses[k], err = types.ParseCIDR(v); err != nil {
				return err
			}
		}
	}
	return nil
}

// Validate checks whether the IPAMData structure contains congruent data
func (i *IPAMData) Validate() error {
	var isV6 bool
	if i.Pool == nil {
		return types.BadRequestErrorf("invalid pool")
	}
	if i.Gateway == nil {
		return types.BadRequestErrorf("invalid gateway address")
	}
	isV6 = i.IsV6()
	if isV6 && i.Gateway.IP.To4() != nil || !isV6 && i.Gateway.IP.To4() == nil {
		return types.BadRequestErrorf("incongruent ip versions for pool and gateway")
	}
	for k, sip := range i.AuxAddresses {
		if isV6 && sip.IP.To4() != nil || !isV6 && sip.IP.To4() == nil {
			return types.BadRequestErrorf("incongruent ip versions for pool and secondary ip address %s", k)
		}
	}
	if !i.Pool.Contains(i.Gateway.IP) {
		return types.BadRequestErrorf("invalid gateway address (%s) does not belong to the pool (%s)", i.Gateway, i.Pool)
	}
	for k, sip := range i.AuxAddresses {
		if !i.Pool.Contains(sip.IP) {
			return types.BadRequestErrorf("invalid secondary address %s (%s) does not belong to the pool (%s)", k, i.Gateway, i.Pool)
		}
	}
	return nil
}

// IsV6 returns whether this is an IPv6 IPAMData structure
func (i *IPAMData) IsV6() bool {
	return nil == i.Pool.IP.To4()
}

func (i *IPAMData) String() string {
	return fmt.Sprintf("AddressSpace: %s\nPool: %v\nGateway: %v\nAddresses: %v", i.AddressSpace, i.Pool, i.Gateway, i.AuxAddresses)
}
