package gnl2go

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

func IPv4ToUint32(ipv4 string) (uint32, error) {
	octets := strings.Split(ipv4, ".")
	if len(octets) != 4 {
		return 0, errors.New("cant convert ipv4 to string")

	}
	ipv4int := uint32(0)
	for cntr := 0; cntr < 4; cntr++ {
		tmpVal, err := strconv.Atoi(octets[cntr])
		if err != nil {
			return 0, errors.New("cant convert ipv4 to string")

		}
		ipv4int += uint32(tmpVal << uint((3-cntr)*8))

	}
	return ipv4int, nil

}

func Uint32IPv4ToString(ipv4 uint32) string {
	ipv4addr := ""
	octet := 0
	for cntr := 0; cntr < 4; cntr++ {
		octet = int((ipv4 >> ((3 - uint(cntr)) * 8)) & 255)
		if cntr == 0 {
			ipv4addr = strconv.Itoa(octet)

		} else {
			ipv4addr = strings.Join([]string{ipv4addr, strconv.Itoa(octet)}, ".")

		}

	}
	return ipv4addr

}

type IPv6Addr [4]uint32

func IPv6StringToAddr(ipv6 string) (IPv6Addr, error) {
	var ipv6addr IPv6Addr
	ipv6fields := strings.Split(ipv6, ":")
	//parts in string's represent of V6 address
	if len(ipv6fields) != 8 {
		tmpAddrPart := make([]string, 0)
		for n, val := range ipv6fields {
			if len(val) == 0 {
				tmpAddrPart = append(tmpAddrPart, ipv6fields[n+1:]...)
				ipv6fields = ipv6fields[:n]
				break

			}

		}
		for len(ipv6fields) != (8 - len(tmpAddrPart)) {
			ipv6fields = append(ipv6fields, "0")

		}
		ipv6fields = append(ipv6fields, tmpAddrPart...)

	}
	for n, part := range ipv6fields {
		for len(part) < 4 {
			part = "0" + part

		}
		ipv6fields[n] = part

	}
	//in ipv6 we have 4parts of 32bits each
	for i := 0; i < 4; i++ {
		val, err := strconv.ParseUint(ipv6fields[2*i]+ipv6fields[2*i+1], 16, 32)
		if err != nil {
			return ipv6addr, fmt.Errorf("cant convert string to v6 addr: %v\n", err)

		}
		ipv6addr[i] = uint32(val)

	}
	return ipv6addr, nil

}

func IPv6AddrToString(ipv6addr IPv6Addr) string {
	ipv6 := ""
	for i := 0; i < 4; i++ {
		ipv6 += strconv.FormatUint(uint64(ipv6addr[i]), 16)
		ipv6 += ":"

	}
	return ipv6

}
