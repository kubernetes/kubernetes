package dhcp4

import (
	"encoding/binary"
	"net"
	"time"
)

// IPRange returns how many ips in the ip range from start to stop (inclusive)
func IPRange(start, stop net.IP) int {
	//return int(Uint([]byte(stop))-Uint([]byte(start))) + 1
	return int(binary.BigEndian.Uint32(stop.To4())) - int(binary.BigEndian.Uint32(start.To4())) + 1
}

// IPAdd returns a copy of start + add.
// IPAdd(net.IP{192,168,1,1},30) returns net.IP{192.168.1.31}
func IPAdd(start net.IP, add int) net.IP { // IPv4 only
	start = start.To4()
	//v := Uvarint([]byte(start))
	result := make(net.IP, 4)
	binary.BigEndian.PutUint32(result, binary.BigEndian.Uint32(start)+uint32(add))
	//PutUint([]byte(result), v+uint64(add))
	return result
}

// IPLess returns where IP a is less than IP b.
func IPLess(a, b net.IP) bool {
	b = b.To4()
	for i, ai := range a.To4() {
		if ai != b[i] {
			return ai < b[i]
		}
	}
	return false
}

// IPInRange returns true if ip is between (inclusive) start and stop.
func IPInRange(start, stop, ip net.IP) bool {
	return !(IPLess(ip, start) || IPLess(stop, ip))
}

// OptionsLeaseTime - converts a time.Duration to a 4 byte slice, compatible
// with OptionIPAddressLeaseTime.
func OptionsLeaseTime(d time.Duration) []byte {
	leaseBytes := make([]byte, 4)
	binary.BigEndian.PutUint32(leaseBytes, uint32(d/time.Second))
	//PutUvarint(leaseBytes, uint64(d/time.Second))
	return leaseBytes
}

// JoinIPs returns a byte slice of IP addresses, one immediately after the other
// This may be useful for creating multiple IP options such as OptionRouter.
func JoinIPs(ips []net.IP) (b []byte) {
	for _, v := range ips {
		b = append(b, v.To4()...)
	}
	return
}
