package bridge

import "testing"

func TestIPConstantValues(t *testing.T) {
	if ipv4|ipv6 != ipvboth {
		t.Fatalf("bitwise or of ipv4(%04b) and ipv6(%04b) must yield ipvboth(%04b)", ipv4, ipv6, ipvboth)
	}
	if ipvboth&(^(ipv4 | ipv6)) != ipvnone {
		t.Fatalf("ipvboth(%04b) with unset ipv4(%04b) and ipv6(%04b) bits shall equal to ipvnone", ipvboth, ipv4, ipv6)
	}
}
