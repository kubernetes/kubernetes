package dhcp4

import (
	"bytes"
	"net"
	"reflect"
	"sort"
	"testing"
	"time"
)

// Verify that all options are returned by Options.SelectOrderOrAll if
// the input order value is nil.
func TestSelectOrderOrAllNil(t *testing.T) {
	assertOptionsSlices(t, 0, "nil order", allOptionsSlice, optMap.SelectOrderOrAll(nil))
}

// Verify that all options are returned by Options.SelectOrderOrAll if
// the input order value is not nil over several tests.
func TestSelectOrderOrAllNotNil(t *testing.T) {
	for i, tt := range selectOrderTests {
		assertOptionsSlices(t, i, tt.description, tt.result, optMap.SelectOrderOrAll(tt.order))
	}
}

// Verify that no options are returned by Options.SelectOrder if
// the input order value is nil.
func TestSelectOrderNil(t *testing.T) {
	assertOptionsSlices(t, 0, "nil order", nil, optMap.SelectOrder(nil))
}

// Verify that all options are returned by Options.SelectOrder if
// the input order value is not nil over several tests.
func TestSelectOrderNotNil(t *testing.T) {
	for i, tt := range selectOrderTests {
		assertOptionsSlices(t, i, tt.description, tt.result, optMap.SelectOrder(tt.order))
	}
}

func TestIPRange(t *testing.T) {
	var tests = []struct {
		start  net.IP
		stop   net.IP
		result int
	}{
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 1, 1),
			result: 1,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 1, 254),
			result: 254,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 10, 1),
			result: 2305,
		},
		{
			start:  net.IPv4(172, 16, 1, 1),
			stop:   net.IPv4(192, 168, 1, 1),
			result: 345505793,
		},
	}

	for _, tt := range tests {
		if result := IPRange(tt.start, tt.stop); result != tt.result {
			t.Fatalf("IPRange(%s, %s), unexpected result: %v != %v",
				tt.start, tt.stop, result, tt.result)
		}
	}
}

func TestIPAdd(t *testing.T) {
	var tests = []struct {
		start  net.IP
		add    int
		result net.IP
	}{
		{
			start:  net.IPv4(192, 168, 1, 1),
			add:    0,
			result: net.IPv4(192, 168, 1, 1),
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			add:    253,
			result: net.IPv4(192, 168, 1, 254),
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			add:    1024,
			result: net.IPv4(192, 168, 5, 1),
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			add:    4096,
			result: net.IPv4(192, 168, 17, 1),
		},
	}

	for _, tt := range tests {
		if result := IPAdd(tt.start, tt.add); !result.Equal(tt.result) {
			t.Fatalf("IPAdd(%s, %d), unexpected result: %v != %v",
				tt.start, tt.add, result, tt.result)
		}
	}
}

func TestIPLess(t *testing.T) {
	var tests = []struct {
		a      net.IP
		b      net.IP
		result bool
	}{
		{
			a:      net.IPv4(192, 168, 1, 1),
			b:      net.IPv4(192, 168, 1, 1),
			result: false,
		},
		{
			a:      net.IPv4(192, 168, 1, 1),
			b:      net.IPv4(192, 168, 0, 1),
			result: false,
		},
		{
			a:      net.IPv4(192, 168, 0, 1),
			b:      net.IPv4(192, 168, 1, 1),
			result: true,
		},
		{
			a:      net.IPv4(192, 168, 0, 1),
			b:      net.IPv4(192, 168, 10, 1),
			result: true,
		},
	}

	for _, tt := range tests {
		if result := IPLess(tt.a, tt.b); result != tt.result {
			t.Fatalf("IPLess(%s, %s), unexpected result: %v != %v",
				tt.a, tt.b, result, tt.result)
		}
	}
}

func TestIPInRange(t *testing.T) {
	var tests = []struct {
		start  net.IP
		stop   net.IP
		ip     net.IP
		result bool
	}{
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 2, 1),
			ip:     net.IPv4(192, 168, 3, 1),
			result: false,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 10, 1),
			ip:     net.IPv4(192, 168, 0, 1),
			result: false,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 10, 1),
			ip:     net.IPv4(192, 168, 5, 1),
			result: true,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 3, 1),
			ip:     net.IPv4(192, 168, 3, 0),
			result: true,
		},
		{
			start:  net.IPv4(192, 168, 1, 1),
			stop:   net.IPv4(192, 168, 1, 1),
			ip:     net.IPv4(192, 168, 1, 1),
			result: true,
		},
	}

	for _, tt := range tests {
		if result := IPInRange(tt.start, tt.stop, tt.ip); result != tt.result {
			t.Fatalf("IPInRange(%s, %s, %s), unexpected result: %v != %v",
				tt.start, tt.stop, tt.ip, result, tt.result)
		}
	}
}

func TestOptionsLeaseTime(t *testing.T) {
	var tests = []struct {
		duration time.Duration
		result   []byte
	}{
		{
			duration: 0 * time.Second,
			result:   []byte{0, 0, 0, 0},
		},
		{
			duration: 2 * time.Second,
			result:   []byte{0, 0, 0, 2},
		},
		{
			duration: 60 * time.Second,
			result:   []byte{0, 0, 0, 60},
		},
		{
			duration: 6 * time.Hour,
			result:   []byte{0, 0, 84, 96},
		},
		{
			duration: 24 * time.Hour,
			result:   []byte{0, 1, 81, 128},
		},
		{
			duration: 365 * 24 * time.Hour,
			result:   []byte{1, 225, 51, 128},
		},
	}

	for _, tt := range tests {
		if result := OptionsLeaseTime(tt.duration); !bytes.Equal(result, tt.result) {
			t.Fatalf("OptionsLeaseTime(%s), unexpected result: %v != %v",
				tt.duration, result, tt.result)
		}
	}
}

func TestJoinIPs(t *testing.T) {
	var tests = []struct {
		ips    []net.IP
		result []byte
	}{
		{
			ips:    []net.IP{net.IPv4(10, 0, 0, 1)},
			result: []byte{10, 0, 0, 1},
		},
		{
			ips:    []net.IP{net.IPv4(192, 168, 1, 1), net.IPv4(192, 168, 2, 1)},
			result: []byte{192, 168, 1, 1, 192, 168, 2, 1},
		},
		{
			ips:    []net.IP{net.IPv4(10, 0, 0, 1), net.IPv4(255, 255, 255, 254)},
			result: []byte{10, 0, 0, 1, 255, 255, 255, 254},
		},
		{
			ips:    []net.IP{net.IPv4(8, 8, 8, 8), net.IPv4(8, 8, 4, 4), net.IPv4(192, 168, 1, 1)},
			result: []byte{8, 8, 8, 8, 8, 8, 4, 4, 192, 168, 1, 1},
		},
	}

	for _, tt := range tests {
		if result := JoinIPs(tt.ips); !bytes.Equal(result, tt.result) {
			t.Fatalf("JoinIPs(%s), unexpected result: %v != %v",
				tt.ips, result, tt.result)
		}
	}
}

// byOptionCode implements sort.Interface for []Option.
type byOptionCode []Option

func (b byOptionCode) Len() int               { return len(b) }
func (b byOptionCode) Less(i int, j int) bool { return b[i].Code < b[j].Code }
func (b byOptionCode) Swap(i int, j int)      { b[i], b[j] = b[j], b[i] }

// assertOptionsSlices is a test helper which verifies that two options slices
// are identical.  Several parameters are passed for easy identification of
// failing tests.
func assertOptionsSlices(t *testing.T, i int, description string, want []Option, got []Option) {
	// Verify slices are same length
	if want, got := len(want), len(got); want != got {
		t.Fatalf("%02d: test %q, mismatched length: %d != %d",
			i, description, want, got)
	}

	// Sort slices
	sort.Sort(byOptionCode(want))
	sort.Sort(byOptionCode(got))

	// Verify slices are identical
	if len(want) > 0 && len(got) > 0 && !reflect.DeepEqual(want, got) {
		t.Fatalf("%02d: test %q, unexpected options: %v != %v",
			i, description, want, got)
	}
}

// optMap is an Options map which contains a number of option
// codes and values, used for testing.
var optMap = Options{
	OptionSubnetMask:       []byte{255, 255, 255, 0},
	OptionRouter:           []byte{192, 168, 1, 1},
	OptionDomainNameServer: []byte{192, 168, 1, 2},
	OptionTimeServer:       []byte{192, 168, 1, 3},
	OptionLogServer:        []byte{192, 168, 1, 4},
}

// allOptionsSlice is a []Option derived from optMap.  It is used
// for some tests.
var allOptionsSlice = []Option{
	Option{
		Code:  OptionSubnetMask,
		Value: optMap[OptionSubnetMask],
	},
	Option{
		Code:  OptionRouter,
		Value: optMap[OptionRouter],
	},
	Option{
		Code:  OptionDomainNameServer,
		Value: optMap[OptionDomainNameServer],
	},
	Option{
		Code:  OptionTimeServer,
		Value: optMap[OptionTimeServer],
	},
	Option{
		Code:  OptionLogServer,
		Value: optMap[OptionLogServer],
	},
}

// selectOrderTests is a set of tests used for Options.SelectOrder
// and Options.SelectOrderOrAll methods.
var selectOrderTests = []struct {
	description string
	order       []byte
	result      []Option
}{
	{
		description: "subnet mask only",
		order: []byte{
			byte(OptionSubnetMask),
		},
		result: []Option{
			Option{
				Code:  OptionSubnetMask,
				Value: optMap[OptionSubnetMask],
			},
		},
	},
	{
		description: "subnet mask and time server",
		order: []byte{
			byte(OptionSubnetMask),
			byte(OptionTimeServer),
		},
		result: []Option{
			Option{
				Code:  OptionSubnetMask,
				Value: optMap[OptionSubnetMask],
			},
			Option{
				Code:  OptionTimeServer,
				Value: optMap[OptionTimeServer],
			},
		},
	},
	{
		description: "domain name server, time server, router",
		order: []byte{
			byte(OptionDomainNameServer),
			byte(OptionTimeServer),
			byte(OptionRouter),
		},
		result: []Option{
			Option{
				Code:  OptionDomainNameServer,
				Value: optMap[OptionDomainNameServer],
			},
			Option{
				Code:  OptionTimeServer,
				Value: optMap[OptionTimeServer],
			},
			Option{
				Code:  OptionRouter,
				Value: optMap[OptionRouter],
			},
		},
	},
	{
		description: "all options in order",
		order: []byte{
			byte(OptionSubnetMask),
			byte(OptionRouter),
			byte(OptionDomainNameServer),
			byte(OptionTimeServer),
			byte(OptionLogServer),
		},
		result: allOptionsSlice,
	},
}
