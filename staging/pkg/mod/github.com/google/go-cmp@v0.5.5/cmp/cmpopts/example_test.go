// Copyright 2020, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmpopts_test

import (
	"fmt"
	"net"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/go-cmp/cmp/internal/flags"
)

func init() {
	flags.Deterministic = true
}

// Use IgnoreFields to ignore fields on a struct type when comparing
// by providing a value of the type and the field names to ignore.
// Typically, a zero value of the type is used (e.g., foo.MyStruct{}).
func ExampleIgnoreFields_testing() {
	// Let got be the hypothetical value obtained from some logic under test
	// and want be the expected golden data.
	got, want := MakeGatewayInfo()

	// While the specified fields will be semantically ignored for the comparison,
	// the fields may be printed in the diff when displaying entire values
	// that are already determined to be different.
	if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(Client{}, "IPAddress")); diff != "" {
		t.Errorf("MakeGatewayInfo() mismatch (-want +got):\n%s", diff)
	}

	// Output:
	// MakeGatewayInfo() mismatch (-want +got):
	//   cmpopts_test.Gateway{
	//   	SSID:      "CoffeeShopWiFi",
	// - 	IPAddress: s"192.168.0.2",
	// + 	IPAddress: s"192.168.0.1",
	//   	NetMask:   {0xff, 0xff, 0x00, 0x00},
	//   	Clients: []cmpopts_test.Client{
	//   		... // 3 identical elements
	//   		{Hostname: "espresso", ...},
	//   		{Hostname: "latte", LastSeen: s"2009-11-10 23:00:23 +0000 UTC", ...},
	// + 		{
	// + 			Hostname:  "americano",
	// + 			IPAddress: s"192.168.0.188",
	// + 			LastSeen:  s"2009-11-10 23:03:05 +0000 UTC",
	// + 		},
	//   	},
	//   }
}

type (
	Gateway struct {
		SSID      string
		IPAddress net.IP
		NetMask   net.IPMask
		Clients   []Client
	}
	Client struct {
		Hostname  string
		IPAddress net.IP
		LastSeen  time.Time
	}
)

func MakeGatewayInfo() (x, y Gateway) {
	x = Gateway{
		SSID:      "CoffeeShopWiFi",
		IPAddress: net.IPv4(192, 168, 0, 1),
		NetMask:   net.IPv4Mask(255, 255, 0, 0),
		Clients: []Client{{
			Hostname:  "ristretto",
			IPAddress: net.IPv4(192, 168, 0, 116),
		}, {
			Hostname:  "aribica",
			IPAddress: net.IPv4(192, 168, 0, 104),
			LastSeen:  time.Date(2009, time.November, 10, 23, 6, 32, 0, time.UTC),
		}, {
			Hostname:  "macchiato",
			IPAddress: net.IPv4(192, 168, 0, 153),
			LastSeen:  time.Date(2009, time.November, 10, 23, 39, 43, 0, time.UTC),
		}, {
			Hostname:  "espresso",
			IPAddress: net.IPv4(192, 168, 0, 121),
		}, {
			Hostname:  "latte",
			IPAddress: net.IPv4(192, 168, 0, 219),
			LastSeen:  time.Date(2009, time.November, 10, 23, 0, 23, 0, time.UTC),
		}, {
			Hostname:  "americano",
			IPAddress: net.IPv4(192, 168, 0, 188),
			LastSeen:  time.Date(2009, time.November, 10, 23, 3, 5, 0, time.UTC),
		}},
	}
	y = Gateway{
		SSID:      "CoffeeShopWiFi",
		IPAddress: net.IPv4(192, 168, 0, 2),
		NetMask:   net.IPv4Mask(255, 255, 0, 0),
		Clients: []Client{{
			Hostname:  "ristretto",
			IPAddress: net.IPv4(192, 168, 0, 116),
		}, {
			Hostname:  "aribica",
			IPAddress: net.IPv4(192, 168, 0, 104),
			LastSeen:  time.Date(2009, time.November, 10, 23, 6, 32, 0, time.UTC),
		}, {
			Hostname:  "macchiato",
			IPAddress: net.IPv4(192, 168, 0, 153),
			LastSeen:  time.Date(2009, time.November, 10, 23, 39, 43, 0, time.UTC),
		}, {
			Hostname:  "espresso",
			IPAddress: net.IPv4(192, 168, 0, 121),
		}, {
			Hostname:  "latte",
			IPAddress: net.IPv4(192, 168, 0, 221),
			LastSeen:  time.Date(2009, time.November, 10, 23, 0, 23, 0, time.UTC),
		}},
	}
	return x, y
}

var t fakeT

type fakeT struct{}

func (t fakeT) Errorf(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) }
