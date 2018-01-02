// Copyright 2016 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package current_test

import (
	"encoding/json"
	"io/ioutil"
	"net"
	"os"

	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/types/current"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func testResult() *current.Result {
	ipv4, err := types.ParseCIDR("1.2.3.30/24")
	Expect(err).NotTo(HaveOccurred())
	Expect(ipv4).NotTo(BeNil())

	routegwv4, routev4, err := net.ParseCIDR("15.5.6.8/24")
	Expect(err).NotTo(HaveOccurred())
	Expect(routev4).NotTo(BeNil())
	Expect(routegwv4).NotTo(BeNil())

	ipv6, err := types.ParseCIDR("abcd:1234:ffff::cdde/64")
	Expect(err).NotTo(HaveOccurred())
	Expect(ipv6).NotTo(BeNil())

	routegwv6, routev6, err := net.ParseCIDR("1111:dddd::aaaa/80")
	Expect(err).NotTo(HaveOccurred())
	Expect(routev6).NotTo(BeNil())
	Expect(routegwv6).NotTo(BeNil())

	// Set every field of the struct to ensure source compatibility
	return &current.Result{
		CNIVersion: "0.3.1",
		Interfaces: []*current.Interface{
			{
				Name:    "eth0",
				Mac:     "00:11:22:33:44:55",
				Sandbox: "/proc/3553/ns/net",
			},
		},
		IPs: []*current.IPConfig{
			{
				Version:   "4",
				Interface: current.Int(0),
				Address:   *ipv4,
				Gateway:   net.ParseIP("1.2.3.1"),
			},
			{
				Version:   "6",
				Interface: current.Int(0),
				Address:   *ipv6,
				Gateway:   net.ParseIP("abcd:1234:ffff::1"),
			},
		},
		Routes: []*types.Route{
			{Dst: *routev4, GW: routegwv4},
			{Dst: *routev6, GW: routegwv6},
		},
		DNS: types.DNS{
			Nameservers: []string{"1.2.3.4", "1::cafe"},
			Domain:      "acompany.com",
			Search:      []string{"somedomain.com", "otherdomain.net"},
			Options:     []string{"foo", "bar"},
		},
	}
}

var _ = Describe("Current types operations", func() {
	It("correctly encodes a 0.3.x Result", func() {
		res := testResult()

		// Redirect stdout to capture JSON result
		oldStdout := os.Stdout
		r, w, err := os.Pipe()
		Expect(err).NotTo(HaveOccurred())

		os.Stdout = w
		err = res.Print()
		w.Close()
		Expect(err).NotTo(HaveOccurred())

		// parse the result
		out, err := ioutil.ReadAll(r)
		os.Stdout = oldStdout
		Expect(err).NotTo(HaveOccurred())

		Expect(string(out)).To(MatchJSON(`{
    "cniVersion": "0.3.1",
    "interfaces": [
        {
            "name": "eth0",
            "mac": "00:11:22:33:44:55",
            "sandbox": "/proc/3553/ns/net"
        }
    ],
    "ips": [
        {
            "version": "4",
            "interface": 0,
            "address": "1.2.3.30/24",
            "gateway": "1.2.3.1"
        },
        {
            "version": "6",
            "interface": 0,
            "address": "abcd:1234:ffff::cdde/64",
            "gateway": "abcd:1234:ffff::1"
        }
    ],
    "routes": [
        {
            "dst": "15.5.6.0/24",
            "gw": "15.5.6.8"
        },
        {
            "dst": "1111:dddd::/80",
            "gw": "1111:dddd::aaaa"
        }
    ],
    "dns": {
        "nameservers": [
            "1.2.3.4",
            "1::cafe"
        ],
        "domain": "acompany.com",
        "search": [
            "somedomain.com",
            "otherdomain.net"
        ],
        "options": [
            "foo",
            "bar"
        ]
    }
}`))
	})

	It("correctly encodes a 0.1.0 Result", func() {
		res, err := testResult().GetAsVersion("0.1.0")
		Expect(err).NotTo(HaveOccurred())

		Expect(res.String()).To(Equal("IP4:{IP:{IP:1.2.3.30 Mask:ffffff00} Gateway:1.2.3.1 Routes:[{Dst:{IP:15.5.6.0 Mask:ffffff00} GW:15.5.6.8}]}, IP6:{IP:{IP:abcd:1234:ffff::cdde Mask:ffffffffffffffff0000000000000000} Gateway:abcd:1234:ffff::1 Routes:[{Dst:{IP:1111:dddd:: Mask:ffffffffffffffffffff000000000000} GW:1111:dddd::aaaa}]}, DNS:{Nameservers:[1.2.3.4 1::cafe] Domain:acompany.com Search:[somedomain.com otherdomain.net] Options:[foo bar]}"))

		// Redirect stdout to capture JSON result
		oldStdout := os.Stdout
		r, w, err := os.Pipe()
		Expect(err).NotTo(HaveOccurred())

		os.Stdout = w
		err = res.Print()
		w.Close()
		Expect(err).NotTo(HaveOccurred())

		// parse the result
		out, err := ioutil.ReadAll(r)
		os.Stdout = oldStdout
		Expect(err).NotTo(HaveOccurred())

		Expect(string(out)).To(MatchJSON(`{
    "cniVersion": "0.2.0",
    "ip4": {
        "ip": "1.2.3.30/24",
        "gateway": "1.2.3.1",
        "routes": [
            {
                "dst": "15.5.6.0/24",
                "gw": "15.5.6.8"
            }
        ]
    },
    "ip6": {
        "ip": "abcd:1234:ffff::cdde/64",
        "gateway": "abcd:1234:ffff::1",
        "routes": [
            {
                "dst": "1111:dddd::/80",
                "gw": "1111:dddd::aaaa"
            }
        ]
    },
    "dns": {
        "nameservers": [
            "1.2.3.4",
            "1::cafe"
        ],
        "domain": "acompany.com",
        "search": [
            "somedomain.com",
            "otherdomain.net"
        ],
        "options": [
            "foo",
            "bar"
        ]
    }
}`))
	})

	It("correctly marshals interface index 0", func() {
		ipc := &current.IPConfig{
			Version:   "4",
			Interface: current.Int(0),
			Address: net.IPNet{
				IP:   net.ParseIP("10.1.2.3"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
		}

		json, err := json.Marshal(ipc)
		Expect(err).NotTo(HaveOccurred())
		Expect(json).To(MatchJSON(`{
    "version": "4",
    "interface": 0,
    "address": "10.1.2.3/24"
}`))
	})

	It("correctly marshals a missing interface index", func() {
		ipc := &current.IPConfig{
			Version: "4",
			Address: net.IPNet{
				IP:   net.ParseIP("10.1.2.3"),
				Mask: net.IPv4Mask(255, 255, 255, 0),
			},
		}

		json, err := json.Marshal(ipc)
		Expect(err).NotTo(HaveOccurred())
		Expect(json).To(MatchJSON(`{
    "version": "4",
    "address": "10.1.2.3/24"
}`))
	})
})
