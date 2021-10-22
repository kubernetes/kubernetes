// +build go1.7

package csm

import "testing"

func TestAddressWithDefaults(t *testing.T) {
	cases := map[string]struct {
		Host, Port string
		Expect     string
	}{
		"ip": {
			Host: "127.0.0.2", Port: "", Expect: "127.0.0.2:31000",
		},
		"localhost": {
			Host: "localhost", Port: "", Expect: "127.0.0.1:31000",
		},
		"uppercase localhost": {
			Host: "LOCALHOST", Port: "", Expect: "127.0.0.1:31000",
		},
		"port": {
			Host: "localhost", Port: "32000", Expect: "127.0.0.1:32000",
		},
		"ip6": {
			Host: "::1", Port: "", Expect: "[::1]:31000",
		},
		"unset": {
			Host: "", Port: "", Expect: "127.0.0.1:31000",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			actual := AddressWithDefaults(c.Host, c.Port)
			if e, a := c.Expect, actual; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}
