package resolvconf

import (
	"bytes"
	"io/ioutil"
	"os"
	"testing"

	"github.com/docker/docker/pkg/ioutils"
	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func TestGet(t *testing.T) {
	resolvConfUtils, err := Get()
	if err != nil {
		t.Fatal(err)
	}
	resolvConfSystem, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		t.Fatal(err)
	}
	if string(resolvConfUtils.Content) != string(resolvConfSystem) {
		t.Fatalf("/etc/resolv.conf and GetResolvConf have different content.")
	}
	hashSystem, err := ioutils.HashData(bytes.NewReader(resolvConfSystem))
	if err != nil {
		t.Fatal(err)
	}
	if resolvConfUtils.Hash != hashSystem {
		t.Fatalf("/etc/resolv.conf and GetResolvConf have different hashes.")
	}
}

func TestGetNameservers(t *testing.T) {
	for resolv, result := range map[string][]string{`
nameserver 1.2.3.4
nameserver 40.3.200.10
search example.com`: {"1.2.3.4", "40.3.200.10"},
		`search example.com`: {},
		`nameserver 1.2.3.4
search example.com
nameserver 4.30.20.100`: {"1.2.3.4", "4.30.20.100"},
		``: {},
		`  nameserver 1.2.3.4   `: {"1.2.3.4"},
		`search example.com
nameserver 1.2.3.4
#nameserver 4.3.2.1`: {"1.2.3.4"},
		`search example.com
nameserver 1.2.3.4 # not 4.3.2.1`: {"1.2.3.4"},
	} {
		test := GetNameservers([]byte(resolv), types.IP)
		if !strSlicesEqual(test, result) {
			t.Fatalf("Wrong nameserver string {%s} should be %v. Input: %s", test, result, resolv)
		}
	}
}

func TestGetNameserversAsCIDR(t *testing.T) {
	for resolv, result := range map[string][]string{`
nameserver 1.2.3.4
nameserver 40.3.200.10
search example.com`: {"1.2.3.4/32", "40.3.200.10/32"},
		`search example.com`: {},
		`nameserver 1.2.3.4
search example.com
nameserver 4.30.20.100`: {"1.2.3.4/32", "4.30.20.100/32"},
		``: {},
		`  nameserver 1.2.3.4   `: {"1.2.3.4/32"},
		`search example.com
nameserver 1.2.3.4
#nameserver 4.3.2.1`: {"1.2.3.4/32"},
		`search example.com
nameserver 1.2.3.4 # not 4.3.2.1`: {"1.2.3.4/32"},
	} {
		test := GetNameserversAsCIDR([]byte(resolv))
		if !strSlicesEqual(test, result) {
			t.Fatalf("Wrong nameserver string {%s} should be %v. Input: %s", test, result, resolv)
		}
	}
}

func TestGetSearchDomains(t *testing.T) {
	for resolv, result := range map[string][]string{
		`search example.com`:           {"example.com"},
		`search example.com # ignored`: {"example.com"},
		`	  search	 example.com	  `: {"example.com"},
		`	  search	 example.com	  # ignored`: {"example.com"},
		`search foo.example.com example.com`: {"foo.example.com", "example.com"},
		`	   search	   foo.example.com	 example.com	`: {"foo.example.com", "example.com"},
		`	   search	   foo.example.com	 example.com	# ignored`: {"foo.example.com", "example.com"},
		``:          {},
		`# ignored`: {},
		`nameserver 1.2.3.4
search foo.example.com example.com`: {"foo.example.com", "example.com"},
		`nameserver 1.2.3.4
search dup1.example.com dup2.example.com
search foo.example.com example.com`: {"foo.example.com", "example.com"},
		`nameserver 1.2.3.4
search foo.example.com example.com
nameserver 4.30.20.100`: {"foo.example.com", "example.com"},
	} {
		test := GetSearchDomains([]byte(resolv))
		if !strSlicesEqual(test, result) {
			t.Fatalf("Wrong search domain string {%s} should be %v. Input: %s", test, result, resolv)
		}
	}
}

func TestGetOptions(t *testing.T) {
	for resolv, result := range map[string][]string{
		`options opt1`:           {"opt1"},
		`options opt1 # ignored`: {"opt1"},
		`	  options	 opt1	  `: {"opt1"},
		`	  options	 opt1	  # ignored`: {"opt1"},
		`options opt1 opt2 opt3`:           {"opt1", "opt2", "opt3"},
		`options opt1 opt2 opt3 # ignored`: {"opt1", "opt2", "opt3"},
		`	   options	 opt1	 opt2	 opt3	`: {"opt1", "opt2", "opt3"},
		`	   options	 opt1	 opt2	 opt3	# ignored`: {"opt1", "opt2", "opt3"},
		``:                   {},
		`# ignored`:          {},
		`nameserver 1.2.3.4`: {},
		`nameserver 1.2.3.4
options opt1 opt2 opt3`: {"opt1", "opt2", "opt3"},
		`nameserver 1.2.3.4
options opt1 opt2
options opt3 opt4`: {"opt3", "opt4"},
	} {
		test := GetOptions([]byte(resolv))
		if !strSlicesEqual(test, result) {
			t.Fatalf("Wrong options string {%s} should be %v. Input: %s", test, result, resolv)
		}
	}
}

func strSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}

func TestBuild(t *testing.T) {
	file, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())

	_, err = Build(file.Name(), []string{"ns1", "ns2", "ns3"}, []string{"search1"}, []string{"opt1"})
	if err != nil {
		t.Fatal(err)
	}

	content, err := ioutil.ReadFile(file.Name())
	if err != nil {
		t.Fatal(err)
	}

	if expected := "search search1\nnameserver ns1\nnameserver ns2\nnameserver ns3\noptions opt1\n"; !bytes.Contains(content, []byte(expected)) {
		t.Fatalf("Expected to find '%s' got '%s'", expected, content)
	}
}

func TestBuildWithZeroLengthDomainSearch(t *testing.T) {
	file, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())

	_, err = Build(file.Name(), []string{"ns1", "ns2", "ns3"}, []string{"."}, []string{"opt1"})
	if err != nil {
		t.Fatal(err)
	}

	content, err := ioutil.ReadFile(file.Name())
	if err != nil {
		t.Fatal(err)
	}

	if expected := "nameserver ns1\nnameserver ns2\nnameserver ns3\noptions opt1\n"; !bytes.Contains(content, []byte(expected)) {
		t.Fatalf("Expected to find '%s' got '%s'", expected, content)
	}
	if notExpected := "search ."; bytes.Contains(content, []byte(notExpected)) {
		t.Fatalf("Expected to not find '%s' got '%s'", notExpected, content)
	}
}

func TestBuildWithNoOptions(t *testing.T) {
	file, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())

	_, err = Build(file.Name(), []string{"ns1", "ns2", "ns3"}, []string{"search1"}, []string{})
	if err != nil {
		t.Fatal(err)
	}

	content, err := ioutil.ReadFile(file.Name())
	if err != nil {
		t.Fatal(err)
	}

	if expected := "search search1\nnameserver ns1\nnameserver ns2\nnameserver ns3\n"; !bytes.Contains(content, []byte(expected)) {
		t.Fatalf("Expected to find '%s' got '%s'", expected, content)
	}
	if notExpected := "search ."; bytes.Contains(content, []byte(notExpected)) {
		t.Fatalf("Expected to not find '%s' got '%s'", notExpected, content)
	}
}

func TestFilterResolvDns(t *testing.T) {
	ns0 := "nameserver 10.16.60.14\nnameserver 10.16.60.21\n"

	if result, _ := FilterResolvDNS([]byte(ns0), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed No Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	ns1 := "nameserver 10.16.60.14\nnameserver 10.16.60.21\nnameserver 127.0.0.1\n"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	ns1 = "nameserver 10.16.60.14\nnameserver 127.0.0.1\nnameserver 10.16.60.21\n"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	ns1 = "nameserver 127.0.1.1\nnameserver 10.16.60.14\nnameserver 10.16.60.21\n"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	ns1 = "nameserver ::1\nnameserver 10.16.60.14\nnameserver 127.0.2.1\nnameserver 10.16.60.21\n"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	ns1 = "nameserver 10.16.60.14\nnameserver ::1\nnameserver 10.16.60.21\nnameserver ::1"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	// with IPv6 disabled (false param), the IPv6 nameserver should be removed
	ns1 = "nameserver 10.16.60.14\nnameserver 2002:dead:beef::1\nnameserver 10.16.60.21\nnameserver ::1"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost+IPv6 off: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	// with IPv6 disabled (false param), the IPv6 link-local nameserver with zone ID should be removed
	ns1 = "nameserver 10.16.60.14\nnameserver FE80::BB1%1\nnameserver FE80::BB1%eth0\nnameserver 10.16.60.21\n"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost+IPv6 off: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	// with IPv6 enabled, the IPv6 nameserver should be preserved
	ns0 = "nameserver 10.16.60.14\nnameserver 2002:dead:beef::1\nnameserver 10.16.60.21\n"
	ns1 = "nameserver 10.16.60.14\nnameserver 2002:dead:beef::1\nnameserver 10.16.60.21\nnameserver ::1"
	if result, _ := FilterResolvDNS([]byte(ns1), true); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed Localhost+IPv6 on: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	// with IPv6 enabled, and no non-localhost servers, Google defaults (both IPv4+IPv6) should be added
	ns0 = "\nnameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 2001:4860:4860::8888\nnameserver 2001:4860:4860::8844"
	ns1 = "nameserver 127.0.0.1\nnameserver ::1\nnameserver 127.0.2.1"
	if result, _ := FilterResolvDNS([]byte(ns1), true); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed no Localhost+IPv6 enabled: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}

	// with IPv6 disabled, and no non-localhost servers, Google defaults (only IPv4) should be added
	ns0 = "\nnameserver 8.8.8.8\nnameserver 8.8.4.4"
	ns1 = "nameserver 127.0.0.1\nnameserver ::1\nnameserver 127.0.2.1"
	if result, _ := FilterResolvDNS([]byte(ns1), false); result != nil {
		if ns0 != string(result.Content) {
			t.Fatalf("Failed no Localhost+IPv6 enabled: expected \n<%s> got \n<%s>", ns0, string(result.Content))
		}
	}
}
