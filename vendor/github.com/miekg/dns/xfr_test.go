// +build net

package dns

import (
	"net"
	"testing"
	"time"
)

func getIP(s string) string {
	a, err := net.LookupAddr(s)
	if err != nil {
		return ""
	}
	return a[0]
}

// flaky, need to setup local server and test from
// that.
func TestAXFR_Miek(t *testing.T) {
	// This test runs against a server maintained by Miek
	if testing.Short() {
		return
	}
	m := new(Msg)
	m.SetAxfr("miek.nl.")

	server := getIP("linode.atoom.net")

	tr := new(Transfer)

	if a, err := tr.In(m, net.JoinHostPort(server, "53")); err != nil {
		t.Fatal("failed to setup axfr: ", err)
	} else {
		for ex := range a {
			if ex.Error != nil {
				t.Errorf("error %v", ex.Error)
				break
			}
			for _, rr := range ex.RR {
				t.Log(rr.String())
			}
		}
	}
}

// fails.
func TestAXFR_NLNL_MultipleEnvelopes(t *testing.T) {
	// This test runs against a server maintained by NLnet Labs
	if testing.Short() {
		return
	}
	m := new(Msg)
	m.SetAxfr("nlnetlabs.nl.")

	server := getIP("open.nlnetlabs.nl.")

	tr := new(Transfer)
	if a, err := tr.In(m, net.JoinHostPort(server, "53")); err != nil {
		t.Fatalf("failed to setup axfr %v for server: %v", err, server)
	} else {
		for ex := range a {
			if ex.Error != nil {
				t.Errorf("error %v", ex.Error)
				break
			}
		}
	}
}

func TestAXFR_Miek_Tsig(t *testing.T) {
	// This test runs against a server maintained by Miek
	if testing.Short() {
		return
	}
	m := new(Msg)
	m.SetAxfr("example.nl.")
	m.SetTsig("axfr.", HmacMD5, 300, time.Now().Unix())

	tr := new(Transfer)
	tr.TsigSecret = map[string]string{"axfr.": "so6ZGir4GPAqINNh9U5c3A=="}

	if a, err := tr.In(m, "176.58.119.54:53"); err != nil {
		t.Fatal("failed to setup axfr: ", err)
	} else {
		for ex := range a {
			if ex.Error != nil {
				t.Errorf("error %v", ex.Error)
				break
			}
			for _, rr := range ex.RR {
				t.Log(rr.String())
			}
		}
	}
}

func TestAXFR_SIDN_NSD3_NONE(t *testing.T)   { testAXFRSIDN(t, "nsd", "") }
func TestAXFR_SIDN_NSD3_MD5(t *testing.T)    { testAXFRSIDN(t, "nsd", HmacMD5) }
func TestAXFR_SIDN_NSD3_SHA1(t *testing.T)   { testAXFRSIDN(t, "nsd", HmacSHA1) }
func TestAXFR_SIDN_NSD3_SHA256(t *testing.T) { testAXFRSIDN(t, "nsd", HmacSHA256) }

func TestAXFR_SIDN_NSD4_NONE(t *testing.T)   { testAXFRSIDN(t, "nsd4", "") }
func TestAXFR_SIDN_NSD4_MD5(t *testing.T)    { testAXFRSIDN(t, "nsd4", HmacMD5) }
func TestAXFR_SIDN_NSD4_SHA1(t *testing.T)   { testAXFRSIDN(t, "nsd4", HmacSHA1) }
func TestAXFR_SIDN_NSD4_SHA256(t *testing.T) { testAXFRSIDN(t, "nsd4", HmacSHA256) }

func TestAXFR_SIDN_BIND9_NONE(t *testing.T)   { testAXFRSIDN(t, "bind9", "") }
func TestAXFR_SIDN_BIND9_MD5(t *testing.T)    { testAXFRSIDN(t, "bind9", HmacMD5) }
func TestAXFR_SIDN_BIND9_SHA1(t *testing.T)   { testAXFRSIDN(t, "bind9", HmacSHA1) }
func TestAXFR_SIDN_BIND9_SHA256(t *testing.T) { testAXFRSIDN(t, "bind9", HmacSHA256) }

func TestAXFR_SIDN_KNOT_NONE(t *testing.T)   { testAXFRSIDN(t, "knot", "") }
func TestAXFR_SIDN_KNOT_MD5(t *testing.T)    { testAXFRSIDN(t, "knot", HmacMD5) }
func TestAXFR_SIDN_KNOT_SHA1(t *testing.T)   { testAXFRSIDN(t, "knot", HmacSHA1) }
func TestAXFR_SIDN_KNOT_SHA256(t *testing.T) { testAXFRSIDN(t, "knot", HmacSHA256) }

func TestAXFR_SIDN_POWERDNS_NONE(t *testing.T)   { testAXFRSIDN(t, "powerdns", "") }
func TestAXFR_SIDN_POWERDNS_MD5(t *testing.T)    { testAXFRSIDN(t, "powerdns", HmacMD5) }
func TestAXFR_SIDN_POWERDNS_SHA1(t *testing.T)   { testAXFRSIDN(t, "powerdns", HmacSHA1) }
func TestAXFR_SIDN_POWERDNS_SHA256(t *testing.T) { testAXFRSIDN(t, "powerdns", HmacSHA256) }

func TestAXFR_SIDN_YADIFA_NONE(t *testing.T)   { testAXFRSIDN(t, "yadifa", "") }
func TestAXFR_SIDN_YADIFA_MD5(t *testing.T)    { testAXFRSIDN(t, "yadifa", HmacMD5) }
func TestAXFR_SIDN_YADIFA_SHA1(t *testing.T)   { testAXFRSIDN(t, "yadifa", HmacSHA1) }
func TestAXFR_SIDN_YADIFA_SHA256(t *testing.T) { testAXFRSIDN(t, "yadifa", HmacSHA256) }

func testAXFRSIDN(t *testing.T, host, alg string) {
	// This tests run against a server maintained by SIDN labs, see:
	// https://workbench.sidnlabs.nl/
	if testing.Short() {
		return
	}
	x := new(Transfer)
	x.TsigSecret = map[string]string{
		"wb_md5.":          "Wu/utSasZUkoeCNku152Zw==",
		"wb_sha1_longkey.": "uhMpEhPq/RAD9Bt4mqhfmi+7ZdKmjLQb/lcrqYPXR4s/nnbsqw==",
		"wb_sha256.":       "npfrIJjt/MJOjGJoBNZtsjftKMhkSpIYMv2RzRZt1f8=",
	}
	keyname := map[string]string{
		HmacMD5:    "wb_md5.",
		HmacSHA1:   "wb_sha1_longkey.",
		HmacSHA256: "wb_sha256.",
	}[alg]

	m := new(Msg)
	m.SetAxfr("types.wb.sidnlabs.nl.")
	if keyname != "" {
		m.SetTsig(keyname, alg, 300, time.Now().Unix())
	}
	c, err := x.In(m, host+".sidnlabs.nl:53")
	if err != nil {
		t.Fatal(err)
	}
	for e := range c {
		if e.Error != nil {
			t.Fatal(e.Error)
		}
	}
}
