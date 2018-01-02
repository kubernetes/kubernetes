package config

import (
	"crypto/rsa"
	"os"
	"testing"

	"github.com/cloudflare/cfssl/log"

	_ "github.com/mattn/go-sqlite3" // import just to initialize SQLite for testing
)

// UnlinkIfExists removes a file if it exists.
func UnlinkIfExists(file string) {
	_, err := os.Stat(file)
	if err != nil && os.IsNotExist(err) {
		panic("failed to remove " + file)
	}
	os.Remove(file)
}

// stringSlicesEqual compares two string lists, checking that they
// contain the same elements.
func stringSlicesEqual(slice1, slice2 []string) bool {
	if len(slice1) != len(slice2) {
		return false
	}

	for i := range slice1 {
		if slice1[i] != slice2[i] {
			return false
		}
	}

	for i := range slice2 {
		if slice1[i] != slice2[i] {
			return false
		}
	}
	return true
}

func TestGoodConfig(t *testing.T) {
	testFile := "testdata/test.conf"
	cmap, err := ParseToRawMap(testFile)
	if err != nil {
		t.Fatalf("%v", err)
	} else if len(cmap) != 2 {
		t.Fatal("expected 2 sections, have", len(cmap))
	}
}

func TestGoodConfig2(t *testing.T) {
	testFile := "testdata/test2.conf"
	cmap, err := ParseToRawMap(testFile)
	if err != nil {
		t.Fatalf("%v", err)
	} else if len(cmap) != 1 {
		t.Fatal("expected 1 section, have", len(cmap))
	} else if len(cmap["default"]) != 3 {
		t.Fatal("expected 3 items in default section, have", len(cmap["default"]))
	}
}

func TestBadConfig(t *testing.T) {
	testFile := "testdata/bad.conf"
	_, err := ParseToRawMap(testFile)
	if err == nil {
		t.Fatal("expected invalid config file to fail")
	}
}

func TestQuotedValue(t *testing.T) {
	testFile := "testdata/test.conf"
	cmap, _ := ParseToRawMap(testFile)
	val := cmap["sectionName"]["key4"]
	if val != " space at beginning and end " {
		t.Fatal("Wrong value in double quotes [", val, "]")
	}

	if !cmap.SectionInConfig("sectionName") {
		t.Fatal("expected SectionInConfig to return true")
	}

	val = cmap["sectionName"]["key5"]
	if val != " is quoted with single quotes " {
		t.Fatal("Wrong value in single quotes [", val, "]")
	}
}

func TestENoEnt(t *testing.T) {
	_, err := ParseToRawMap("testdata/enoent")
	if err == nil {
		t.Fatal("expected error on non-existent file")
	}
}

func TestLoadRoots(t *testing.T) {
	roots, err := Parse("testdata/roots.conf")
	if err != nil {
		t.Fatalf("%v", err)
	}

	if len(roots) != 2 {
		t.Fatal("expected one CA in the roots")
	}

	if root, ok := roots["primary"]; !ok {
		t.Fatal("expected a primary CA section")
	} else if _, ok := root.PrivateKey.(*rsa.PrivateKey); !ok {
		t.Fatal("expected an RSA private key")
	}
}

func TestLoadDERRoots(t *testing.T) {
	roots, err := Parse("testdata/roots_der.conf")
	if err != nil {
		t.Fatalf("%v", err)
	}

	if len(roots) != 2 {
		t.Fatal("expected one CA in the roots")
	}

	if root, ok := roots["primary"]; !ok {
		t.Fatal("expected a primary CA section")
	} else if _, ok := root.PrivateKey.(*rsa.PrivateKey); !ok {
		t.Fatal("expected an RSA private key")
	}
}

func TestLoadKSMRoot(t *testing.T) {
	_, err := Parse("testdata/roots_ksm.conf")
	if err == nil {
		t.Fatal("ksm specs are not supported yet")
	}
}

func TestLoadBadRootConfs(t *testing.T) {
	confs := []string{
		"testdata/roots_bad_db.conf",
		"testdata/roots_bad_certificate.conf",
		"testdata/roots_bad_private_key.conf",
		"testdata/roots_badconfig.conf",
		"testdata/roots_badspec.conf",
		"testdata/roots_badspec2.conf",
		"testdata/roots_badspec3.conf",
		"testdata/roots_bad_whitelist.conf",
		"testdata/roots_bad_whitelist.conf2",
		"testdata/roots_missing_certificate.conf",
		"testdata/roots_missing_certificate_entry.conf",
		"testdata/roots_missing_private_key.conf",
		"testdata/roots_missing_private_key_entry.conf",
	}

	for _, cf := range confs {
		_, err := Parse(cf)
		if err == nil {
			t.Fatalf("expected config file %s to fail", cf)
		}
		log.Debugf("%s: %v", cf, err)
	}
}

const confWhitelist = "testdata/roots_whitelist.conf"

func TestLoadWhitelist(t *testing.T) {
	roots, err := Parse(confWhitelist)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if roots["backup"].ACL != nil {
		t.Fatal("Expected a nil ACL for the backup root")
	}

	if roots["primary"].ACL == nil {
		t.Fatal("Expected a non-nil ACL for the primary root")
	}

	validIPs := [][]byte{
		{10, 0, 2, 3},
		{10, 0, 2, 247},
		{172, 16, 3, 9},
		{192, 168, 3, 15},
	}
	badIPs := [][]byte{
		{192, 168, 0, 1},
		{127, 0, 0, 1},
		{192, 168, 3, 14},
		{192, 168, 3, 16},
		{255, 255, 0, 1},
		{0, 0, 0, 0},
	}

	wl := roots["primary"].ACL
	for i := range validIPs {
		if !wl.Permitted(validIPs[i]) {
			t.Fatalf("ACL should have permitted IP %v", validIPs[i])
		}
	}

	for i := range badIPs {
		if wl.Permitted(badIPs[i]) {
			t.Fatalf("ACL should not have permitted IP %v", badIPs[i])
		}
	}
}

const confWhitelistIPv6 = "testdata/roots_whitelist_ipv6.conf"

func TestLoadIPv6Whitelist(t *testing.T) {
	roots, err := Parse(confWhitelistIPv6)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if roots["backup"].ACL != nil {
		t.Fatal("Expected a nil ACL for the backup root")
	}

	if roots["primary"].ACL == nil {
		t.Fatal("Expected a non-nil ACL for the primary root")
	}

	validIPs := [][]byte{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
		{253, 77, 152, 85, 16, 29, 230, 139, 0, 0, 0, 0, 0, 0, 0, 1},
		{253, 77, 152, 85, 16, 29, 230, 139, 0, 0, 0, 0, 32, 0, 0, 1},
		{10, 0, 4, 18},
	}

	badIPs := [][]byte{
		{192, 168, 0, 1},
		{127, 0, 0, 1},
		{192, 168, 3, 14},
		{192, 168, 3, 16},
		{255, 255, 0, 1},
		{0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2},
		{253, 77, 152, 85, 16, 29, 230, 140, 0, 0, 0, 0, 0, 0, 0, 1},
		{254, 77, 152, 85, 16, 29, 230, 139, 0, 0, 0, 0, 0, 0, 0, 1},
	}

	wl := roots["primary"].ACL
	for i := range validIPs {
		if !wl.Permitted(validIPs[i]) {
			t.Fatalf("ACL should have permitted IP %v", validIPs[i])
		}
	}

	for i := range badIPs {
		if wl.Permitted(badIPs[i]) {
			t.Fatalf("ACL should not have permitted IP %v", badIPs[i])
		}
	}
}

const confDBConfig = "testdata/roots_db.conf"

func TestLoadDBConfig(t *testing.T) {
	roots, err := Parse(confDBConfig)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if roots["backup"].DB != nil {
		t.Fatal("Expected a nil DB for the backup root")
	}

	if roots["primary"].DB == nil {
		t.Fatal("Expected a non-nil DB for the primary root")
	}
}
