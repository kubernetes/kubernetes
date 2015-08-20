package pkcs11uri

import (
	"fmt"
	"testing"

	"github.com/cloudflare/cfssl/signer/pkcs11"
)

type pkcs11UriTest struct {
	URI    string
	Config *pkcs11.Config
}

func cmpConfigs(a, b *pkcs11.Config) bool {
	if a == nil {
		if b == nil {
			return true
		}
		return false
	}

	if b == nil {
		return false
	}

	return (a.Module == b.Module) &&
		(a.Token == b.Token) &&
		(a.PIN == b.PIN) &&
		(a.Label == b.Label)
}

func diffConfigs(want, have *pkcs11.Config) {
	if have == nil && want != nil {
		fmt.Printf("Expected config, have nil.")
		return
	} else if have == nil && want == nil {
		return
	}

	diff := func(kind, v1, v2 string) {
		if v1 != v2 {
			fmt.Printf("%s: want '%s', have '%s'\n", kind, v1, v2)
		}
	}

	diff("Module", want.Module, have.Module)
	diff("Token", want.Token, have.Token)
	diff("PIN", want.PIN, have.PIN)
	diff("Label", want.Label, have.Label)
}

/* Config from PKCS #11 signer
type Config struct {
	Module string
	Token  string
	PIN    string
	Label  string
}
*/

var pkcs11UriCases = []pkcs11UriTest{
	{"pkcs11:token=Software%20PKCS%2311%20softtoken;manufacturer=Snake%20Oil,%20Inc.?pin-value=the-pin",
		&pkcs11.Config{
			Token: "Software PKCS#11 softtoken",
			PIN:   "the-pin",
		}},
	{"pkcs11:slot-description=Sun%20Metaslot",
		&pkcs11.Config{
			Label: "Sun Metaslot",
		}},
	{"pkcs11:slot-description=test-label;token=test-token?pin-source=file:testdata/pin&module-name=test-module",
		&pkcs11.Config{
			Label:  "test-label",
			Token:  "test-token",
			PIN:    "123456",
			Module: "test-module",
		}},
}

func TestParsePKCS11URI(t *testing.T) {
	for _, c := range pkcs11UriCases {
		cfg, err := ParsePKCS11URI(c.URI)
		if err != nil {
			t.Fatalf("Failed on URI '%s'", c.URI)
		}
		if !cmpConfigs(c.Config, cfg) {
			diffConfigs(c.Config, cfg)
			t.Fatal("Configs don't match.")
		}
	}
}

var pkcs11UriFails = []string{
	"https://github.com/cloudflare/cfssl",
	"pkcs11:?pin-source=http://foo",
	"pkcs11:?pin-source=file:testdata/nosuchfile",
}

func TestParsePKCS11URIFail(t *testing.T) {
	for _, c := range pkcs11UriFails {
		_, err := ParsePKCS11URI(c)
		if err == nil {
			t.Fatalf("Expected URI '%s' to fail to parse.", c)
		}
	}
}
