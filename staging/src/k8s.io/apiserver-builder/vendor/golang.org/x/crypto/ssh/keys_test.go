// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/crypto/ed25519"
	"golang.org/x/crypto/ssh/testdata"
)

func rawKey(pub PublicKey) interface{} {
	switch k := pub.(type) {
	case *rsaPublicKey:
		return (*rsa.PublicKey)(k)
	case *dsaPublicKey:
		return (*dsa.PublicKey)(k)
	case *ecdsaPublicKey:
		return (*ecdsa.PublicKey)(k)
	case ed25519PublicKey:
		return (ed25519.PublicKey)(k)
	case *Certificate:
		return k
	}
	panic("unknown key type")
}

func TestKeyMarshalParse(t *testing.T) {
	for _, priv := range testSigners {
		pub := priv.PublicKey()
		roundtrip, err := ParsePublicKey(pub.Marshal())
		if err != nil {
			t.Errorf("ParsePublicKey(%T): %v", pub, err)
		}

		k1 := rawKey(pub)
		k2 := rawKey(roundtrip)

		if !reflect.DeepEqual(k1, k2) {
			t.Errorf("got %#v in roundtrip, want %#v", k2, k1)
		}
	}
}

func TestUnsupportedCurves(t *testing.T) {
	raw, err := ecdsa.GenerateKey(elliptic.P224(), rand.Reader)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	if _, err = NewSignerFromKey(raw); err == nil || !strings.Contains(err.Error(), "only P-256") {
		t.Fatalf("NewPrivateKey should not succeed with P-224, got: %v", err)
	}

	if _, err = NewPublicKey(&raw.PublicKey); err == nil || !strings.Contains(err.Error(), "only P-256") {
		t.Fatalf("NewPublicKey should not succeed with P-224, got: %v", err)
	}
}

func TestNewPublicKey(t *testing.T) {
	for _, k := range testSigners {
		raw := rawKey(k.PublicKey())
		// Skip certificates, as NewPublicKey does not support them.
		if _, ok := raw.(*Certificate); ok {
			continue
		}
		pub, err := NewPublicKey(raw)
		if err != nil {
			t.Errorf("NewPublicKey(%#v): %v", raw, err)
		}
		if !reflect.DeepEqual(k.PublicKey(), pub) {
			t.Errorf("NewPublicKey(%#v) = %#v, want %#v", raw, pub, k.PublicKey())
		}
	}
}

func TestKeySignVerify(t *testing.T) {
	for _, priv := range testSigners {
		pub := priv.PublicKey()

		data := []byte("sign me")
		sig, err := priv.Sign(rand.Reader, data)
		if err != nil {
			t.Fatalf("Sign(%T): %v", priv, err)
		}

		if err := pub.Verify(data, sig); err != nil {
			t.Errorf("publicKey.Verify(%T): %v", priv, err)
		}
		sig.Blob[5]++
		if err := pub.Verify(data, sig); err == nil {
			t.Errorf("publicKey.Verify on broken sig did not fail")
		}
	}
}

func TestParseRSAPrivateKey(t *testing.T) {
	key := testPrivateKeys["rsa"]

	rsa, ok := key.(*rsa.PrivateKey)
	if !ok {
		t.Fatalf("got %T, want *rsa.PrivateKey", rsa)
	}

	if err := rsa.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

func TestParseECPrivateKey(t *testing.T) {
	key := testPrivateKeys["ecdsa"]

	ecKey, ok := key.(*ecdsa.PrivateKey)
	if !ok {
		t.Fatalf("got %T, want *ecdsa.PrivateKey", ecKey)
	}

	if !validateECPublicKey(ecKey.Curve, ecKey.X, ecKey.Y) {
		t.Fatalf("public key does not validate.")
	}
}

// See Issue https://github.com/golang/go/issues/6650.
func TestParseEncryptedPrivateKeysFails(t *testing.T) {
	const wantSubstring = "encrypted"
	for i, tt := range testdata.PEMEncryptedKeys {
		_, err := ParsePrivateKey(tt.PEMBytes)
		if err == nil {
			t.Errorf("#%d key %s: ParsePrivateKey successfully parsed, expected an error", i, tt.Name)
			continue
		}

		if !strings.Contains(err.Error(), wantSubstring) {
			t.Errorf("#%d key %s: got error %q, want substring %q", i, tt.Name, err, wantSubstring)
		}
	}
}

func TestParseDSA(t *testing.T) {
	// We actually exercise the ParsePrivateKey codepath here, as opposed to
	// using the ParseRawPrivateKey+NewSignerFromKey path that testdata_test.go
	// uses.
	s, err := ParsePrivateKey(testdata.PEMBytes["dsa"])
	if err != nil {
		t.Fatalf("ParsePrivateKey returned error: %s", err)
	}

	data := []byte("sign me")
	sig, err := s.Sign(rand.Reader, data)
	if err != nil {
		t.Fatalf("dsa.Sign: %v", err)
	}

	if err := s.PublicKey().Verify(data, sig); err != nil {
		t.Errorf("Verify failed: %v", err)
	}
}

// Tests for authorized_keys parsing.

// getTestKey returns a public key, and its base64 encoding.
func getTestKey() (PublicKey, string) {
	k := testPublicKeys["rsa"]

	b := &bytes.Buffer{}
	e := base64.NewEncoder(base64.StdEncoding, b)
	e.Write(k.Marshal())
	e.Close()

	return k, b.String()
}

func TestMarshalParsePublicKey(t *testing.T) {
	pub, pubSerialized := getTestKey()
	line := fmt.Sprintf("%s %s user@host", pub.Type(), pubSerialized)

	authKeys := MarshalAuthorizedKey(pub)
	actualFields := strings.Fields(string(authKeys))
	if len(actualFields) == 0 {
		t.Fatalf("failed authKeys: %v", authKeys)
	}

	// drop the comment
	expectedFields := strings.Fields(line)[0:2]

	if !reflect.DeepEqual(actualFields, expectedFields) {
		t.Errorf("got %v, expected %v", actualFields, expectedFields)
	}

	actPub, _, _, _, err := ParseAuthorizedKey([]byte(line))
	if err != nil {
		t.Fatalf("cannot parse %v: %v", line, err)
	}
	if !reflect.DeepEqual(actPub, pub) {
		t.Errorf("got %v, expected %v", actPub, pub)
	}
}

type authResult struct {
	pubKey   PublicKey
	options  []string
	comments string
	rest     string
	ok       bool
}

func testAuthorizedKeys(t *testing.T, authKeys []byte, expected []authResult) {
	rest := authKeys
	var values []authResult
	for len(rest) > 0 {
		var r authResult
		var err error
		r.pubKey, r.comments, r.options, rest, err = ParseAuthorizedKey(rest)
		r.ok = (err == nil)
		t.Log(err)
		r.rest = string(rest)
		values = append(values, r)
	}

	if !reflect.DeepEqual(values, expected) {
		t.Errorf("got %#v, expected %#v", values, expected)
	}
}

func TestAuthorizedKeyBasic(t *testing.T) {
	pub, pubSerialized := getTestKey()
	line := "ssh-rsa " + pubSerialized + " user@host"
	testAuthorizedKeys(t, []byte(line),
		[]authResult{
			{pub, nil, "user@host", "", true},
		})
}

func TestAuth(t *testing.T) {
	pub, pubSerialized := getTestKey()
	authWithOptions := []string{
		`# comments to ignore before any keys...`,
		``,
		`env="HOME=/home/root",no-port-forwarding ssh-rsa ` + pubSerialized + ` user@host`,
		`# comments to ignore, along with a blank line`,
		``,
		`env="HOME=/home/root2" ssh-rsa ` + pubSerialized + ` user2@host2`,
		``,
		`# more comments, plus a invalid entry`,
		`ssh-rsa data-that-will-not-parse user@host3`,
	}
	for _, eol := range []string{"\n", "\r\n"} {
		authOptions := strings.Join(authWithOptions, eol)
		rest2 := strings.Join(authWithOptions[3:], eol)
		rest3 := strings.Join(authWithOptions[6:], eol)
		testAuthorizedKeys(t, []byte(authOptions), []authResult{
			{pub, []string{`env="HOME=/home/root"`, "no-port-forwarding"}, "user@host", rest2, true},
			{pub, []string{`env="HOME=/home/root2"`}, "user2@host2", rest3, true},
			{nil, nil, "", "", false},
		})
	}
}

func TestAuthWithQuotedSpaceInEnv(t *testing.T) {
	pub, pubSerialized := getTestKey()
	authWithQuotedSpaceInEnv := []byte(`env="HOME=/home/root dir",no-port-forwarding ssh-rsa ` + pubSerialized + ` user@host`)
	testAuthorizedKeys(t, []byte(authWithQuotedSpaceInEnv), []authResult{
		{pub, []string{`env="HOME=/home/root dir"`, "no-port-forwarding"}, "user@host", "", true},
	})
}

func TestAuthWithQuotedCommaInEnv(t *testing.T) {
	pub, pubSerialized := getTestKey()
	authWithQuotedCommaInEnv := []byte(`env="HOME=/home/root,dir",no-port-forwarding ssh-rsa ` + pubSerialized + `   user@host`)
	testAuthorizedKeys(t, []byte(authWithQuotedCommaInEnv), []authResult{
		{pub, []string{`env="HOME=/home/root,dir"`, "no-port-forwarding"}, "user@host", "", true},
	})
}

func TestAuthWithQuotedQuoteInEnv(t *testing.T) {
	pub, pubSerialized := getTestKey()
	authWithQuotedQuoteInEnv := []byte(`env="HOME=/home/\"root dir",no-port-forwarding` + "\t" + `ssh-rsa` + "\t" + pubSerialized + `   user@host`)
	authWithDoubleQuotedQuote := []byte(`no-port-forwarding,env="HOME=/home/ \"root dir\"" ssh-rsa ` + pubSerialized + "\t" + `user@host`)
	testAuthorizedKeys(t, []byte(authWithQuotedQuoteInEnv), []authResult{
		{pub, []string{`env="HOME=/home/\"root dir"`, "no-port-forwarding"}, "user@host", "", true},
	})

	testAuthorizedKeys(t, []byte(authWithDoubleQuotedQuote), []authResult{
		{pub, []string{"no-port-forwarding", `env="HOME=/home/ \"root dir\""`}, "user@host", "", true},
	})
}

func TestAuthWithInvalidSpace(t *testing.T) {
	_, pubSerialized := getTestKey()
	authWithInvalidSpace := []byte(`env="HOME=/home/root dir", no-port-forwarding ssh-rsa ` + pubSerialized + ` user@host
#more to follow but still no valid keys`)
	testAuthorizedKeys(t, []byte(authWithInvalidSpace), []authResult{
		{nil, nil, "", "", false},
	})
}

func TestAuthWithMissingQuote(t *testing.T) {
	pub, pubSerialized := getTestKey()
	authWithMissingQuote := []byte(`env="HOME=/home/root,no-port-forwarding ssh-rsa ` + pubSerialized + ` user@host
env="HOME=/home/root",shared-control ssh-rsa ` + pubSerialized + ` user@host`)

	testAuthorizedKeys(t, []byte(authWithMissingQuote), []authResult{
		{pub, []string{`env="HOME=/home/root"`, `shared-control`}, "user@host", "", true},
	})
}

func TestInvalidEntry(t *testing.T) {
	authInvalid := []byte(`ssh-rsa`)
	_, _, _, _, err := ParseAuthorizedKey(authInvalid)
	if err == nil {
		t.Errorf("got valid entry for %q", authInvalid)
	}
}

var knownHostsParseTests = []struct {
	input string
	err   string

	marker  string
	comment string
	hosts   []string
	rest    string
}{
	{
		"",
		"EOF",

		"", "", nil, "",
	},
	{
		"# Just a comment",
		"EOF",

		"", "", nil, "",
	},
	{
		"   \t   ",
		"EOF",

		"", "", nil, "",
	},
	{
		"localhost ssh-rsa {RSAPUB}",
		"",

		"", "", []string{"localhost"}, "",
	},
	{
		"localhost\tssh-rsa {RSAPUB}",
		"",

		"", "", []string{"localhost"}, "",
	},
	{
		"localhost\tssh-rsa {RSAPUB}\tcomment comment",
		"",

		"", "comment comment", []string{"localhost"}, "",
	},
	{
		"localhost\tssh-rsa {RSAPUB}\tcomment comment\n",
		"",

		"", "comment comment", []string{"localhost"}, "",
	},
	{
		"localhost\tssh-rsa {RSAPUB}\tcomment comment\r\n",
		"",

		"", "comment comment", []string{"localhost"}, "",
	},
	{
		"localhost\tssh-rsa {RSAPUB}\tcomment comment\r\nnext line",
		"",

		"", "comment comment", []string{"localhost"}, "next line",
	},
	{
		"localhost,[host2:123]\tssh-rsa {RSAPUB}\tcomment comment",
		"",

		"", "comment comment", []string{"localhost", "[host2:123]"}, "",
	},
	{
		"@marker \tlocalhost,[host2:123]\tssh-rsa {RSAPUB}",
		"",

		"marker", "", []string{"localhost", "[host2:123]"}, "",
	},
	{
		"@marker \tlocalhost,[host2:123]\tssh-rsa aabbccdd",
		"short read",

		"", "", nil, "",
	},
}

func TestKnownHostsParsing(t *testing.T) {
	rsaPub, rsaPubSerialized := getTestKey()

	for i, test := range knownHostsParseTests {
		var expectedKey PublicKey
		const rsaKeyToken = "{RSAPUB}"

		input := test.input
		if strings.Contains(input, rsaKeyToken) {
			expectedKey = rsaPub
			input = strings.Replace(test.input, rsaKeyToken, rsaPubSerialized, -1)
		}

		marker, hosts, pubKey, comment, rest, err := ParseKnownHosts([]byte(input))
		if err != nil {
			if len(test.err) == 0 {
				t.Errorf("#%d: unexpectedly failed with %q", i, err)
			} else if !strings.Contains(err.Error(), test.err) {
				t.Errorf("#%d: expected error containing %q, but got %q", i, test.err, err)
			}
			continue
		} else if len(test.err) != 0 {
			t.Errorf("#%d: succeeded but expected error including %q", i, test.err)
			continue
		}

		if !reflect.DeepEqual(expectedKey, pubKey) {
			t.Errorf("#%d: expected key %#v, but got %#v", i, expectedKey, pubKey)
		}

		if marker != test.marker {
			t.Errorf("#%d: expected marker %q, but got %q", i, test.marker, marker)
		}

		if comment != test.comment {
			t.Errorf("#%d: expected comment %q, but got %q", i, test.comment, comment)
		}

		if !reflect.DeepEqual(test.hosts, hosts) {
			t.Errorf("#%d: expected hosts %#v, but got %#v", i, test.hosts, hosts)
		}

		if rest := string(rest); rest != test.rest {
			t.Errorf("#%d: expected remaining input to be %q, but got %q", i, test.rest, rest)
		}
	}
}
