// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knownhosts

import (
	"bytes"
	"fmt"
	"net"
	"reflect"
	"testing"

	"golang.org/x/crypto/ssh"
)

const edKeyStr = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGBAarftlLeoyf+v+nVchEZII/vna2PCV8FaX4vsF5BX"
const alternateEdKeyStr = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIXffBYeYL+WVzVru8npl5JHt2cjlr4ornFTWzoij9sx"
const ecKeyStr = "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNLCu01+wpXe3xB5olXCN4SqU2rQu0qjSRKJO4Bg+JRCPU+ENcgdA5srTU8xYDz/GEa4dzK5ldPw4J/gZgSXCMs="

var ecKey, alternateEdKey, edKey ssh.PublicKey
var testAddr = &net.TCPAddr{
	IP:   net.IP{198, 41, 30, 196},
	Port: 22,
}

var testAddr6 = &net.TCPAddr{
	IP: net.IP{198, 41, 30, 196,
		1, 2, 3, 4,
		1, 2, 3, 4,
		1, 2, 3, 4,
	},
	Port: 22,
}

func init() {
	var err error
	ecKey, _, _, _, err = ssh.ParseAuthorizedKey([]byte(ecKeyStr))
	if err != nil {
		panic(err)
	}
	edKey, _, _, _, err = ssh.ParseAuthorizedKey([]byte(edKeyStr))
	if err != nil {
		panic(err)
	}
	alternateEdKey, _, _, _, err = ssh.ParseAuthorizedKey([]byte(alternateEdKeyStr))
	if err != nil {
		panic(err)
	}
}

func testDB(t *testing.T, s string) *hostKeyDB {
	db := newHostKeyDB()
	if err := db.Read(bytes.NewBufferString(s), "testdb"); err != nil {
		t.Fatalf("Read: %v", err)
	}

	return db
}

func TestRevoked(t *testing.T) {
	db := testDB(t, "\n\n@revoked * "+edKeyStr+"\n")
	want := &RevokedError{
		Revoked: KnownKey{
			Key:      edKey,
			Filename: "testdb",
			Line:     3,
		},
	}
	if err := db.check("", &net.TCPAddr{
		Port: 42,
	}, edKey); err == nil {
		t.Fatal("no error for revoked key")
	} else if !reflect.DeepEqual(want, err) {
		t.Fatalf("got %#v, want %#v", want, err)
	}
}

func TestHostAuthority(t *testing.T) {
	for _, m := range []struct {
		authorityFor string
		address      string

		good bool
	}{
		{authorityFor: "localhost", address: "localhost:22", good: true},
		{authorityFor: "localhost", address: "localhost", good: false},
		{authorityFor: "localhost", address: "localhost:1234", good: false},
		{authorityFor: "[localhost]:1234", address: "localhost:1234", good: true},
		{authorityFor: "[localhost]:1234", address: "localhost:22", good: false},
		{authorityFor: "[localhost]:1234", address: "localhost", good: false},
	} {
		db := testDB(t, `@cert-authority `+m.authorityFor+` `+edKeyStr)
		if ok := db.IsHostAuthority(db.lines[0].knownKey.Key, m.address); ok != m.good {
			t.Errorf("IsHostAuthority: authority %s, address %s, wanted good = %v, got good = %v",
				m.authorityFor, m.address, m.good, ok)
		}
	}
}

func TestBracket(t *testing.T) {
	db := testDB(t, `[git.eclipse.org]:29418,[198.41.30.196]:29418 `+edKeyStr)

	if err := db.check("git.eclipse.org:29418", &net.TCPAddr{
		IP:   net.IP{198, 41, 30, 196},
		Port: 29418,
	}, edKey); err != nil {
		t.Errorf("got error %v, want none", err)
	}

	if err := db.check("git.eclipse.org:29419", &net.TCPAddr{
		Port: 42,
	}, edKey); err == nil {
		t.Fatalf("no error for unknown address")
	} else if ke, ok := err.(*KeyError); !ok {
		t.Fatalf("got type %T, want *KeyError", err)
	} else if len(ke.Want) > 0 {
		t.Fatalf("got Want %v, want []", ke.Want)
	}
}

func TestNewKeyType(t *testing.T) {
	str := fmt.Sprintf("%s %s", testAddr, edKeyStr)
	db := testDB(t, str)
	if err := db.check("", testAddr, ecKey); err == nil {
		t.Fatalf("no error for unknown address")
	} else if ke, ok := err.(*KeyError); !ok {
		t.Fatalf("got type %T, want *KeyError", err)
	} else if len(ke.Want) == 0 {
		t.Fatalf("got empty KeyError.Want")
	}
}

func TestSameKeyType(t *testing.T) {
	str := fmt.Sprintf("%s %s", testAddr, edKeyStr)
	db := testDB(t, str)
	if err := db.check("", testAddr, alternateEdKey); err == nil {
		t.Fatalf("no error for unknown address")
	} else if ke, ok := err.(*KeyError); !ok {
		t.Fatalf("got type %T, want *KeyError", err)
	} else if len(ke.Want) == 0 {
		t.Fatalf("got empty KeyError.Want")
	} else if got, want := ke.Want[0].Key.Marshal(), edKey.Marshal(); !bytes.Equal(got, want) {
		t.Fatalf("got key %q, want %q", got, want)
	}
}

func TestIPAddress(t *testing.T) {
	str := fmt.Sprintf("%s %s", testAddr, edKeyStr)
	db := testDB(t, str)
	if err := db.check("", testAddr, edKey); err != nil {
		t.Errorf("got error %q, want none", err)
	}
}

func TestIPv6Address(t *testing.T) {
	str := fmt.Sprintf("%s %s", testAddr6, edKeyStr)
	db := testDB(t, str)

	if err := db.check("", testAddr6, edKey); err != nil {
		t.Errorf("got error %q, want none", err)
	}
}

func TestBasic(t *testing.T) {
	str := fmt.Sprintf("#comment\n\nserver.org,%s %s\notherhost %s", testAddr, edKeyStr, ecKeyStr)
	db := testDB(t, str)
	if err := db.check("server.org:22", testAddr, edKey); err != nil {
		t.Errorf("got error %v, want none", err)
	}

	want := KnownKey{
		Key:      edKey,
		Filename: "testdb",
		Line:     3,
	}
	if err := db.check("server.org:22", testAddr, ecKey); err == nil {
		t.Errorf("succeeded, want KeyError")
	} else if ke, ok := err.(*KeyError); !ok {
		t.Errorf("got %T, want *KeyError", err)
	} else if len(ke.Want) != 1 {
		t.Errorf("got %v, want 1 entry", ke)
	} else if !reflect.DeepEqual(ke.Want[0], want) {
		t.Errorf("got %v, want %v", ke.Want[0], want)
	}
}

func TestHostNamePrecedence(t *testing.T) {
	var evilAddr = &net.TCPAddr{
		IP:   net.IP{66, 66, 66, 66},
		Port: 22,
	}

	str := fmt.Sprintf("server.org,%s %s\nevil.org,%s %s", testAddr, edKeyStr, evilAddr, ecKeyStr)
	db := testDB(t, str)

	if err := db.check("server.org:22", evilAddr, ecKey); err == nil {
		t.Errorf("check succeeded")
	} else if _, ok := err.(*KeyError); !ok {
		t.Errorf("got %T, want *KeyError", err)
	}
}

func TestDBOrderingPrecedenceKeyType(t *testing.T) {
	str := fmt.Sprintf("server.org,%s %s\nserver.org,%s %s", testAddr, edKeyStr, testAddr, alternateEdKeyStr)
	db := testDB(t, str)

	if err := db.check("server.org:22", testAddr, alternateEdKey); err == nil {
		t.Errorf("check succeeded")
	} else if _, ok := err.(*KeyError); !ok {
		t.Errorf("got %T, want *KeyError", err)
	}
}

func TestNegate(t *testing.T) {
	str := fmt.Sprintf("%s,!server.org %s", testAddr, edKeyStr)
	db := testDB(t, str)
	if err := db.check("server.org:22", testAddr, ecKey); err == nil {
		t.Errorf("succeeded")
	} else if ke, ok := err.(*KeyError); !ok {
		t.Errorf("got error type %T, want *KeyError", err)
	} else if len(ke.Want) != 0 {
		t.Errorf("got expected keys %d (first of type %s), want []", len(ke.Want), ke.Want[0].Key.Type())
	}
}

func TestWildcard(t *testing.T) {
	str := fmt.Sprintf("server*.domain %s", edKeyStr)
	db := testDB(t, str)

	want := &KeyError{
		Want: []KnownKey{{
			Filename: "testdb",
			Line:     1,
			Key:      edKey,
		}},
	}

	got := db.check("server.domain:22", &net.TCPAddr{}, ecKey)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestLine(t *testing.T) {
	for in, want := range map[string]string{
		"server.org":                             "server.org " + edKeyStr,
		"server.org:22":                          "server.org " + edKeyStr,
		"server.org:23":                          "[server.org]:23 " + edKeyStr,
		"[c629:1ec4:102:304:102:304:102:304]:22": "[c629:1ec4:102:304:102:304:102:304] " + edKeyStr,
		"[c629:1ec4:102:304:102:304:102:304]:23": "[c629:1ec4:102:304:102:304:102:304]:23 " + edKeyStr,
	} {
		if got := Line([]string{in}, edKey); got != want {
			t.Errorf("Line(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWildcardMatch(t *testing.T) {
	for _, c := range []struct {
		pat, str string
		want     bool
	}{
		{"a?b", "abb", true},
		{"ab", "abc", false},
		{"abc", "ab", false},
		{"a*b", "axxxb", true},
		{"a*b", "axbxb", true},
		{"a*b", "axbxbc", false},
		{"a*?", "axbxc", true},
		{"a*b*", "axxbxxxxxx", true},
		{"a*b*c", "axxbxxxxxxc", true},
		{"a*b*?", "axxbxxxxxxc", true},
		{"a*b*z", "axxbxxbxxxz", true},
		{"a*b*z", "axxbxxzxxxz", true},
		{"a*b*z", "axxbxxzxxx", false},
	} {
		got := wildcardMatch([]byte(c.pat), []byte(c.str))
		if got != c.want {
			t.Errorf("wildcardMatch(%q, %q) = %v, want %v", c.pat, c.str, got, c.want)
		}

	}
}

// TODO(hanwen): test coverage for certificates.

const testHostname = "hostname"

// generated with keygen -H -f
const encodedTestHostnameHash = "|1|IHXZvQMvTcZTUU29+2vXFgx8Frs=|UGccIWfRVDwilMBnA3WJoRAC75Y="

func TestHostHash(t *testing.T) {
	testHostHash(t, testHostname, encodedTestHostnameHash)
}

func TestHashList(t *testing.T) {
	encoded := HashHostname(testHostname)
	testHostHash(t, testHostname, encoded)
}

func testHostHash(t *testing.T, hostname, encoded string) {
	typ, salt, hash, err := decodeHash(encoded)
	if err != nil {
		t.Fatalf("decodeHash: %v", err)
	}

	if got := encodeHash(typ, salt, hash); got != encoded {
		t.Errorf("got encoding %s want %s", got, encoded)
	}

	if typ != sha1HashType {
		t.Fatalf("got hash type %q, want %q", typ, sha1HashType)
	}

	got := hashHost(hostname, salt)
	if !bytes.Equal(got, hash) {
		t.Errorf("got hash %x want %x", got, hash)
	}
}

func TestNormalize(t *testing.T) {
	for in, want := range map[string]string{
		"127.0.0.1:22":             "127.0.0.1",
		"[127.0.0.1]:22":           "127.0.0.1",
		"[127.0.0.1]:23":           "[127.0.0.1]:23",
		"127.0.0.1:23":             "[127.0.0.1]:23",
		"[a.b.c]:22":               "a.b.c",
		"[abcd:abcd:abcd:abcd]":    "[abcd:abcd:abcd:abcd]",
		"[abcd:abcd:abcd:abcd]:22": "[abcd:abcd:abcd:abcd]",
		"[abcd:abcd:abcd:abcd]:23": "[abcd:abcd:abcd:abcd]:23",
	} {
		got := Normalize(in)
		if got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHashedHostkeyCheck(t *testing.T) {
	str := fmt.Sprintf("%s %s", HashHostname(testHostname), edKeyStr)
	db := testDB(t, str)
	if err := db.check(testHostname+":22", testAddr, edKey); err != nil {
		t.Errorf("check(%s): %v", testHostname, err)
	}
	want := &KeyError{
		Want: []KnownKey{{
			Filename: "testdb",
			Line:     1,
			Key:      edKey,
		}},
	}
	if got := db.check(testHostname+":22", testAddr, alternateEdKey); !reflect.DeepEqual(got, want) {
		t.Errorf("got error %v, want %v", got, want)
	}
}
