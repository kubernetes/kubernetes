// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"bytes"
	"crypto/tls"
	"encoding/binary"
	"fmt"
	"testing"
	"time"
)

var testDSNs = []struct {
	in  string
	out string
	loc *time.Location
}{
	{"username:password@protocol(address)/dbname?param=value", "&{user:username passwd:password net:protocol addr:address dbname:dbname params:map[param:value] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"username:password@protocol(address)/dbname?param=value&columnsWithAlias=true", "&{user:username passwd:password net:protocol addr:address dbname:dbname params:map[param:value] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:true interpolateParams:false}", time.UTC},
	{"user@unix(/path/to/socket)/dbname?charset=utf8", "&{user:user passwd: net:unix addr:/path/to/socket dbname:dbname params:map[charset:utf8] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"user:password@tcp(localhost:5555)/dbname?charset=utf8&tls=true", "&{user:user passwd:password net:tcp addr:localhost:5555 dbname:dbname params:map[charset:utf8] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"user:password@tcp(localhost:5555)/dbname?charset=utf8mb4,utf8&tls=skip-verify", "&{user:user passwd:password net:tcp addr:localhost:5555 dbname:dbname params:map[charset:utf8mb4,utf8] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"user:password@/dbname?loc=UTC&timeout=30s&allowAllFiles=1&clientFoundRows=true&allowOldPasswords=TRUE&collation=utf8mb4_unicode_ci", "&{user:user passwd:password net:tcp addr:127.0.0.1:3306 dbname:dbname params:map[] loc:%p tls:<nil> timeout:30000000000 collation:224 allowAllFiles:true allowOldPasswords:true allowCleartextPasswords:false clientFoundRows:true columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"user:p@ss(word)@tcp([de:ad:be:ef::ca:fe]:80)/dbname?loc=Local", "&{user:user passwd:p@ss(word) net:tcp addr:[de:ad:be:ef::ca:fe]:80 dbname:dbname params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.Local},
	{"/dbname", "&{user: passwd: net:tcp addr:127.0.0.1:3306 dbname:dbname params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"@/", "&{user: passwd: net:tcp addr:127.0.0.1:3306 dbname: params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"/", "&{user: passwd: net:tcp addr:127.0.0.1:3306 dbname: params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"", "&{user: passwd: net:tcp addr:127.0.0.1:3306 dbname: params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"user:p@/ssword@/", "&{user:user passwd:p@/ssword net:tcp addr:127.0.0.1:3306 dbname: params:map[] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
	{"unix/?arg=%2Fsome%2Fpath.ext", "&{user: passwd: net:unix addr:/tmp/mysql.sock dbname: params:map[arg:/some/path.ext] loc:%p tls:<nil> timeout:0 collation:33 allowAllFiles:false allowOldPasswords:false allowCleartextPasswords:false clientFoundRows:false columnsWithAlias:false interpolateParams:false}", time.UTC},
}

func TestDSNParser(t *testing.T) {
	var cfg *config
	var err error
	var res string

	for i, tst := range testDSNs {
		cfg, err = parseDSN(tst.in)
		if err != nil {
			t.Error(err.Error())
		}

		// pointer not static
		cfg.tls = nil

		res = fmt.Sprintf("%+v", cfg)
		if res != fmt.Sprintf(tst.out, tst.loc) {
			t.Errorf("%d. parseDSN(%q) => %q, want %q", i, tst.in, res, fmt.Sprintf(tst.out, tst.loc))
		}
	}
}

func TestDSNParserInvalid(t *testing.T) {
	var invalidDSNs = []string{
		"@net(addr/",                  // no closing brace
		"@tcp(/",                      // no closing brace
		"tcp(/",                       // no closing brace
		"(/",                          // no closing brace
		"net(addr)//",                 // unescaped
		"user:pass@tcp(1.2.3.4:3306)", // no trailing slash
		//"/dbname?arg=/some/unescaped/path",
	}

	for i, tst := range invalidDSNs {
		if _, err := parseDSN(tst); err == nil {
			t.Errorf("invalid DSN #%d. (%s) didn't error!", i, tst)
		}
	}
}

func TestDSNWithCustomTLS(t *testing.T) {
	baseDSN := "user:password@tcp(localhost:5555)/dbname?tls="
	tlsCfg := tls.Config{}

	RegisterTLSConfig("utils_test", &tlsCfg)

	// Custom TLS is missing
	tst := baseDSN + "invalid_tls"
	cfg, err := parseDSN(tst)
	if err == nil {
		t.Errorf("Invalid custom TLS in DSN (%s) but did not error.  Got config: %#v", tst, cfg)
	}

	tst = baseDSN + "utils_test"

	// Custom TLS with a server name
	name := "foohost"
	tlsCfg.ServerName = name
	cfg, err = parseDSN(tst)

	if err != nil {
		t.Error(err.Error())
	} else if cfg.tls.ServerName != name {
		t.Errorf("Did not get the correct TLS ServerName (%s) parsing DSN (%s).", name, tst)
	}

	// Custom TLS without a server name
	name = "localhost"
	tlsCfg.ServerName = ""
	cfg, err = parseDSN(tst)

	if err != nil {
		t.Error(err.Error())
	} else if cfg.tls.ServerName != name {
		t.Errorf("Did not get the correct ServerName (%s) parsing DSN (%s).", name, tst)
	}

	DeregisterTLSConfig("utils_test")
}

func TestDSNUnsafeCollation(t *testing.T) {
	_, err := parseDSN("/dbname?collation=gbk_chinese_ci&interpolateParams=true")
	if err != errInvalidDSNUnsafeCollation {
		t.Error("Expected %v, Got %v", errInvalidDSNUnsafeCollation, err)
	}

	_, err = parseDSN("/dbname?collation=gbk_chinese_ci&interpolateParams=false")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}

	_, err = parseDSN("/dbname?collation=gbk_chinese_ci")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}

	_, err = parseDSN("/dbname?collation=ascii_bin&interpolateParams=true")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}

	_, err = parseDSN("/dbname?collation=latin1_german1_ci&interpolateParams=true")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}

	_, err = parseDSN("/dbname?collation=utf8_general_ci&interpolateParams=true")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}

	_, err = parseDSN("/dbname?collation=utf8mb4_general_ci&interpolateParams=true")
	if err != nil {
		t.Error("Expected %v, Got %v", nil, err)
	}
}

func BenchmarkParseDSN(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		for _, tst := range testDSNs {
			if _, err := parseDSN(tst.in); err != nil {
				b.Error(err.Error())
			}
		}
	}
}

func TestScanNullTime(t *testing.T) {
	var scanTests = []struct {
		in    interface{}
		error bool
		valid bool
		time  time.Time
	}{
		{tDate, false, true, tDate},
		{sDate, false, true, tDate},
		{[]byte(sDate), false, true, tDate},
		{tDateTime, false, true, tDateTime},
		{sDateTime, false, true, tDateTime},
		{[]byte(sDateTime), false, true, tDateTime},
		{tDate0, false, true, tDate0},
		{sDate0, false, true, tDate0},
		{[]byte(sDate0), false, true, tDate0},
		{sDateTime0, false, true, tDate0},
		{[]byte(sDateTime0), false, true, tDate0},
		{"", true, false, tDate0},
		{"1234", true, false, tDate0},
		{0, true, false, tDate0},
	}

	var nt = NullTime{}
	var err error

	for _, tst := range scanTests {
		err = nt.Scan(tst.in)
		if (err != nil) != tst.error {
			t.Errorf("%v: expected error status %t, got %t", tst.in, tst.error, (err != nil))
		}
		if nt.Valid != tst.valid {
			t.Errorf("%v: expected valid status %t, got %t", tst.in, tst.valid, nt.Valid)
		}
		if nt.Time != tst.time {
			t.Errorf("%v: expected time %v, got %v", tst.in, tst.time, nt.Time)
		}
	}
}

func TestLengthEncodedInteger(t *testing.T) {
	var integerTests = []struct {
		num     uint64
		encoded []byte
	}{
		{0x0000000000000000, []byte{0x00}},
		{0x0000000000000012, []byte{0x12}},
		{0x00000000000000fa, []byte{0xfa}},
		{0x0000000000000100, []byte{0xfc, 0x00, 0x01}},
		{0x0000000000001234, []byte{0xfc, 0x34, 0x12}},
		{0x000000000000ffff, []byte{0xfc, 0xff, 0xff}},
		{0x0000000000010000, []byte{0xfd, 0x00, 0x00, 0x01}},
		{0x0000000000123456, []byte{0xfd, 0x56, 0x34, 0x12}},
		{0x0000000000ffffff, []byte{0xfd, 0xff, 0xff, 0xff}},
		{0x0000000001000000, []byte{0xfe, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}},
		{0x123456789abcdef0, []byte{0xfe, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12}},
		{0xffffffffffffffff, []byte{0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
	}

	for _, tst := range integerTests {
		num, isNull, numLen := readLengthEncodedInteger(tst.encoded)
		if isNull {
			t.Errorf("%x: expected %d, got NULL", tst.encoded, tst.num)
		}
		if num != tst.num {
			t.Errorf("%x: expected %d, got %d", tst.encoded, tst.num, num)
		}
		if numLen != len(tst.encoded) {
			t.Errorf("%x: expected size %d, got %d", tst.encoded, len(tst.encoded), numLen)
		}
		encoded := appendLengthEncodedInteger(nil, num)
		if !bytes.Equal(encoded, tst.encoded) {
			t.Errorf("%v: expected %x, got %x", num, tst.encoded, encoded)
		}
	}
}

func TestOldPass(t *testing.T) {
	scramble := []byte{9, 8, 7, 6, 5, 4, 3, 2}
	vectors := []struct {
		pass string
		out  string
	}{
		{" pass", "47575c5a435b4251"},
		{"pass ", "47575c5a435b4251"},
		{"123\t456", "575c47505b5b5559"},
		{"C0mpl!ca ted#PASS123", "5d5d554849584a45"},
	}
	for _, tuple := range vectors {
		ours := scrambleOldPassword(scramble, []byte(tuple.pass))
		if tuple.out != fmt.Sprintf("%x", ours) {
			t.Errorf("Failed old password %q", tuple.pass)
		}
	}
}

func TestFormatBinaryDateTime(t *testing.T) {
	rawDate := [11]byte{}
	binary.LittleEndian.PutUint16(rawDate[:2], 1978)   // years
	rawDate[2] = 12                                    // months
	rawDate[3] = 30                                    // days
	rawDate[4] = 15                                    // hours
	rawDate[5] = 46                                    // minutes
	rawDate[6] = 23                                    // seconds
	binary.LittleEndian.PutUint32(rawDate[7:], 987654) // microseconds
	expect := func(expected string, inlen, outlen uint8) {
		actual, _ := formatBinaryDateTime(rawDate[:inlen], outlen, false)
		bytes, ok := actual.([]byte)
		if !ok {
			t.Errorf("formatBinaryDateTime must return []byte, was %T", actual)
		}
		if string(bytes) != expected {
			t.Errorf(
				"expected %q, got %q for length in %d, out %d",
				bytes, actual, inlen, outlen,
			)
		}
	}
	expect("0000-00-00", 0, 10)
	expect("0000-00-00 00:00:00", 0, 19)
	expect("1978-12-30", 4, 10)
	expect("1978-12-30 15:46:23", 7, 19)
	expect("1978-12-30 15:46:23.987654", 11, 26)
}

func TestEscapeBackslash(t *testing.T) {
	expect := func(expected, value string) {
		actual := string(escapeBytesBackslash([]byte{}, []byte(value)))
		if actual != expected {
			t.Errorf(
				"expected %s, got %s",
				expected, actual,
			)
		}

		actual = string(escapeStringBackslash([]byte{}, value))
		if actual != expected {
			t.Errorf(
				"expected %s, got %s",
				expected, actual,
			)
		}
	}

	expect("foo\\0bar", "foo\x00bar")
	expect("foo\\nbar", "foo\nbar")
	expect("foo\\rbar", "foo\rbar")
	expect("foo\\Zbar", "foo\x1abar")
	expect("foo\\\"bar", "foo\"bar")
	expect("foo\\\\bar", "foo\\bar")
	expect("foo\\'bar", "foo'bar")
}

func TestEscapeQuotes(t *testing.T) {
	expect := func(expected, value string) {
		actual := string(escapeBytesQuotes([]byte{}, []byte(value)))
		if actual != expected {
			t.Errorf(
				"expected %s, got %s",
				expected, actual,
			)
		}

		actual = string(escapeStringQuotes([]byte{}, value))
		if actual != expected {
			t.Errorf(
				"expected %s, got %s",
				expected, actual,
			)
		}
	}

	expect("foo\x00bar", "foo\x00bar") // not affected
	expect("foo\nbar", "foo\nbar")     // not affected
	expect("foo\rbar", "foo\rbar")     // not affected
	expect("foo\x1abar", "foo\x1abar") // not affected
	expect("foo''bar", "foo'bar")      // affected
	expect("foo\"bar", "foo\"bar")     // not affected
}
