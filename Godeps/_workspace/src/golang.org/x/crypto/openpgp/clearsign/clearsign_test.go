// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clearsign

import (
	"bytes"
	"golang.org/x/crypto/openpgp"
	"testing"
)

func testParse(t *testing.T, input []byte, expected, expectedPlaintext string) {
	b, rest := Decode(input)
	if b == nil {
		t.Fatal("failed to decode clearsign message")
	}
	if !bytes.Equal(rest, []byte("trailing")) {
		t.Errorf("unexpected remaining bytes returned: %s", string(rest))
	}
	if b.ArmoredSignature.Type != "PGP SIGNATURE" {
		t.Errorf("bad armor type, got:%s, want:PGP SIGNATURE", b.ArmoredSignature.Type)
	}
	if !bytes.Equal(b.Bytes, []byte(expected)) {
		t.Errorf("bad body, got:%x want:%x", b.Bytes, expected)
	}

	if !bytes.Equal(b.Plaintext, []byte(expectedPlaintext)) {
		t.Errorf("bad plaintext, got:%x want:%x", b.Plaintext, expectedPlaintext)
	}

	keyring, err := openpgp.ReadArmoredKeyRing(bytes.NewBufferString(signingKey))
	if err != nil {
		t.Errorf("failed to parse public key: %s", err)
	}

	if _, err := openpgp.CheckDetachedSignature(keyring, bytes.NewBuffer(b.Bytes), b.ArmoredSignature.Body); err != nil {
		t.Errorf("failed to check signature: %s", err)
	}
}

func TestParse(t *testing.T) {
	testParse(t, clearsignInput, "Hello world\r\nline 2", "Hello world\nline 2\n")
	testParse(t, clearsignInput2, "\r\n\r\n(This message has a couple of blank lines at the start and end.)\r\n\r\n", "\n\n(This message has a couple of blank lines at the start and end.)\n\n\n")
}

func TestParseWithNoNewlineAtEnd(t *testing.T) {
	input := clearsignInput
	input = input[:len(input)-len("trailing")-1]
	b, rest := Decode(input)
	if b == nil {
		t.Fatal("failed to decode clearsign message")
	}
	if len(rest) > 0 {
		t.Errorf("unexpected remaining bytes returned: %s", string(rest))
	}
}

var signingTests = []struct {
	in, signed, plaintext string
}{
	{"", "", ""},
	{"a", "a", "a\n"},
	{"a\n", "a", "a\n"},
	{"-a\n", "-a", "-a\n"},
	{"--a\nb", "--a\r\nb", "--a\nb\n"},
}

func TestSigning(t *testing.T) {
	keyring, err := openpgp.ReadArmoredKeyRing(bytes.NewBufferString(signingKey))
	if err != nil {
		t.Errorf("failed to parse public key: %s", err)
	}

	for i, test := range signingTests {
		var buf bytes.Buffer

		plaintext, err := Encode(&buf, keyring[0].PrivateKey, nil)
		if err != nil {
			t.Errorf("#%d: error from Encode: %s", i, err)
			continue
		}
		if _, err := plaintext.Write([]byte(test.in)); err != nil {
			t.Errorf("#%d: error from Write: %s", i, err)
			continue
		}
		if err := plaintext.Close(); err != nil {
			t.Fatalf("#%d: error from Close: %s", i, err)
			continue
		}

		b, _ := Decode(buf.Bytes())
		if b == nil {
			t.Errorf("#%d: failed to decode clearsign message", i)
			continue
		}
		if !bytes.Equal(b.Bytes, []byte(test.signed)) {
			t.Errorf("#%d: bad result, got:%x, want:%x", i, b.Bytes, test.signed)
			continue
		}
		if !bytes.Equal(b.Plaintext, []byte(test.plaintext)) {
			t.Errorf("#%d: bad result, got:%x, want:%x", i, b.Plaintext, test.plaintext)
			continue
		}

		if _, err := openpgp.CheckDetachedSignature(keyring, bytes.NewBuffer(b.Bytes), b.ArmoredSignature.Body); err != nil {
			t.Errorf("#%d: failed to check signature: %s", i, err)
		}
	}
}

var clearsignInput = []byte(`
;lasjlkfdsa

-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA1

Hello world
line 2
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v1.4.10 (GNU/Linux)

iJwEAQECAAYFAk8kMuEACgkQO9o98PRieSpMsAQAhmY/vwmNpflrPgmfWsYhk5O8
pjnBUzZwqTDoDeINjZEoPDSpQAHGhjFjgaDx/Gj4fAl0dM4D0wuUEBb6QOrwflog
2A2k9kfSOMOtk0IH/H5VuFN1Mie9L/erYXjTQIptv9t9J7NoRBMU0QOOaFU0JaO9
MyTpno24AjIAGb+mH1U=
=hIJ6
-----END PGP SIGNATURE-----
trailing`)

var clearsignInput2 = []byte(`
asdlfkjasdlkfjsadf

-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256



(This message has a couple of blank lines at the start and end.)


-----BEGIN PGP SIGNATURE-----
Version: GnuPG v1.4.11 (GNU/Linux)

iJwEAQEIAAYFAlPpSREACgkQO9o98PRieSpZTAP+M8QUoCt/7Rf3YbXPcdzIL32v
pt1I+cMNeopzfLy0u4ioEFi8s5VkwpL1AFmirvgViCwlf82inoRxzZRiW05JQ5LI
ESEzeCoy2LIdRCQ2hcrG8pIUPzUO4TqO5D/dMbdHwNH4h5nNmGJUAEG6FpURlPm+
qZg6BaTvOxepqOxnhVU=
=e+C6
-----END PGP SIGNATURE-----

trailing`)

var signingKey = `-----BEGIN PGP PRIVATE KEY BLOCK-----
Version: GnuPG v1.4.10 (GNU/Linux)

lQHYBE2rFNoBBADFwqWQIW/DSqcB4yCQqnAFTJ27qS5AnB46ccAdw3u4Greeu3Bp
idpoHdjULy7zSKlwR1EA873dO/k/e11Ml3dlAFUinWeejWaK2ugFP6JjiieSsrKn
vWNicdCS4HTWn0X4sjl0ZiAygw6GNhqEQ3cpLeL0g8E9hnYzJKQ0LWJa0QARAQAB
AAP/TB81EIo2VYNmTq0pK1ZXwUpxCrvAAIG3hwKjEzHcbQznsjNvPUihZ+NZQ6+X
0HCfPAdPkGDCLCb6NavcSW+iNnLTrdDnSI6+3BbIONqWWdRDYJhqZCkqmG6zqSfL
IdkJgCw94taUg5BWP/AAeQrhzjChvpMQTVKQL5mnuZbUCeMCAN5qrYMP2S9iKdnk
VANIFj7656ARKt/nf4CBzxcpHTyB8+d2CtPDKCmlJP6vL8t58Jmih+kHJMvC0dzn
gr5f5+sCAOOe5gt9e0am7AvQWhdbHVfJU0TQJx+m2OiCJAqGTB1nvtBLHdJnfdC9
TnXXQ6ZXibqLyBies/xeY2sCKL5qtTMCAKnX9+9d/5yQxRyrQUHt1NYhaXZnJbHx
q4ytu0eWz+5i68IYUSK69jJ1NWPM0T6SkqpB3KCAIv68VFm9PxqG1KmhSrQIVGVz
dCBLZXmIuAQTAQIAIgUCTasU2gIbAwYLCQgHAwIGFQgCCQoLBBYCAwECHgECF4AA
CgkQO9o98PRieSoLhgQAkLEZex02Qt7vGhZzMwuN0R22w3VwyYyjBx+fM3JFETy1
ut4xcLJoJfIaF5ZS38UplgakHG0FQ+b49i8dMij0aZmDqGxrew1m4kBfjXw9B/v+
eIqpODryb6cOSwyQFH0lQkXC040pjq9YqDsO5w0WYNXYKDnzRV0p4H1pweo2VDid
AdgETasU2gEEAN46UPeWRqKHvA99arOxee38fBt2CI08iiWyI8T3J6ivtFGixSqV
bRcPxYO/qLpVe5l84Nb3X71GfVXlc9hyv7CD6tcowL59hg1E/DC5ydI8K8iEpUmK
/UnHdIY5h8/kqgGxkY/T/hgp5fRQgW1ZoZxLajVlMRZ8W4tFtT0DeA+JABEBAAEA
A/0bE1jaaZKj6ndqcw86jd+QtD1SF+Cf21CWRNeLKnUds4FRRvclzTyUMuWPkUeX
TaNNsUOFqBsf6QQ2oHUBBK4VCHffHCW4ZEX2cd6umz7mpHW6XzN4DECEzOVksXtc
lUC1j4UB91DC/RNQqwX1IV2QLSwssVotPMPqhOi0ZLNY7wIA3n7DWKInxYZZ4K+6
rQ+POsz6brEoRHwr8x6XlHenq1Oki855pSa1yXIARoTrSJkBtn5oI+f8AzrnN0BN
oyeQAwIA/7E++3HDi5aweWrViiul9cd3rcsS0dEnksPhvS0ozCJiHsq/6GFmy7J8
QSHZPteedBnZyNp5jR+H7cIfVN3KgwH/Skq4PsuPhDq5TKK6i8Pc1WW8MA6DXTdU
nLkX7RGmMwjC0DBf7KWAlPjFaONAX3a8ndnz//fy1q7u2l9AZwrj1qa1iJ8EGAEC
AAkFAk2rFNoCGwwACgkQO9o98PRieSo2/QP/WTzr4ioINVsvN1akKuekmEMI3LAp
BfHwatufxxP1U+3Si/6YIk7kuPB9Hs+pRqCXzbvPRrI8NHZBmc8qIGthishdCYad
AHcVnXjtxrULkQFGbGvhKURLvS9WnzD/m1K2zzwxzkPTzT9/Yf06O6Mal5AdugPL
VrM0m72/jnpKo04=
=zNCn
-----END PGP PRIVATE KEY BLOCK-----
`
