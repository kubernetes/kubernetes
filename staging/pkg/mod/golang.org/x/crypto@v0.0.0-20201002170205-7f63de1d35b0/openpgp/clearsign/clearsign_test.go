// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clearsign

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"golang.org/x/crypto/openpgp"
	"golang.org/x/crypto/openpgp/packet"
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
	// leading whitespace
	{" a\n", " a", " a\n"},
	{"  a\n", "  a", "  a\n"},
	// trailing whitespace (should be stripped)
	{"a \n", "a", "a\n"},
	{"a ", "a", "a\n"},
	// whitespace-only lines (should be stripped)
	{"  \n", "", "\n"},
	{"  ", "", "\n"},
	{"a\n  \n  \nb\n", "a\r\n\r\n\r\nb", "a\n\n\nb\n"},
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

// We use this to make test keys, so that they aren't all the same.
type quickRand byte

func (qr *quickRand) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(*qr)
	}
	*qr++
	return len(p), nil
}

func TestMultiSign(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long test in -short mode")
	}

	zero := quickRand(0)
	config := packet.Config{Rand: &zero}

	for nKeys := 0; nKeys < 4; nKeys++ {
	nextTest:
		for nExtra := 0; nExtra < 4; nExtra++ {
			var signKeys []*packet.PrivateKey
			var verifyKeys openpgp.EntityList

			desc := fmt.Sprintf("%d keys; %d of which will be used to verify", nKeys+nExtra, nKeys)
			for i := 0; i < nKeys+nExtra; i++ {
				e, err := openpgp.NewEntity("name", "comment", "email", &config)
				if err != nil {
					t.Errorf("cannot create key: %v", err)
					continue nextTest
				}
				if i < nKeys {
					verifyKeys = append(verifyKeys, e)
				}
				signKeys = append(signKeys, e.PrivateKey)
			}

			input := []byte("this is random text\r\n4 17")
			var output bytes.Buffer
			w, err := EncodeMulti(&output, signKeys, nil)
			if err != nil {
				t.Errorf("EncodeMulti (%s) failed: %v", desc, err)
			}
			if _, err := w.Write(input); err != nil {
				t.Errorf("Write(%q) to signer (%s) failed: %v", string(input), desc, err)
			}
			if err := w.Close(); err != nil {
				t.Errorf("Close() of signer (%s) failed: %v", desc, err)
			}

			block, _ := Decode(output.Bytes())
			if string(block.Bytes) != string(input) {
				t.Errorf("Inline data didn't match original; got %q want %q", string(block.Bytes), string(input))
			}
			_, err = openpgp.CheckDetachedSignature(verifyKeys, bytes.NewReader(block.Bytes), block.ArmoredSignature.Body)
			if nKeys == 0 {
				if err == nil {
					t.Errorf("verifying inline (%s) succeeded; want failure", desc)
				}
			} else {
				if err != nil {
					t.Errorf("verifying inline (%s) failed (%v); want success", desc, err)
				}
			}
		}
	}
}

func TestDecodeMissingCRC(t *testing.T) {
	block, rest := Decode(clearsignInput3)
	if block == nil {
		t.Fatal("failed to decode PGP signature missing a CRC")
	}
	if len(rest) > 0 {
		t.Fatalf("Decode should not have any remaining data left: %s", rest)
	}
	if _, err := packet.Read(block.ArmoredSignature.Body); err != nil {
		t.Error(err)
	}
	if _, err := packet.Read(block.ArmoredSignature.Body); err != io.EOF {
		t.Error(err)
	}
}

const signatureBlock = `
-----BEGIN PGP SIGNATURE-----
Version: OpenPrivacy 0.99

yDgBO22WxBHv7O8X7O/jygAEzol56iUKiXmV+XmpCtmpqQUKiQrFqclFqUDBovzS
vBSFjNSiVHsuAA==
=njUN
-----END PGP SIGNATURE-----
`

var invalidInputs = []string{
	`
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

(This message was truncated.)
`,
	`
-----BEGIN PGP SIGNED MESSAGE-----garbage
Hash: SHA256

_o/
` + signatureBlock,
	`
garbage-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

_o/
` + signatureBlock,
	`
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA` + "\x0b\x0b" + `256

_o/
` + signatureBlock,
	`
-----BEGIN PGP SIGNED MESSAGE-----
NotHash: SHA256

_o/
` + signatureBlock,
}

func TestParseInvalid(t *testing.T) {
	for i, input := range invalidInputs {
		if b, rest := Decode([]byte(input)); b != nil {
			t.Errorf("#%d: decoded a bad clearsigned message without any error", i)
		} else if string(rest) != input {
			t.Errorf("#%d: did not return all data with a bad message", i)
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

var clearsignInput3 = []byte(`-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

Origin: vscode stable
Label: vscode stable
Suite: stable
Codename: stable
Date: Mon, 13 Jan 2020 08:41:45 UTC
Architectures: amd64
Components: main
Description: Generated by aptly
MD5Sum:
 66437152b3082616d8053e52c4bafafb  5821166 Contents-amd64
 8024662ed51109946a517754bbafdd33   286298 Contents-amd64.gz
 66437152b3082616d8053e52c4bafafb  5821166 main/Contents-amd64
 8024662ed51109946a517754bbafdd33   286298 main/Contents-amd64.gz
 3062a08b3eca94a65d6d17ba1dafcf3e  1088265 main/binary-amd64/Packages
 b8ee22200fba8fa3be56c1ff946cdd24   159344 main/binary-amd64/Packages.bz2
 f89c47c81ebd25caf287c8e6dda16c1a   169456 main/binary-amd64/Packages.gz
 4c9ca25b556f111a5536c78df885ad82       95 main/binary-amd64/Release
SHA1:
 2b62d0e322746b7d094878278f49993ca4314bf7  5821166 Contents-amd64
 aafe35cce12e03d8b1939e403ddf5c0958c6e9bd   286298 Contents-amd64.gz
 2b62d0e322746b7d094878278f49993ca4314bf7  5821166 main/Contents-amd64
 aafe35cce12e03d8b1939e403ddf5c0958c6e9bd   286298 main/Contents-amd64.gz
 30316ac5d4ce3b472a96a797eeb0a2a82d43ed3e  1088265 main/binary-amd64/Packages
 6507e0b4da8194fd1048fcbb74c6e7433edaf3d6   159344 main/binary-amd64/Packages.bz2
 ec9d39c39567c74001221e4900fb5d11ec11b833   169456 main/binary-amd64/Packages.gz
 58bf20987a91d35936f18efce75ea233d43dbf8b       95 main/binary-amd64/Release
SHA256:
 deff9ebfc44bf482e10a6ea10f608c6bb0fdc8373bf86b88cad9d99879ae3c39  5821166 Contents-amd64
 f163bc65c7666ef58e0be3336e8c846ae2b7b388fbb2d7db0bcdc3fd1abae462   286298 Contents-amd64.gz
 deff9ebfc44bf482e10a6ea10f608c6bb0fdc8373bf86b88cad9d99879ae3c39  5821166 main/Contents-amd64
 f163bc65c7666ef58e0be3336e8c846ae2b7b388fbb2d7db0bcdc3fd1abae462   286298 main/Contents-amd64.gz
 0fba50799ef72d0c2b354d0bcbbc8c623f6dae5a7fd7c218a54ea44dd8a49d5e  1088265 main/binary-amd64/Packages
 69382470a88b67acde80fe45ab223016adebc445713ff0aa3272902581d21f13   159344 main/binary-amd64/Packages.bz2
 1724b8ace5bd8882943e9463d8525006f33ca704480da0186fd47937451dc216   169456 main/binary-amd64/Packages.gz
 0f509a0cb07e0ab433176fa47a21dccccc6b519f25f640cc58561104c11de6c2       95 main/binary-amd64/Release
SHA512:
 f69f09c6180ceb6625a84b5f7123ad27972983146979dcfd9c38b2990459b52b4975716f85374511486bb5ad5852ebb1ef8265176df7134fc15b17ada3ba596c  5821166 Contents-amd64
 46031bf89166188989368957d20cdcaac6eec72bab3f9839c9704bb08cbee3174ca6da11e290b0eab0e6b5754c1e7feb06d18ec9c5a0c955029cef53235e0a3a   286298 Contents-amd64.gz
 f69f09c6180ceb6625a84b5f7123ad27972983146979dcfd9c38b2990459b52b4975716f85374511486bb5ad5852ebb1ef8265176df7134fc15b17ada3ba596c  5821166 main/Contents-amd64
 46031bf89166188989368957d20cdcaac6eec72bab3f9839c9704bb08cbee3174ca6da11e290b0eab0e6b5754c1e7feb06d18ec9c5a0c955029cef53235e0a3a   286298 main/Contents-amd64.gz
 3f78baf5adbaf0100996555b154807c794622fd0b5879b568ae0b6560e988fbfabed8d97db5a703d1a58514b9690fc6b60f9ad2eeece473d86ab257becd0ae41  1088265 main/binary-amd64/Packages
 18f26df90beff29192662ca40525367c3c04f4581d59d2e9ab1cd0700a145b6a292a1609ca33ebe1c211f13718a8eee751f41fd8189cf93d52aa3e0851542dfc   159344 main/binary-amd64/Packages.bz2
 6a6d917229e0cf06c493e174a87d76e815717676f2c70bcbd3bc689a80bd3c5489ea97db83b8f74cba8e70f374f9d9974f22b1ed2687a4ba1dacd22fdef7e14d   169456 main/binary-amd64/Packages.gz
 e1a4378ad266c13c2edf8a0e590fa4d11973ab99ce79f15af005cb838f1600f66f3dc6da8976fa8b474da9073c118039c27623ab3360c6df115071497fe4f50c       95 main/binary-amd64/Release

-----BEGIN PGP SIGNATURE-----
Version: BSN Pgp v1.0.0.0

iQEcBAEBCAAGBQJeHC1bAAoJEOs+lK2+EinPAg8H/1rrhcgfm1HYL+Vmr9Ns6ton
LWQ8r13ADN66UTRa3XsO9V+q1fYowTqpXq6EZt2Gmlby/cpDf7mFPM5IteOXWLl7
QcWxPKHcdPIUi+h5F7BkFW65imP9GyX+V5Pxx5X544op7hYKaI0gAQ1oYtWDb3HE
4D27fju6icbj8w6E8TePcrDn82UvWAcaI5WSLboyhXCt2DxS3PNGFlyaP58zKJ8F
9cbBzksuMgMaTPAAMrU0zrFGfGeQz0Yo6nV/gRGiQaL9pSeIJWSKLNCMG/nIGmv2
xHVNFqTEetREY6UcQmuhwOn4HezyigH6XCBVp/Uez1izXiNdwBOet34SSvnkuJ4=
-----END PGP SIGNATURE-----`)

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
