// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package verification

import (
	"bytes"
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"

	"github.com/coreos/go-tspi/tspiconst"
)

func pad(plaintext []byte, bsize int) ([]byte, error) {
	if bsize >= 256 {
		return nil, errors.New("bsize must be < 256")
	}
	pad := bsize - (len(plaintext) % bsize)
	if pad == 0 {
		pad = bsize
	}
	for i := 0; i < pad; i++ {
		plaintext = append(plaintext, byte(pad))
	}
	return plaintext, nil
}

// GenerateChallenge takes a copy of the EK certificate, the public half of
// the AIK to be challenged and a secret. It then symmetrically encrypts the
// secret with a randomly generated AES key and Asymmetrically encrypts the
// AES key with the public half of the EK. These can then be provided to the
// TPM in order to ensure that the AIK is under the control of the TPM. It
// returns the asymmetrically and symmetrically encrypted data, along with
// any error.
func GenerateChallenge(ekcert []byte, aikpub []byte, secret []byte) (asymenc []byte, symenc []byte, err error) {
	aeskey := make([]byte, 16)
	iv := make([]byte, 16)

	_, err = rand.Read(aeskey)
	if err != nil {
		return nil, nil, err
	}

	_, err = rand.Read(iv)
	if err != nil {
		return nil, nil, err
	}

	/*
	 * The EK certificate has an OID for rsaesOaep which will break
	 * parsing. Replace it with rsaEncryption instead.
	 */
	ekcert = bytes.Replace(ekcert, []byte{0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x07}, []byte{0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01}, 1)
	cert, err := x509.ParseCertificate(ekcert)
	pubkey := cert.PublicKey.(*rsa.PublicKey)

	asymplain := []byte{0x00, 0x00, 0x00, 0x06, 0x00, 0xff, 0x00, 0x10}
	asymplain = append(asymplain, aeskey...)
	hash := sha1.Sum(aikpub)
	asymplain = append(asymplain, hash[:]...)

	label := []byte{'T', 'C', 'P', 'A'}
	asymenc, err = rsa.EncryptOAEP(sha1.New(), rand.Reader, pubkey, asymplain, label)
	block, err := aes.NewCipher(aeskey)
	if err != nil {
		return nil, nil, err
	}
	cbc := cipher.NewCBCEncrypter(block, iv)
	secret, err = pad(secret, len(iv))
	if err != nil {
		return nil, nil, err
	}
	symenc = make([]byte, len(secret))
	cbc.CryptBlocks(symenc, secret)

	symheader := new(bytes.Buffer)
	err = binary.Write(symheader, binary.BigEndian, (uint32)(len(symenc)+len(iv)))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint32)(tspiconst.TPM_ALG_AES))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint16)(tspiconst.TPM_ES_SYM_CBC_PKCS5PAD))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint16)(tspiconst.TPM_SS_NONE))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint32)(12))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint32)(128))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint32)(len(iv)))
	if err != nil {
		return nil, nil, err
	}
	err = binary.Write(symheader, binary.BigEndian, (uint32)(0))
	if err != nil {
		return nil, nil, err
	}
	header := make([]byte, 28)
	err = binary.Read(symheader, binary.BigEndian, &header)
	header = append(header, iv...)
	header = append(header, symenc...)
	symenc = header

	return asymenc, symenc, nil
}

// VerifyEKCert verifies that the provided EK certificate is signed by a
// trusted manufacturer.
func VerifyEKCert(ekcert []byte) error {
	trustedCerts := map[string]string{
		"STM1": `-----BEGIN CERTIFICATE-----
MIIDzDCCArSgAwIBAgIEAAAAATANBgkqhkiG9w0BAQsFADBKMQswCQYDVQQGEwJD
SDEeMBwGA1UEChMVU1RNaWNyb2VsZWN0cm9uaWNzIE5WMRswGQYDVQQDExJTVE0g
VFBNIEVLIFJvb3QgQ0EwHhcNMDkwNzI4MDAwMDAwWhcNMjkxMjMxMDAwMDAwWjBV
MQswCQYDVQQGEwJDSDEeMBwGA1UEChMVU1RNaWNyb2VsZWN0cm9uaWNzIE5WMSYw
JAYDVQQDEx1TVE0gVFBNIEVLIEludGVybWVkaWF0ZSBDQSAwMTCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAJQYnWO8iw955vWqakWNr3YyazQnNzqV97+l
Qa+wUKMVY+lsyhAyOyXO31j4+clvsj6+JhNEwQtcnpkSc+TX60eZvLhgZPUgRVuK
B9w4GUVyg/db593QUmP8K41Is8E+l32CQdcVh9go0toqf/oS/za1TDFHEHLlB4dC
joKkfr3/hkGA9XJaoUopO2ELt4Otop12aw1BknoiTh1+YbzrZtAlIwK2TX99GW3S
IjaCi+fLoXyK2Fmx8vKnr9JfNL888xK9BQfhZzKmbKm/eLD1e1CFRs1B3z2gd3ax
pW5j1OIkSBMOIUeip5+7xvYo2gor5mxatB+rzSvrWup9AwIcymMCAwEAAaOBrjCB
qzAdBgNVHQ4EFgQU88kVdKbnc/8TvwxrrXp7Zc8ceCAwHwYDVR0jBBgwFoAUb+bF
bAe3bIsKgZKDXMtBHva00ScwRQYDVR0gAQH/BDswOTA3BgRVHSAAMC8wLQYIKwYB
BQUHAgEWIWh0dHA6Ly93d3cuc3QuY29tL1RQTS9yZXBvc2l0b3J5LzAOBgNVHQ8B
Af8EBAMCAAQwEgYDVR0TAQH/BAgwBgEB/wIBADANBgkqhkiG9w0BAQsFAAOCAQEA
uZqViou3aZDGvaAn29gghOkj04SkEWViZR3dU3DGrA+5ZX+zr6kZduus3Hf0bVHT
I318PZGTml1wm6faDRomE8bI5xADWhPiCQ1Gf7cFPiqaPkq7mgdC6SGlQtRAfoP8
ISUJlih0UtsqBWGql4lpk5G6YmvAezguWmMR0/O5Cx5w8YKfXkwAhegGmMGIoJFO
oSzJrS7jK2GnGCuRG65OQVC5HiQY2fFF0JePLWG/D56djNxMbPNGTHF3+yBWg0DU
0xJKYKGFdjFcw0Wi0m2j49Pv3JD1f78c2Z3I/65pkklZGu4awnKQcHeGIbdYF0hQ
LtDSBV4DR9q5GVxSR9JPgQ==
-----END CERTIFICATE-----`,
		"STM2": `-----BEGIN CERTIFICATE-----
MIIDzDCCArSgAwIBAgIEAAAAAzANBgkqhkiG9w0BAQsFADBKMQswCQYDVQQGEwJD
SDEeMBwGA1UEChMVU1RNaWNyb2VsZWN0cm9uaWNzIE5WMRswGQYDVQQDExJTVE0g
VFBNIEVLIFJvb3QgQ0EwHhcNMTEwMTIxMDAwMDAwWhcNMjkxMjMxMDAwMDAwWjBV
MQswCQYDVQQGEwJDSDEeMBwGA1UEChMVU1RNaWNyb2VsZWN0cm9uaWNzIE5WMSYw
JAYDVQQDEx1TVE0gVFBNIEVLIEludGVybWVkaWF0ZSBDQSAwMjCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAJO3ihn/uHgV3HrlPZpv8+1+xg9ccLf3pVXJ
oT5n8PHHixN6ZRBmf/Ng85/ODZzxnotC64WD8GHMLyQ0Cna3MJF+MGJZ5R5JkuJR
B4CtgTPwcTVZIsCuup0aDWnPzYqHwvfaiD2FD0aaxCnTKIjWU9OztTD2I61xW2LK
EY4Vde+W3C7WZgS5TpqkbhJzy2NJj6oSMDKklfI3X8jVf7bngMcCR3X3NcIo349I
Dt1r1GfwB+oWrhogZVnMFJKAoSYP8aQrLDVl7SQOAgTXz2IDD6bo1jga/8Kb72dD
h8D2qrkqWh7Hwdas3jqqbb9uiq6O2dJJY86FjffjXPo3jGlFjTsCAwEAAaOBrjCB
qzAdBgNVHQ4EFgQUVx+Aa0fM55v6NZR87Yi40QBa4J4wHwYDVR0jBBgwFoAUb+bF
bAe3bIsKgZKDXMtBHvaO0ScwRQYDVR0gAQH/BDswOTA3BgRVHSAAMC8wLQYIKwYB
BQUHAgEWIWh0dHA6Ly93d3cuc3QuY29tL1RQTS9yZXBvc2l0b3J5LzAOBgNVHQ8B
Af8EBAMCAAQwEgYDVR0TAQH/BAgwBgEB/wIBATANBgkqhkiG9w0BAQsFAAOCAQEA
4gllWq44PFWcv0JgMPOtyXDQx30YB5vBpjS0in7f/Y/r+1Dd8q3EZwNOwYApe+Lp
/ldNqCXw4XzmO8ZCVWOdQdVOqHZuSOhe++Jn0S7M4z2/1PQ6EbRczGfw3dlX63Ec
cEnrn6YMcgPC63Q+ID53vbTS3gpeX/SGpngtVwnzpuJ5rBajqSQUo5jBTBtuGQpO
Ko6Eu7U6Ouz7BVgOSn0mLbfSRb77PjOLZ3+97gSiMmV0iofS7ufemYqA8sF7ZFv/
lM2eOe/eeS56Jw+IPsnEU0Tf8Tn9hnEig1KP8VByRTWAJgiEOgX2nTs5iJbyZeIZ
RUjDHQQ5onqhgjpfRsC95g==
-----END CERTIFICATE-----`,
		"NTC1": `-----BEGIN CERTIFICATE-----
MIIDSjCCAjKgAwIBAgIGAK3jXfbVMA0GCSqGSIb3DQEBBQUAMFIxUDAcBgNVBAMT
FU5UQyBUUE0gRUsgUm9vdCBDQSAwMTAlBgNVBAoTHk51dm90b24gVGVjaG5vbG9n
eSBDb3Jwb3JhdGlvbjAJBgNVBAYTAlRXMB4XDTEyMDcxMTE2MjkzMFoXDTMyMDcx
MTE2MjkzMFowUjFQMBwGA1UEAxMVTlRDIFRQTSBFSyBSb290IENBIDAxMCUGA1UE
ChMeTnV2b3RvbiBUZWNobm9sb2d5IENvcnBvcmF0aW9uMAkGA1UEBhMCVFcwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDoNqxhtD4yUtXhqKQGGZemoKJy
uj1RnWvmNgzItLeejNU8B6fOnpMQyoS4K72tMhhFRK2jV9RYzyJMSjEwyX0ASTO1
2yMti2UJQS60d36eGwk8WLgrFnnITlemshi01h9t1MOmay3TO1LLH/3/VDKJ+jbd
cbfIO2bBquN8r3/ojYUaNSPj6pK1mmsMoJXF4dGRSEwb/4ozBIw5dugm1MEq4Zj3
GZ0YPg5wyLRugQbt7DkUOX4FGuK5p/C0u5zX8u33EGTrDrRz3ye3zO+aAY1xXF/m
qwEqgxX5M8f0/DXTTO/CfeIksuPeOzujFtXfi5Cy64eeIZ0nAUG3jbtnGjoFAgMB
AAGjJjAkMA4GA1UdDwEB/wQEAwICBDASBgNVHRMBAf8ECDAGAQH/AgEAMA0GCSqG
SIb3DQEBBQUAA4IBAQBBQznOPJAsD4Yvyt/hXtVJSgBX/+rRfoaqbdt3UMbUPJYi
pUoTUgaTx02DVRwommO+hLx7CS++1F2zorWC8qQyvNbg7iffQbbjWitt8NPE6kCr
q0Y5g7M/LkQDd5N3cFfC15uFJOtlj+A2DGzir8dlXU/0qNq9dBFbi+y+Y3rAT+wK
fktmN82UT861wTUzDvnXO+v7H5DYXjUU8kejPW6q+GgsccIbVTOdHNNWbMrcD9yf
oS91nMZ/+/n7IfFWXNN82qERsrvOFCDsbIzUOR30N0IP++oqGfwAbKFfCOCFUz6j
jpXUdJlh22tp12UMsreibmi5bsWYBgybwSbRgvzE
-----END CERTIFICATE-----`,
		"NTC2": `-----BEGIN CERTIFICATE-----
MIIDSjCCAjKgAwIBAgIGAPadBmPZMA0GCSqGSIb3DQEBBQUAMFIxUDAcBgNVBAMT
FU5UQyBUUE0gRUsgUm9vdCBDQSAwMjAlBgNVBAoTHk51dm90b24gVGVjaG5vbG9n
eSBDb3Jwb3JhdGlvbjAJBgNVBAYTAlRXMB4XDTEyMDcxMTE2MzMyNFoXDTMyMDcx
MTE2MzMyNFowUjFQMBwGA1UEAxMVTlRDIFRQTSBFSyBSb290IENBIDAyMCUGA1UE
ChMeTnV2b3RvbiBUZWNobm9sb2d5IENvcnBvcmF0aW9uMAkGA1UEBhMCVFcwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDSagWxaANT1YA2YUSN7sq7yzOT
1ymbIM+WijhE5AGcLwLFoJ9fmaQrYL6fAW2EW/Q3yu97Q9Ysr8yYZ2XCCfxfseEr
Vs80an8Nk6LkTDz8+0Hm0Cct0klvNUAZEIvWpmgHZMvGijXyOcp4z494d8B28Ynb
I7x0JMXZZQQKQi+WfuHtntF+2osYScweocipPrGeONLKU9sngWZ2vnnvw1SBneTa
irxq0Q0SD6Bx9jtxvdf87euk8JzfPhX8jp8GEeAjmLwGR+tnOQrDmczGNmp7YYNN
R+Q7NZVoYWHw5jaoZnNxbouWUXZZxFqDsB/ndCKWtsIzRYPuWcqrFcmUN4SVAgMB
AAGjJjAkMA4GA1UdDwEB/wQEAwICBDASBgNVHRMBAf8ECDAGAQH/AgEAMA0GCSqG
SIb3DQEBBQUAA4IBAQAIkdDSErzPLPYrVthw4lKjW4tRYelUicMPEHKjQeVUAAS5
y9XTzB4DWISDAFsgtQjqHJj0xCG+vpY0Rmn2FCO/0YpP+YBQkdbJOsiyXCdFy9e4
gGjQ24gw1B+rr84+pkI51y952NYBdoQDeb7diPe+24U94f//DYt/JQ8cJua4alr3
2Pohhh5TxCXXfU2EHt67KyqBSxCSy9m4OkCOGLHL2X5nQIdXVj178mw6DSAwyhwR
n3uJo5MvUEoQTFZJKGSXfab619mIgzEr+YHsIQToqf44VfDMDdM+MFiXQ3a5fLii
hEKQ9DhBPtpHAbhFA4jhCiG9HA8FdEplJ+M4uxNz
-----END CERTIFICATE-----`,
		"IFX1": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEMV64bDANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0wNTEwMjAxMzQ3NDNaFw0yNTEwMjAxMzQ3NDNaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwMTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALftPhYN
t4rE+JnU/XOPICbOBLvfo6iA7nuq7zf4DzsAWBdsZEdFJQfaK331ihG3IpQnlQ2i
YtDim289265f0J4OkPFpKeFU27CsfozVaNUm6UR/uzwA8ncxFc3iZLRMRNLru/Al
VG053ULVDQMVx2iwwbBSAYO9pGiGbk1iMmuZaSErMdb9v0KRUyZM7yABiyDlM3cz
UQX5vLWV0uWqxdGoHwNva5u3ynP9UxPTZWHZOHE6+14rMzpobs6Ww2RR8BgF96rh
4rRAZEl8BXhwiQq4STvUXkfvdpWH4lzsGcDDtrB6Nt3KvVNvsKz+b07Dk+Xzt+EH
NTf3Byk2HlvX+scCAwEAAaOCATswggE3MB0GA1UdDgQWBBQ4k8292HPEIzMV4bE7
qWoNI8wQxzAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBABJ1+Ap3rNlxZ0FW0aIgdzktbNHlvXWNxFdYIBbM
OKjmbOos0Y4O60eKPu259XmMItCUmtbzF3oKYXq6ybARUT2Lm+JsseMF5VgikSlU
BJALqpKVjwAds81OtmnIQe2LSu4xcTSavpsL4f52cUAu/maMhtSgN9mq5roYptq9
DnSSDZrX4uYiMPl//rBaNDBflhJ727j8xo9CCohF3yQUoQm7coUgbRMzyO64yMIO
3fhb+Vuc7sNwrMOz3VJN14C3JMoGgXy0c57IP/kD5zGRvljKEvrRC2I147+fPeLS
DueRMS6lblvRKiZgmGAg7YaKOkOaEmVDMQ+fTo2Po7hI5wc=
-----END CERTIFICATE-----`,
		"IFX2": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEaItIgTANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0wNjEyMjExMDM0MDBaFw0yNjEyMjExMDM0MDBaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwMjCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK6KnP5R
8ppq9TtPu3mAs3AFxdWhzK5ks+BixGR6mpzyXG64Bjl4xzBXeBIVtlBZXYvIAJ5s
eCTEEsnZc9eKNJeFLdmXQ/siRrTeonyxoS4aL1mVEQebLUz2gN9J6j1ewly+OvGk
jEYouGCzA+fARzLeRIrhuhBI0kUChbH7VM8FngJsbT4xKB3EJ6Wttma25VSimkAr
SPS6dzUDRS1OFCWtAtHJW6YjBnA4wgR8WfpXsnjeNpwEEB+JciWu1VAueLNI+Kis
RiferCfsgWRvHkR6RQf04h+FlhnYHJnf1ktqcEi1oYAjLsbYOAwqyoU1Pev9cS28
EA6FTJcxjuHhH9ECAwEAAaOCATswggE3MB0GA1UdDgQWBBRDMlr1UAQGVIkwzamm
fceAZ7l4ATAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBAIZAaYGzf9AYv6DqoUNx6wdpayhCeX75/IHuFQ/d
gLzat9Vd6qNKdAByskpOjpE0KRauEzD/BhTtkEJDazPSmVP1QxAPjqGaD+JjqhS/
Q6aY+1PSDi2zRIDA66V2yFJDcUBTtShbdTg144YSkVSY5UCKhQrsdg8yAbs7saAB
LHzVebTXffjmkTk5GZk26d/AZQRjfssta1N/TWhWTfuZtwYvjZmgDPeCfr6AOPLr
pVJz+ntzUKGpQ+5mwDJXMZ0qeiFIgXUlU0D+lfuajc/x9rgix9cM+o7amgDlRi1T
55Uu2vzUQ9jLUaISFaTTMag+quBDhx8BDVu+igLp5hvBtxQ=
-----END CERTIFICATE-----`,
		"IFX3": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEH7fYljANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0wNzA0MTMxNjQ0MjRaFw0yNzA0MTMxNjQ0MjRaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwMzCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAJWdPAuH
z/p1tIwB1QXlPD/PjedZ4uBZdwPH5tI3Uve0TzbR/mO5clx/loWn7nZ5cHkH1nhB
R67JEFY0a9GithPfITh0XRxPcisLBE/SoqZ90KHFaS+N6SwOpdCP0GlUg1OesKCF
79Z6fXrkTZsVpPqdawdZK+oUsDO9z9U6xqV7bwsS75Y+QiHsm6UTgAkSNQnuFMP3
NqQyDi/BaWaYRGQ6K8pM7Y7e1h21z/+5X7LncZXU8hgpYpu2zQPg96IkYboVUKL4
00snaPcOvfagsBUGlBltNfz7geaSuWTCdwEiwlkCYZqCtbkAj5FiStajrzP72BfT
2fshIv+5eF7Qp5ECAwEAAaOCATswggE3MB0GA1UdDgQWBBTGyypNtylL6RFyT1BB
MQtMQvibsjAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBAGN1bkh4J90DGcOPP2BlwE6ejJ0iDKf1zF+7CLu5
WS5K4dvuzsWUoQ5eplUt1LrIlorLr46mLokZD0RTG8t49Rcw4AvxMgWk7oYk69q2
0MGwXwgZ5OQypHaPwslmddLcX+RyEvjrdGpQx3E/87ZrQP8OKnmqI3pBlB8QwCGL
SV9AERaGDpzIHoObLlUjgHuD6aFekPfeIu1xbN25oZCWmqFVIhkKxWE1Xu+qqHIA
dnCFhoIWH3ie9OsJh/iDRaANYYGyplIibDx1FJA8fqiBiBBKUlPoJvbqmZs4meMd
OoeOuCvQ7op28UtaoV6H6BSYmN5dOgW7r1lX2Re0nd84NGE=
-----END CERTIFICATE-----`,
		"IFX4": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEDhD4wDANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0wNzEyMDMxMzA3NTVaFw0yNzEyMDMxMzA3NTVaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwNDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAN3UBmDk
jJzzJ+WCgrq4tILtE9KJPMGHwvCsbJOlo7eHiEb8JQzGK1prkPQ3dowFRXPnqONP
WUa36/J3R32xgvuZHqAdliZCt8IUb9qYhDenuXo1SSqJ8LWp30QIJ0vnkaQ2TCkO
bveZZR3hK2OZKRTkFaV/iy2RH+Qs4JAe3diD8mlIu2gXAXnKJSkrzW6gbMzrlTOi
RCuGcatpy7Hfmodbz/0Trbuwtc3dyJZ3Ko1z9bz2Oirjh93RrmYjbtL0HhkAjMOR
83GLrzwUddSqmxtXXX8j5i+/gmE3AO71swOIESdGugxaKUzJ1jTqWKMZcx0E6BFI
lDIfKk0fJlSxHfECAwEAAaOCATswggE3MB0GA1UdDgQWBBSIs8E/YQXRBCKfWsDr
SZVkrNRzvTAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBAFtqClQNBLOzcGZUpsBqlz3frzM45iiBpxosG1Re
IgoAgtIBEtl609TG51tmpm294KqpfKZVO+xNzovm8k/heGb0jmYf+q1ggrk2qT4v
Qy2jgE0jbP/P8WWq8NHC13uMcBUGPaka7yofEDDwz7TcduQyJVfG2pd1vflnzP0+
iiJpfCk3CAQQnb+B7zsOp7jHNwpvHP+FhNwZaikaa0OdR/ML9da1sOOW3oJSTEjW
SMLuhaZHtcVgitvtOVvCI/aq47rNJku3xQ7c/s8FHnFzQQ+Q4TExbP20SrqQIlL/
9sFAb7/nKYNauusakiF3pfvMrJOJigNfJyIcWaGfyyQtVVI=
-----END CERTIFICATE-----`,
		"IFX5": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEVuRoqzANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0wOTEyMTExMDM4NDJaFw0yOTEyMTExMDM4NDJaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwNTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAL79zMCO
bjkg7gCWEuyGO49CisF/QrGoz9adW1FBuSW8U9IOlvWXNsvoasC1mhrsfkRRojuU
mWifxxxcVfOI9v1SbRfJ+i6lG21IcVe6ywLJdDliT+3vzvrb/2hU/XjCCMDWb/Pw
aZslV5iL4QEiKxvRIiWMYHW0MkkL7mzRBDVN/Vz3ZiL5Lpq7awiKuX9OXpS2a1wf
qSGAlm2TxjU884q9Ky85JJugn0Q/C3dc8aaFPKLHlRs6rIvN1l0LwB1b5EWPzTPJ
d9EhRPFJOAbJS66nSgX06Fl7eWB71ow6w/25otLQCbpy6OrF8wBVMtPMHqFb1c32
PaaNzpCBnIU7vaMCAwEAAaOCATswggE3MB0GA1UdDgQWBBS7z3zBhCExZtq1vlOo
cBTd00jYzDAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBAHomNJtmFNtRJI2+s6ZwdzCTHXXIcR/T+N/lfPbE
hIUG4Kg+3uQMP7zBi22m3I3Kk9SXsjLqV5mnsQUGMGlF7jw5W5Q+d6NSJz4taw9D
2DsiUxE/i5vrjWiUaWxv2Eckd4MUexe5Qz8YSh4FPqLB8FZnAlgx2kfdzRIUjkMq
EgFK8ZRSUjXdczvsud68YPVMIZTxK0L8POGJ6RYiDrjTelprfZ4pKKZ79XwxwAIo
pG6emUEf+doRT0KoHoCHr9vvWCWKhojqlQ6jflPZcEsNBMbq5KHVN77vOU58OKx1
56v3EaqrZenVFt8+n6h2NzhOmg2quQXIr0V9jEg8GAMehDs=
-----END CERTIFICATE-----`,
		"IFX8": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIEfGoY6jANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0xMjA3MTcwOTI0NTJaFw0zMDEwMTgyMzU5NTlaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAwODCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAOJaIJu6
r/betrMgWJ/JZ5j8ytoAA9RWq0cw7+W0e5L2kDLJMM288wYT+iEbfwx6sWSLAl7q
okXYDtTB9MFNhQ5ZWFLslFXbYigtXJxwANcSdPISTF1Czn6LLi1fu1EHddwCXFC8
xaX0iGgQ9pZklvAy2ijK9BPHquWisisEiWZNRT9dCVylzOR3+p2YOC3ZrRmg7Bj+
DkC7dltTTO6dPR+LNOFe01pJlpZdF4YHcu4EC10gRu0quZz1LtDZWFKezK7rg5Rj
LSAJbKOsGXjl6hQXMtADEX9Vlz1vItD21OYCNRsu6VdipiL0bl0aAio4BV3GMyjk
0gHnQwCk9k/YPU8CAwEAAaOCATswggE3MB0GA1UdDgQWBBRMS01kiQjkW/5aENNj
h6aIrsHPeDAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBALMiDyQ9WKH/eTI84Mk8KYk+TXXEwf+fhgeCvxOQ
G0FTSmOpJaNIzxWXr/gDbY3dO0ODjWRKYvhimZUuV+ckMA+wZX2C6o8g5njpWIOH
pSAa+W35ijArh0Zt3MASJ46avd+fnQGTdzT0hK46gx6n2KixLvaZsR3JtuwUFYlQ
wzmz/UsbBNEoPiR8p5E0Zf5GEGiTqkmBVYyS6XA34axpMMRHy0wI7AGs0gVihwUM
rr0iWOu+GAcrm11lcYzqJvuEkfenAF62ufA2Ktv+Ut2xiRC0jUIp73CeplAJsqBr
camV3pJn3qYPI5c1njMRYnoRFWQbrOR5ADWDQLFQPYRrJmg=
-----END CERTIFICATE-----`,
		"IFX15": `-----BEGIN CERTIFICATE-----
MIIEnzCCA4egAwIBAgIER3V5aDANBgkqhkiG9w0BAQUFADBtMQswCQYDVQQGEwJE
RTEQMA4GA1UECBMHQmF2YXJpYTEhMB8GA1UEChMYSW5maW5lb24gVGVjaG5vbG9n
aWVzIEFHMQwwCgYDVQQLEwNBSU0xGzAZBgNVBAMTEklGWCBUUE0gRUsgUm9vdCBD
QTAeFw0xMjExMTQxNDQzMzRaFw0zMDEwMTgyMzU5NTlaMHcxCzAJBgNVBAYTAkRF
MQ8wDQYDVQQIEwZTYXhvbnkxITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2ll
cyBBRzEMMAoGA1UECxMDQUlNMSYwJAYDVQQDEx1JRlggVFBNIEVLIEludGVybWVk
aWF0ZSBDQSAxNTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKS6pgcg
OQWSozVbMkdf9jZkpGdT4U735zs0skfpjoKK2CgpLMO/+oGKbObm/DQPRQO/oxvq
jJNBKz55QBgKd+MoQ6t+2J8mcQ91Nfwqnm1C4r+c4zezJ1Utk/KIYNqpFDAzefBA
/lK8IxQ6kmzxcIFE4skaFsSgkearSZGG6sA9A51yxwvs8yUrQF51ICEUM7wDb4cM
53utaFdm6p6m9UZGSmmrdTiemOkuuwtl8IUQXfuk9lFyQsACBTM95Hrts0IzI6hX
QeTwSL4JqyEnKP9vbtT4eXzWNycqSYBf0+Uo/HHZo9WuVDUaA4I9zcmD0qCvSOT0
NAj4ifJ7SPGInU0CAwEAAaOCATswggE3MB0GA1UdDgQWBBR4pAnEV95pJvbfQsYR
TrflaptW5zAOBgNVHQ8BAf8EBAMCAgQwEgYDVR0TAQH/BAgwBgEB/wIBADBYBgNV
HSABAf8ETjBMMEoGC2CGSAGG+EUBBy8BMDswOQYIKwYBBQUHAgEWLWh0dHA6Ly93
d3cudmVyaXNpZ24uY29tL3JlcG9zaXRvcnkvaW5kZXguaHRtbDCBlwYDVR0jBIGP
MIGMgBRW65FEhWPWcrOu1EWWC/eUDlRCpqFxpG8wbTELMAkGA1UEBhMCREUxEDAO
BgNVBAgTB0JhdmFyaWExITAfBgNVBAoTGEluZmluZW9uIFRlY2hub2xvZ2llcyBB
RzEMMAoGA1UECxMDQUlNMRswGQYDVQQDExJJRlggVFBNIEVLIFJvb3QgQ0GCAQMw
DQYJKoZIhvcNAQEFBQADggEBAAnZDdJZs5QgAXnl0Jo5sCqktZcwcK+V1+uhsqrT
Z7OJ9Ze1YJ9XB14KxRfmck7Erl5HVc6PtUcLnR0tuJKKKqm7dTe4sQEFYd5usjrW
KSG6y7BOH7AdonocILY9OIxuNwxMAqhK8LIjkkRCeOWSvCqLnaLtrP52C0fBkTTM
SWX7YnsutXEpwhro3Qsnm9hL9s3s/WoIuNKUcLFH/qWKztpxXnF0zip73gcZbwEy
1GPQQpYnxFJ2R2ab2RHlO+3Uf3FDxn+eRLXNl95ZZ6GE4OIIpKEg2urIiig0HmGA
ijO6JfJxT30H9QNsx78sjYs7pOfMw6DfiqJ8Fx82GcCUOyM=
-----END CERTIFICATE-----`,
	}

	trustedKeys := map[string]string{
		"ATM1": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA5a+4yXtubUnGJPz7ZVjW
spGhe5Tr9BhrWQ9QOfDoVlkmi+tg2Gqc9qA9R39WhiiEmKuQ3+4XIvO+XFFrVCv+
8YuEnBIIu46FEX1LBG7LnUJbkgAV0pzHMWOghLhDIGcGg/6kO9fH+7KcwfwNah+Z
Lkfbfis3cm09RlOG8RSq8LyK8Zc07QH2M7L/8PFnRRQ5MnAgW7vuk8h/M62TTy+y
DVuWl/jnh3vrKgVMoOyL/iM3EdFHIV+r/lJOI/HRAlkieCRGiJ1kJI4tQpUXiL+t
K2iB1yAxnqcSQzWWxlNuu6sAUV2tI+qMVM2o6eT1f/MM7+b4nmyuBrdOIO5dvTek
7wIDAQAB
-----END PUBLIC KEY-----`,
		"ATM10": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1HEeCn82r8Vrg64howvt
s/2Yj5yuXhEv7FwUaISo7rYFnAbstyMVGNsL24reRW7aiyj65iHEojx1Car3Jlu8
hYfTqiagFmEK/qe8erEatwg5jFQ0GdK3a9Tw7nhYZjjpe98sRjL/DQlI5NiKClPF
4ZNqI418MGmzmPN/QNxRnKJAr5LTE8FZCMShm5V9ege72oLj4VfPuz29fQLBZWJA
2+NUy6gCpcUShejGfm9DTGBk0m3wruqyXuzKzJ4U0Xo5nTqp7ZV/JLbSWnTJxs72
hZ4D1nbi0paPQgcK5FYbIwQ1WuEpN5QMfilsAZFbokQnQa+jX2Rzleoa0mURVEnG
TQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM11": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqrS1Ei1VvLxVYWNv/s35
aYmHa5MOnvQybgW/i1XuINZuPQqYNw02wrWRA5eft2ts+FkJDvbz6edtqH30Cjx+
pmE9OvotY7O7rboruz2W2PpcQ0/inLUixNiq/A8giKRWJaTNsiF6VLNEWEe5sf3r
e6+C/ZQUfwupM56bU0JrIMxEgp8SCFguQsNwcq2txtr1ujgX4PN6DOet6BcrTpsY
vxKhQPbL/Tfl/kMqYVzqqyrBGgkg8+2FRxO6bVSJCwryrA37ZwkgJl3k/qibTFzw
JVfJnGaT57L/TUVJRwADtVN+oJFcQamR7N6VIuohx7fp7U8t8nrKhZpIwK405hLE
1wIDAQAB
-----END PUBLIC KEY-----`,
		"ATM12": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1R4fVBbUI7tsaRuS0VvH
TiouZ7L/HAokHIsozeLKamistnoCNKMyz40pwGenK8l0XnvJveoAlT+3tbKmmeo7
iR1QP28shN8ggnObHZpBN2cPiySh0A8uYIEXY3sJMrNZ/h2MMbjYD2nUyreVpuj+
mytSbvmB1eqcj9p4JBY1aqwXBrm0DqxZXJsVNBkjJ7c9kFy3ze8yccTFKx/wXe7v
PHgoWKA7RLtM2i/ppJdQgT/IubOrhZTyWPpvLQDpNf0p/SDmByw3/RzinPFlxUmA
eBzvTDwM7jEsZIUQGDZIBhifK45VLLrQiDTTOL8eVw9D8H0zn8AJu7cF9w9oYQj6
fQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM13": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwSvUzUQRjD3KyZMJu5CZ
sHVLqAN0mBT86nivjCV+RhHsR0uSU110NKjb2IHXM3fYaJlgvZRTAQJ3c8jPc0bV
tC0+521QKKt86OLi0+nrNT2QzLtrJgx6eRoYiu9se2F0u/zB+PHQ0R8qVOI1xw0W
FVK2w2+mRT3WN4Udp0Rn6LKPRXCq59JPfuD7hNBVmwp9uzj2TLp8ljMRSbChRjek
u9H4G+b2qa4wf2g9R5/Eqq5cMwYsD4357cuhlfqWQxD61J93ro2Hf5+30JrZF2yo
/TX5ocdQqnvLeape1KZGTDrikNXD9eIhR53yT4aBAK/RB8scsbwjn5tnoIBO0GAE
wwIDAQAB
-----END PUBLIC KEY-----`,
		"ATM2": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA6e/oMMyv7NPlNKo9A7ls
wJDkE14LZGSsKiRyzyJe2qHCoBwsZEq3QV2w4ducngcOysRaczDd8Cku9Z332dbi
9Tq6RZapk/eL7mcBF/XBNu7Wft3PDRTzs8gpM3xFHo59+OAjeax3TE8yR6Phiipp
42uI9f184vQsgwzEWNz6kMUP5mwe3Hm9y8RlTzrpJAtVtW9w7LmoSeOHHgzsEGG9
q5F2PkY6X0Ft9eDQaIOlTofkSHzvG2VLv8MJINMjnXrPnvF8dqG38lzAIvDfJKKx
E5MjY2aoFc1dfISQDv4sYx02b6jwfDl8pFCrwCzH7cWmIE0hD2BV2LpurURTi+kk
MQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM3": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmFAUATJ/8xIeuFW0aKNZ
cwCtDLDOQoGIez+sg64EmJ+rEKNHfPQj7ee3UxiARWoLvmrMwMZFHXVuHb6Q6QUB
H2OmUZfhcBXm94ZbXrFQyDxosR3lU+JZu5JK0X76kZAQtiBI2/UkJMt0xRUm1P7p
InKnqk9HYw2HpwrCYZII78R2N8bfGJAXOpz1oOpI7jVHewtJY/gun1jgXuOwNW+1
ouDaUrB/8t9m0lvsoeqduq0Nhvl4AL6kxVFp4sSX3gFuSORcbFP4hu7UMJy6g59d
bsgpulbltd2+HDqxEa4VPOe3IlC4iUso091vq+V49ThYOZ0TXFjk1dNdcTFuK2O8
kQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM4": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxtl1salu/2SXeqCGMtac
0GMCqx6Qeh4KtrviPfGVdObksuyGRPPuaW25/oYNGLlf2x/kDlknB2eZFMiW8otK
XV+xO50uFcxoTHXZ1wnMD8x6CgM0z0j3DQ5d0eu5sRRDd7GHYEpuvMj0x8pgkZk+
C+Ux2hWlj8xEiEDH8q5wyP5AX92GkpoUAL0e4vUV7/cCEBcsLOs8Oq6rKzmig82k
sDiL/Tn/8TtsSmUAwT/FzM/gi79x0D1hHtXyLOFPDIrOy3d3cz5QMOSSLDHGjmoa
DBF0+rcBRJCCU2iBtfIMBevJO1IBUmj/EgerHWWaf5pNNJAL62eAeeaWKm5OaX6b
VQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM5": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqg1FjhOQ/GsqgUL+WSBc
fYr3hkxxl1EBaCIqiolc1R5TmZabaHoCUIJmhwXfFGhFvhZrI0AcEqIYANLgY0ri
vn4h6vV+rrwtxM1X7X8hmT+AdvxL2p9lOCp7XNpQ2Rw+o/2JhuBSUv+opW7K6wPu
mwUpy4gL+dAozZyBjan5Co9sP5Yh3kdG3+ezRsfxf3CnS76NhjHglK9AWUdM1c6m
FGqTP+fU9ySTpoabrnDUzIcedu5OsMZa5L1LeLKHrqrbM3b995aU9RPrIQoO3x+B
Sau9Ab0d614zpD0TeWwjBT7Y5rIg14bIx7DiA+4kGRE5fHZtpdSWpgJmlkcObSSd
vQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM6": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAw9s9ykpgzH1Birz8JVmH
c/jXNSaFsDQYM9hcS3513P7arS+r/musmUqfdybY/GQ8+YqAA6IZq+xlGZ7WUEmx
vo/Hlnm/FQj3epZGD3ruApm+8dLV4Hvs4RjZCS8cfpkRZygh+nRdrZ3aAwZRjQbh
vl8mAJOQ2Rp9ANaZMzYjre7adkuqIFDa83nKgn4XkMIft+MthOvcMYmZtSL64gvA
NZ7PiryB2cMhS+0yAMGK/Oqm0ELIRk8wqAR/l3txRVxoQEqA9fBk2bvGNiWva92c
hxCLpEX9+CcZWvUeH6Up5a2EDxw8BgxE0L6sqjINWn2uN0x7jlKysg+XEyo7qUpM
RQIDAQAB
-----END PUBLIC KEY-----`,
		"ATM7": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAlYRfDD3bMJB7w6Fl+Kez
oIaEkzz+2VCeF8v3ohsNnSKCgcx8uk7Seyp50bEtLB6vOzwgcM5WjFItKyYLZaHt
P43q3iHsvISjAGNcYGcQ3NrM4ia6pc2geVraupldPTdQFOQjwd+0P/Q2/RMbWY6g
+sbjIDtItEc1vhf6bI3TinT+bGxxXdR9B6qWgLNc/H31wsBNN2nPv1K4wE8MWfAV
7hcK2NoIc1zGEY+IiWPtXSu48zmvS2lxKWWYuiHBwCt5ns/U7M8XSjkaj7SbhD02
yKwiMavu7SMAceSWWMBuhjgm9/zfMAPKO9weQixVCeuXspByefsekPuk5ohbWOS/
bwIDAQAB
-----END PUBLIC KEY-----`,
		"ATM8": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAylXag4JmzoqaTK8eCcSj
DSOGRcrX8IK/fBawgEWKAgPy/yu+8FGXXK10XDeI1fmlAefDr7349Cn+v9/y+Mff
XIduKCqR5ZP9M1+fBk7k5l4x2SeC9NfIMG+c0yJ6m1kEmsSN04nlKk8foo54L7DY
+Kkjfh6EPxyzxsMDqqTrMifBThF2kqo5KUBDdBjIkOUH1dbLWGjEgUpKocCu2M41
U7jmjToatkA+d3V+PyQRF4/lJkTYiv6Jy1Zl1qpEkLvR0qKrzoHkOij7zBpQ/46B
2BW1pARjU7AIp3NtfS5THL59mpXxpuyj6y2h0wxNVRkPZpnVA0ZWHgz9J4TcIgAN
PwIDAQAB
-----END PUBLIC KEY-----`,
		"ATM9": `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxa0XBIlBj9t5bYB81Cea
mlV3I3rTathgyUn2VL7/wng2vK5x7BB3a1wjXt5k/uwR/M5tfIDOSjug/SdDfTWk
ZRe82Ij/oYJOytcrFzEzouIZVA7t/uKdXzRwz7dZ6LPnNVkp8GiWUDCDpTNNmnvi
eZcjqofajDcc0EC720NT+NqmkAJd0qYMs+i4TA2om2C2Lxpkq+YdJp2pJOCCQquC
wmLXB8OxrddjX4QW5mHzigx4fanC/GJcXuRQPXqLruHZlWmUqU7Sl6yZEluItHbB
ri4BxD3Q7eXH0sM3rZGUCLr3lMPXjGsfpogZT4zLr075QIxtwe5OUpnslU61c34X
BwIDAQAB
-----END PUBLIC KEY-----`,
	}

	cert, err := x509.ParseCertificate(ekcert)
	if err != nil {
		return fmt.Errorf("Unable to parse EKCert: %s", err)
	}

	for vendor, certificate := range trustedCerts {
		block, _ := pem.Decode([]byte(certificate))
		testcert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return fmt.Errorf("Unable to parse %s: %s", vendor, err)
		}
		err = testcert.CheckSignature(cert.SignatureAlgorithm, cert.RawTBSCertificate, cert.Signature)
		if err == nil {
			return nil
		}
	}

	for vendor, key := range trustedKeys {
		block, _ := pem.Decode([]byte(key))
		pubkey, err := x509.ParsePKIXPublicKey(block.Bytes)
		if err != nil {
			return fmt.Errorf("Unable to parse %s: %s", vendor, err)
		}
		hashdata := sha1.Sum(cert.RawTBSCertificate[:])
		err = rsa.VerifyPKCS1v15(pubkey.(*rsa.PublicKey), crypto.SHA1, hashdata[:], cert.Signature)
		if err == nil {
			return nil
		}
	}

	return fmt.Errorf("No matching certificate found")
}

// QuoteVerify verifies that a quote was genuinely provided by the TPM. It
// takes the quote data, quote validation blob, public half of the AIK,
// current PCR values and the nonce used in the original quote request. It
// then verifies that the validation block is a valid signature for the
// quote data, that the secrets are the same (in order to avoid replay
// attacks), and that the PCR values are the same. It returns an error if
// any stage of the validation fails.
func QuoteVerify(data []byte, validation []byte, aikpub []byte, pcrvalues [][]byte, secret []byte) error {
	n := big.NewInt(0)
	n.SetBytes(aikpub)
	e := 65537

	pKey := rsa.PublicKey{N: n, E: int(e)}

	dataHash := sha1.Sum(data[:])

	err := rsa.VerifyPKCS1v15(&pKey, crypto.SHA1, dataHash[:], validation)
	if err != nil {
		return err
	}

	pcrHash := data[8:28]
	nonceHash := data[28:48]

	secretHash := sha1.Sum(secret[:])

	if bytes.Equal(secretHash[:], nonceHash) == false {
		return fmt.Errorf("Secret doesn't match")
	}

	pcrComposite := []byte{0x00, 0x02, 0xff, 0xff, 0x00, 0x00, 0x01, 0x40}
	for i := 0; i < 16; i++ {
		pcrComposite = append(pcrComposite, pcrvalues[i]...)
	}
	pcrCompositeHash := sha1.Sum(pcrComposite[:])

	if bytes.Equal(pcrCompositeHash[:], pcrHash) == false {
		return fmt.Errorf("PCR values don't match")
	}

	return nil
}

// KeyVerify verifies that a key certification request was genuinely
// provided by the TPM. It takes the certification data, certification
// validation blob, the public half of the AIK, the public half of the key
// to be certified and the nonce used in the original quote request. It then
// verifies that the validation block is a valid signature for the
// certification data, that the certification data matches the certified key
// and that the secrets are the same (in order to avoid replay attacks). It
// returns an error if any stage of the validation fails.
func KeyVerify(data []byte, validation []byte, aikpub []byte, keypub []byte, secret []byte) error {
	n := big.NewInt(0)
	n.SetBytes(aikpub)
	e := 65537

	pKey := rsa.PublicKey{N: n, E: int(e)}

	dataHash := sha1.Sum(data[:])

	err := rsa.VerifyPKCS1v15(&pKey, crypto.SHA1, dataHash[:], validation)
	if err != nil {
		return err
	}

	keyHash := data[43:63]
	nonceHash := data[63:83]

	secretHash := sha1.Sum(secret[:])

	if bytes.Equal(secretHash[:], nonceHash) == false {
		return fmt.Errorf("Secret doesn't match")
	}

	certHash := sha1.Sum(keypub[:])

	if bytes.Equal(certHash[:], keyHash) == false {
		return fmt.Errorf("Key doesn't match")
	}

	return nil
}
