/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package dynamiccertificates

import "testing"

type fakeCertKeyProvider struct {
	cert   []byte
	key    []byte
	called bool
}

func (f *fakeCertKeyProvider) Name() string {
	return "FakeProvider"
}

func (f *fakeCertKeyProvider) CurrentCertKeyContent() ([]byte, []byte) {
	f.called = true
	return f.cert, f.key
}

func TestDynamicGetCertFunctionEmpty(t *testing.T) {
	fp := &fakeCertKeyProvider{serverCrt1, serverKey1, false}
	target := &DynamicGetCertFunction{certKeyContentProvider: fp}

	for i := 0; i < 10; i++ {
		certA, errA := target.GetCert()
		if errA != nil {
			t.Fatalf("unexpected err:%v", errA)
		}
		for j := 0; j < 10; j++ {
			certB, errB := target.GetCert()
			if errB != nil {
				t.Fatalf("unexpected err:%v", errB)
			}
			if certA != certB {
				t.Errorf("iteration %d:%d, the address of returned function from target is different", i, j)
			}
		}
	}
}

func TestDynamicGetCertFunction(t *testing.T) {
	fp := &fakeCertKeyProvider{serverCrt1, serverKey1, false}
	target := &DynamicGetCertFunction{certKeyContentProvider: fp}
	target.Enqueue()

	if !fp.called {
		t.Fatal("the cert provider hasn't been called")
	}

	for i := 0; i < 10; i++ {
		certA, errA := target.GetCert()
		if errA != nil {
			t.Fatalf("unexpected err:%v", errA)
		}
		for j := 0; j < 10; j++ {
			certB, errB := target.GetCert()
			if errB != nil {
				t.Fatalf("unexpected err:%v", errB)
			}
			if certA != certB {
				t.Errorf("iteration %d:%d, the address of returned function from target is different", i, j)
			}
		}
	}
}

func TestDynamicGetCertFunctionError(t *testing.T) {
	fp := &fakeCertKeyProvider{[]byte{1, 2}, serverKey, false}
	target := &DynamicGetCertFunction{certKeyContentProvider: fp}
	target.Enqueue()

	if !fp.called {
		t.Fatal("the cert provider hasn't been called")
	}

	certA, errA := target.GetCert()
	if errA == nil {
		t.Fatal("expected an error")
	}
	if certA == nil {
		t.Fatal("returned certificate is nil")
	}
}

func TestDynamicGetCertFunctionWithChange(t *testing.T) {
	fp := &fakeCertKeyProvider{serverCrt1, serverKey1, false}
	target := &DynamicGetCertFunction{certKeyContentProvider: fp}
	target.Enqueue()

	if !fp.called {
		t.Fatal("the cert provider hasn't been called")
	}

	certA, errA := target.GetCert()
	if errA != nil {
		t.Fatalf("unexpected err:%v", errA)
	}

	fp.called = false
	fp.cert = serverCrt2
	fp.key = serverKey2
	target.Enqueue()

	if !fp.called {
		t.Fatal("the cert provider hasn't been called")
	}

	certB, errB := target.GetCert()
	if errB != nil {
		t.Fatalf("unexpected err:%v", errA)
	}

	if certA == certB {
		t.Fatal("expected to get a different certificate after reloading")
	}
}

// valid for localhost
var serverCrt1 = []byte(`-----BEGIN CERTIFICATE-----
MIIC5TCCAc2gAwIBAgIJAOS8kx4rqQxcMA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNV
BAMMCWxvY2FsaG9zdDAeFw0yMDA2MTUxMTE2MDFaFw0yMDA3MTUxMTE2MDFaMBQx
EjAQBgNVBAMMCWxvY2FsaG9zdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoC
ggEBAPL7bdiu1h8BadQPn5tgN4cBbMLmP8jpNduoC7KExbtKz7mdbCi7t5/vRgEq
tEgJqcsBbCCZzAYQHExkRqchaiVOQf1JvYywHSJ0IQ9IpIB4WwZiRitKsBoUwufn
ekpvHNOUwKbjQWdxCz26sCSgDsLNK2COmJwFoTFUQWuC0X1SYsT5KqnJwTMP19Xq
GFI0sWsZoQxe7QhJBYu8ierA+OkS0yZiBvFX8Cb1ChjUA3D1Bred2eNSZSafij2z
ZsvpAQea7lUmRVAJe/+HHGgptXiHR+voWh5LnI+SGTfRIjgXSogc6rSlkDxBL3qs
BSOKoiF8sy9WkowY8FKGGQkMmJcCAwEAAaM6MDgwFAYDVR0RBA0wC4IJbG9jYWxo
b3N0MAsGA1UdDwQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcDATANBgkqhkiG9w0B
AQsFAAOCAQEAvqdSHV2OAY36Xwe+5egq2oH98zfxTyp9hgsIO/8VJf/ukw+sSKFY
ZEl3ABzjHk9BDyLLoj6DjvjHva6Ghk/ruYg9Q312+dkn/RRCuKx2cOUSq+SFZxra
Lv4BMO8miiPeVmvP1klhqZZMCV7qpC/MdVVn3SgGYB9ymhGQa0iE0scUk1+zDNIg
p7iHbi227WZ/pROEFt8sSf1MltaQ/0QI9G2yCxDgjPSNte8vCqVDbXZkXBE5i6qF
TbvIk4/K13UC3YAgfhedNzf5Smbe6moK008BCp7itKL6IDb20gI9htBsgzolT8O+
G5lxVI5gSU/VdcGjW0EyWcKEct4LZMUTCg==
-----END CERTIFICATE-----
`)

var serverKey1 = []byte(`-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDy+23YrtYfAWnU
D5+bYDeHAWzC5j/I6TXbqAuyhMW7Ss+5nWwou7ef70YBKrRICanLAWwgmcwGEBxM
ZEanIWolTkH9Sb2MsB0idCEPSKSAeFsGYkYrSrAaFMLn53pKbxzTlMCm40FncQs9
urAkoA7CzStgjpicBaExVEFrgtF9UmLE+SqpycEzD9fV6hhSNLFrGaEMXu0ISQWL
vInqwPjpEtMmYgbxV/Am9QoY1ANw9Qa3ndnjUmUmn4o9s2bL6QEHmu5VJkVQCXv/
hxxoKbV4h0fr6FoeS5yPkhk30SI4F0qIHOq0pZA8QS96rAUjiqIhfLMvVpKMGPBS
hhkJDJiXAgMBAAECggEAIreyFkfE6GE3Ucl5sKWqyWt2stJbQsWvoFb+dN9rsTsb
OxY3IgrQTdXOVtRXNgPLcuodHPtcn3El2fRp8+9eTz5DR4GFx9hSEV4uaxSiDIkl
2F+qTv049EELKD92xbPiloimjjHiYnlQdd161YDZGxRdoko9m+1h/r5fKpFihVk8
5H6RaGb2hga6iuIvAoZ0sGPOIchSOOXC0Dpz7AimSW8JnE2aWNlRu/jiBQ9RxhAr
WP5Ey2FpNqgQfD22pbx3Ql7ULdFV2GP4owo3eWDbvHtIq+Q9WibE7zfWTtBuTKYu
oeo2e8mkKR83KmtqWzLRGEgxDvzwT/fk8ldoiSZewQKBgQD8kdEYrqyJA6kGCrQY
YjX8BXu+c4fkm3yxwGLJiA7RExckQ5smxy1Fzl4I4PApxWStOWxm5Uh2s0xSbDW3
TnRyuzVq5XehQiB5vFzPgU3ywKLy4hXrKxSotH/k6yHQMF4QJZaPpkxPQNIbdN03
6yntrdNB4sUxpYbrtAeSYaqziQKBgQD2SEbbOOO6Zl96KAiQ0D3v7vP86H4gjyLV
w1VDiyRCPimHbT3kCNKQZMQdKPssvf3ie6JBMNQc0K3lkzU2qvihI6jOb6QK6QIF
5eqySPDV7ZysU4CaFHSLXg6pyJ5XB+3Y8mmxnEm6EmpeOuI+4MCZ5zcFR1+kIRHU
ORzLGERDHwKBgQDKHXJjuxyNBKXVFOmr/aPPyx+Md+2OjrMJl7g2KDAbNZi2R3e4
X3mmPA/aMQ9fjfwT9zj9WoxTmQYBi2CtERZ03cVQhtLl9AIDCS6IS6RyF6AOl8gM
ikwc+VzDdzp23M3ZRAspZ133qhq5KBsDbaf+8LR3LB67rQe8RTQt+wRcaQKBgQDE
BS7wWU1YFRc1IRwANt61U5k62MlanNJ7FWeNxPdtChD/y0EReLwvVSSKmQ2hxO6I
DyNLg9Ovw6BFM2+NPXN6vekjtdP5IxALJb4xfMDDZMXomuWmvVUtgAVnuVfdqV/z
5q2dQemkgffLXE6rATQKyu8N8or7FZ8dLP/v3jamvQKBgER5Rr91lvS2HFXow8Rg
tAmpti96MRH0SK2gdAoT7Xr9hsqqC8dAtgAdF+jzeecQk5IaNlGS0SRQpCdMIvE1
Qy8OUgg/TEsBhDpg4FbFXwqlOE1PVsV+HNw578YHkOkamSH1rxTW9EJI8h+aiwqr
Yw8ovJTCPLC33LushxKas9hY
-----END PRIVATE KEY-----
`)

var serverKey2 = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA13f50PPWuR/InxLIoJjHdNSG+jVUd25CY7ZL2J023X2BAY+1
M6jkLR6C2nSFZnn58ubiB74/d1g/Fg1Twd419iR615A013f+qOoyFx3LFHxU1S6e
v22fgJ6ntK/+4QD5MwNgOwD8k1jN2WxHqNWn16IF4Tidbv8M9A35YHAdtYDYaOJC
kzjVztzRw1y6bKRakpMXxHylQyWmAKDJ2GSbRTbGtjr7Ji54WBfG43k94tO5X8K4
VGbz/uxrKe1IFMHNOlrjR438dbOXusksx9EIqDA9a42J3qjr5NKSqzCIbgBFl6qu
45V3A7cdRI/sJ2G1aqlWIXh2fAQiaFQAEBrPfwIDAQABAoIBAAZbxgWCjJ2d8H+x
QDZtC8XI18redAWqPU9P++ECkrHqmDoBkalanJEwS1BDDATAKL4gTh9IX/sXoZT3
A7e+5PzEitN9r/GD2wIFF0FTYcDTAnXgEFM52vEivXQ5lV3yd2gn+1kCaHG4typp
ZZv34iIc5+uDjjHOWQWCvA86f8XxX5EfYH+GkjfixTtN2xhWWlfi9vzYeESS4Jbt
tqfH0iEaZ1Bm/qvb8vFgKiuSTOoSpaf+ojAdtPtXDjf1bBtQQG+RSQkP59O/taLM
FCVuRrU8EtdB0+9anwmAP+O2UqjL5izA578lQtdIh13jHtGEgOcnfGNUphK11y9r
Mg5V28ECgYEA9fwI6Xy1Rb9b9irp4bU5Ec99QXa4x2bxld5cDdNOZWJQu9OnaIbg
kw/1SyUkZZCGMmibM/BiWGKWoDf8E+rn/ujGOtd70sR9U0A94XMPqEv7iHxhpZmD
rZuSz4/snYbOWCZQYXFoD/nqOwE7Atnz7yh+Jti0qxBQ9bmkb9o0QW8CgYEA4D3d
okzodg5QQ1y9L0J6jIC6YysoDedveYZMd4Un9bKlZEJev4OwiT4xXmSGBYq/7dzo
OJOvN6qgPfibr27mSB8NkAk6jL/VdJf3thWxNYmjF4E3paLJ24X31aSipN1Ta6K3
KKQUQRvixVoI1q+8WHAubBDEqvFnNYRHD+AjKvECgYBkekjhpvEcxme4DBtw+OeQ
4OJXJTmhKemwwB12AERboWc88d3GEqIVMEWQJmHRotFOMfCDrMNfOxYv5+5t7FxL
gaXHT1Hi7CQNJ4afWrKgmjjqrXPtguGIvq2fXzjVt8T9uNjIlNxe+kS1SXFjXsgH
ftDY6VgTMB0B4ozKq6UAvQKBgQDER8K5buJHe+3rmMCMHn+Qfpkndr4ftYXQ9Kn4
MFiy6sV0hdfTgRzEdOjXu9vH/BRVy3iFFVhYvIR42iTEIal2VaAUhM94Je5cmSyd
eE1eFHTqfRPNazmPaqttmSc4cfa0D4CNFVoZR6RupIl6Cect7jvkIaVUD+wMXxWo
osOFsQKBgDLwVhZWoQ13RV/jfQxS3veBUnHJwQJ7gKlL1XZ16mpfEOOVnJF7Es8j
TIIXXYhgSy/XshUbsgXQ+YGliye/rXSCTXHBXvWShOqxEMgeMYMRkcm8ZLp/DH7C
kC2pemkLPUJqgSh1PASGcJbDJIvFGUfP69tUCYpHpk3nHzexuAg3
-----END RSA PRIVATE KEY-----`)

var serverCrt2 = []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----`)
