/*
Copyright 2014 The Kubernetes Authors.

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

package x509

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"net/http"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/auth/user"
)

const (
	rootCACert = `-----BEGIN CERTIFICATE-----
MIIDOTCCAqKgAwIBAgIJAOoObf5kuGgZMA0GCSqGSIb3DQEBBQUAMGcxCzAJBgNV
BAYTAlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0G
A1UEChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MRAwDgYDVQQDEwdST09UIENB
MB4XDTE0MTIwODIwMjU1N1oXDTI0MTIwNTIwMjU1N1owZzELMAkGA1UEBhMCVVMx
ETAPBgNVBAgTCE15IFN0YXRlMRAwDgYDVQQHEwdNeSBDaXR5MQ8wDQYDVQQKEwZN
eSBPcmcxEDAOBgNVBAsTB015IFVuaXQxEDAOBgNVBAMTB1JPT1QgQ0EwgZ8wDQYJ
KoZIhvcNAQEBBQADgY0AMIGJAoGBAMfcayGpuF4vwrP8SXKDMCTJ9HV1cvb1NYEc
UgKF0RtcWpK+i0jvhcEs0TPDZIwLSwFw6UMEt5xy4LUlv1K/SHGY3Ym3m/TXMnB9
gkfrbWlY9LBIm4oVXwrPWyNIe74qAh1Oi03J1492uUPdHhcEmf01RIP6IIqIDuDL
xNNggeIrAgMBAAGjgewwgekwHQYDVR0OBBYEFD3w9zA9O+s6VWj69UPJx6zhPxB4
MIGZBgNVHSMEgZEwgY6AFD3w9zA9O+s6VWj69UPJx6zhPxB4oWukaTBnMQswCQYD
VQQGEwJVUzERMA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENpdHkxDzAN
BgNVBAoTBk15IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMHUk9PVCBD
QYIJAOoObf5kuGgZMAwGA1UdEwQFMAMBAf8wCwYDVR0PBAQDAgEGMBEGCWCGSAGG
+EIBAQQEAwIBBjANBgkqhkiG9w0BAQUFAAOBgQBSrJjMevHUgBKkjaSyeKhOqd8V
XlbA//N/mtJTD3eD/HUZBgyMcBH+sk6hnO8N9ICHtndkTrCElME9N3JA+wg2fHLW
Lj09yrFm7u/0Wd+lcnBnczzoMDhlOjyVqsgIMhisFEw1VVaMoHblYnzY0B+oKNnu
H9oc7u5zhTGXeV8WPg==
-----END CERTIFICATE-----
`

	selfSignedCert = `-----BEGIN CERTIFICATE-----
MIIDEzCCAnygAwIBAgIJAMaPaFbGgJN+MA0GCSqGSIb3DQEBBQUAMGUxCzAJBgNV
BAYTAlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0G
A1UEChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MQ4wDAYDVQQDEwVzZWxmMTAe
Fw0xNDEyMDgyMDI1NThaFw0yNDEyMDUyMDI1NThaMGUxCzAJBgNVBAYTAlVTMREw
DwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UEChMGTXkg
T3JnMRAwDgYDVQQLEwdNeSBVbml0MQ4wDAYDVQQDEwVzZWxmMTCBnzANBgkqhkiG
9w0BAQEFAAOBjQAwgYkCgYEA2NAe5AE//Uccy/HSqr4TBhzSe4QD5NYOWuTSKVeX
LLJ0IK2SD3PfnFM/Y0wERx6ORZPGxM0ByPO1RgZe14uFSPEdnD2WTx4lcALK9Jci
IrsvGRyMH0ZT6Q+35ScchAOdOJJYcvXEWf/heZauogzNQAGskwZdYxQB4zwC/es/
EE0CAwEAAaOByjCBxzAdBgNVHQ4EFgQUfKsCqEU/sCgvcZFSonHu2UArQ3EwgZcG
A1UdIwSBjzCBjIAUfKsCqEU/sCgvcZFSonHu2UArQ3GhaaRnMGUxCzAJBgNVBAYT
AlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UE
ChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MQ4wDAYDVQQDEwVzZWxmMYIJAMaP
aFbGgJN+MAwGA1UdEwQFMAMBAf8wDQYJKoZIhvcNAQEFBQADgYEAxpo9Nyp4d3TT
FnEC4erqQGgbc15fOF47J7bgXxsKK8o8oR/CzQ+08KhoDn3WgV39rEfX2jENDdWp
ze3kOoP+iWSmTySHMSKVMppp0Xnls6t38mrsXtPuY8fGD2GS6VllaizMqc3wShNK
4HADGF3q5z8hZYSV9ICQYHu5T9meF8M=
-----END CERTIFICATE-----
`

	clientCNCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 1 (0x1)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Dec  8 20:25:58 2014 GMT
            Not After : Dec  5 20:25:58 2024 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=client_cn
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:a5:30:b3:2b:c0:bd:cb:29:cf:e2:d8:fd:68:b0:
                    03:c3:a6:3b:1b:ec:36:73:a1:52:5d:27:ee:02:35:
                    5c:51:ed:3d:3b:54:d7:11:f5:38:94:ee:fd:cc:0c:
                    22:a8:f8:8e:11:2f:7c:43:5a:aa:07:3f:95:4f:50:
                    22:7d:aa:e2:5d:2a:90:3d:02:1a:5b:d2:cf:3f:fb:
                    dc:58:32:c5:ce:2f:81:58:31:20:eb:35:d3:53:d3:
                    42:47:c2:13:68:93:62:58:b6:46:60:48:17:df:d2:
                    8c:c3:40:47:cf:67:ea:27:0f:09:78:e9:d5:2a:64:
                    1e:c4:33:5a:d6:0d:7a:79:93
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Basic Constraints: 
                CA:FALSE
            Netscape Comment: 
                OpenSSL Generated Certificate
            X509v3 Subject Key Identifier: 
                E7:FB:1F:45:F0:71:77:AF:8C:10:4A:0A:42:03:F5:1F:1F:07:CF:DF
            X509v3 Authority Key Identifier: 
                keyid:3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78
                DirName:/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=ROOT CA
                serial:EA:0E:6D:FE:64:B8:68:19

            X509v3 Subject Alternative Name: 
                <EMPTY>

            X509v3 Extended Key Usage: 
                TLS Web Client Authentication
            Netscape Cert Type: 
                SSL Client
    Signature Algorithm: sha256WithRSAEncryption
         08:bc:b4:80:a5:3b:be:9a:78:f9:47:3f:c0:2d:75:e3:10:89:
         61:b1:6a:dd:f4:a4:c4:6a:d3:6f:27:30:7f:2d:07:78:d9:12:
         03:bc:a5:44:68:f3:10:bc:aa:32:e3:3f:6a:16:12:25:eb:82:
         ac:ae:30:ef:0d:be:87:11:13:e7:2f:78:69:67:36:62:ba:aa:
         51:8a:ee:6e:1e:ca:35:75:95:25:2d:db:e6:cb:71:70:95:25:
         76:99:13:02:57:99:56:25:a3:33:55:a2:6a:30:87:8b:97:e6:
         68:f3:c1:37:3c:c1:14:26:90:a0:dd:d3:02:3a:e9:c2:9e:59:
         d2:44
-----BEGIN CERTIFICATE-----
MIIDczCCAtygAwIBAgIBATANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENpdHkxDzANBgNVBAoTBk15
IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMHUk9PVCBDQTAeFw0xNDEy
MDgyMDI1NThaFw0yNDEyMDUyMDI1NThaMGkxCzAJBgNVBAYTAlVTMREwDwYDVQQI
EwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UEChMGTXkgT3JnMRAw
DgYDVQQLEwdNeSBVbml0MRIwEAYDVQQDFAljbGllbnRfY24wgZ8wDQYJKoZIhvcN
AQEBBQADgY0AMIGJAoGBAKUwsyvAvcspz+LY/WiwA8OmOxvsNnOhUl0n7gI1XFHt
PTtU1xH1OJTu/cwMIqj4jhEvfENaqgc/lU9QIn2q4l0qkD0CGlvSzz/73Fgyxc4v
gVgxIOs101PTQkfCE2iTYli2RmBIF9/SjMNAR89n6icPCXjp1SpkHsQzWtYNenmT
AgMBAAGjggErMIIBJzAJBgNVHRMEAjAAMCwGCWCGSAGG+EIBDQQfFh1PcGVuU1NM
IEdlbmVyYXRlZCBDZXJ0aWZpY2F0ZTAdBgNVHQ4EFgQU5/sfRfBxd6+MEEoKQgP1
Hx8Hz98wgZkGA1UdIwSBkTCBjoAUPfD3MD076zpVaPr1Q8nHrOE/EHiha6RpMGcx
CzAJBgNVBAYTAlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0
eTEPMA0GA1UEChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MRAwDgYDVQQDEwdS
T09UIENBggkA6g5t/mS4aBkwCQYDVR0RBAIwADATBgNVHSUEDDAKBggrBgEFBQcD
AjARBglghkgBhvhCAQEEBAMCB4AwDQYJKoZIhvcNAQELBQADgYEACLy0gKU7vpp4
+Uc/wC114xCJYbFq3fSkxGrTbycwfy0HeNkSA7ylRGjzELyqMuM/ahYSJeuCrK4w
7w2+hxET5y94aWc2YrqqUYrubh7KNXWVJS3b5stxcJUldpkTAleZViWjM1WiajCH
i5fmaPPBNzzBFCaQoN3TAjrpwp5Z0kQ=
-----END CERTIFICATE-----`

	clientDNSCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 4 (0x4)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Dec  8 20:25:58 2014 GMT
            Not After : Dec  5 20:25:58 2024 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=client_dns
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:b0:6d:16:6a:fc:28:f7:dc:da:2c:a8:e4:0c:27:
                    3c:27:ce:ae:d5:72:d9:3c:eb:af:3d:a3:83:98:5b:
                    85:d8:68:f4:bd:53:57:d2:ad:e8:71:b1:18:8e:ae:
                    37:8e:02:9c:b2:6c:92:09:cc:5e:e6:74:a1:4b:e1:
                    50:41:08:9a:5e:d4:20:0b:6f:c7:c0:34:a8:e6:be:
                    77:1d:43:1f:2c:df:dc:ca:9d:1a:0a:9f:a3:6e:0a:
                    60:f1:6d:d9:7f:f0:f1:ea:66:9d:4c:f3:de:62:af:
                    b1:92:70:f1:bb:8a:81:f4:9c:3c:b8:c9:e8:04:18:
                    70:2f:77:74:48:d9:cd:e5:af
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Basic Constraints: 
                CA:FALSE
            Netscape Comment: 
                OpenSSL Generated Certificate
            X509v3 Subject Key Identifier: 
                6E:A3:F6:01:52:79:4D:46:78:3C:D0:AB:4A:75:96:AC:7D:6C:08:BE
            X509v3 Authority Key Identifier: 
                keyid:3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78
                DirName:/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=ROOT CA
                serial:EA:0E:6D:FE:64:B8:68:19

            X509v3 Subject Alternative Name: 
                DNS:client_dns.example.com
            X509v3 Extended Key Usage: 
                TLS Web Client Authentication
            Netscape Cert Type: 
                SSL Client
    Signature Algorithm: sha256WithRSAEncryption
         69:20:83:0f:16:f8:b6:f5:04:98:56:a4:b2:67:32:e0:82:80:
         da:8e:54:06:94:96:cd:56:eb:90:4c:f4:3c:50:80:6a:25:ac:
         3d:e2:81:05:e4:89:2b:55:63:9a:2d:4a:da:3b:c4:97:5e:1a:
         e9:6f:83:b8:05:4a:dc:bd:ab:b0:a0:75:d0:1e:b5:c5:8d:f3:
         f6:92:f1:52:d2:81:67:fc:6f:74:ee:49:37:73:08:bc:f5:26:
         86:67:f5:82:04:ff:db:5a:9f:f9:6b:df:2f:f5:75:61:f2:a5:
         91:0b:05:56:5b:e8:d1:36:d7:56:7a:ed:7d:e5:5f:2a:08:87:
         c2:48
-----BEGIN CERTIFICATE-----
MIIDjDCCAvWgAwIBAgIBBDANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENpdHkxDzANBgNVBAoTBk15
IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMHUk9PVCBDQTAeFw0xNDEy
MDgyMDI1NThaFw0yNDEyMDUyMDI1NThaMGoxCzAJBgNVBAYTAlVTMREwDwYDVQQI
EwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UEChMGTXkgT3JnMRAw
DgYDVQQLEwdNeSBVbml0MRMwEQYDVQQDFApjbGllbnRfZG5zMIGfMA0GCSqGSIb3
DQEBAQUAA4GNADCBiQKBgQCwbRZq/Cj33NosqOQMJzwnzq7Vctk86689o4OYW4XY
aPS9U1fSrehxsRiOrjeOApyybJIJzF7mdKFL4VBBCJpe1CALb8fANKjmvncdQx8s
39zKnRoKn6NuCmDxbdl/8PHqZp1M895ir7GScPG7ioH0nDy4yegEGHAvd3RI2c3l
rwIDAQABo4IBQzCCAT8wCQYDVR0TBAIwADAsBglghkgBhvhCAQ0EHxYdT3BlblNT
TCBHZW5lcmF0ZWQgQ2VydGlmaWNhdGUwHQYDVR0OBBYEFG6j9gFSeU1GeDzQq0p1
lqx9bAi+MIGZBgNVHSMEgZEwgY6AFD3w9zA9O+s6VWj69UPJx6zhPxB4oWukaTBn
MQswCQYDVQQGEwJVUzERMA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENp
dHkxDzANBgNVBAoTBk15IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMH
Uk9PVCBDQYIJAOoObf5kuGgZMCEGA1UdEQQaMBiCFmNsaWVudF9kbnMuZXhhbXBs
ZS5jb20wEwYDVR0lBAwwCgYIKwYBBQUHAwIwEQYJYIZIAYb4QgEBBAQDAgeAMA0G
CSqGSIb3DQEBCwUAA4GBAGkggw8W+Lb1BJhWpLJnMuCCgNqOVAaUls1W65BM9DxQ
gGolrD3igQXkiStVY5otSto7xJdeGulvg7gFSty9q7CgddAetcWN8/aS8VLSgWf8
b3TuSTdzCLz1JoZn9YIE/9tan/lr3y/1dWHypZELBVZb6NE211Z67X3lXyoIh8JI
-----END CERTIFICATE-----`

	clientEmailCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 2 (0x2)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Dec  8 20:25:58 2014 GMT
            Not After : Dec  5 20:25:58 2024 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=client_email
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:bf:f3:c3:d7:50:d5:64:d6:d2:e3:6c:bb:7e:5d:
                    4b:41:63:76:9c:c4:c8:33:9a:37:ee:68:24:1e:26:
                    cf:de:57:79:d6:dc:53:b6:da:12:c6:c0:95:7d:69:
                    b8:af:1d:4e:8f:a5:83:8b:22:78:e3:94:cc:6e:fe:
                    24:e2:05:91:ed:1c:01:b7:e1:53:91:aa:51:53:7a:
                    55:6e:fe:0c:ef:c1:66:70:12:0c:85:94:95:c6:3e:
                    f5:35:58:4d:3f:11:b1:5a:d6:ec:a1:f5:21:c1:e6:
                    1f:c1:91:5b:67:89:25:2a:e3:86:27:6b:d8:31:7b:
                    f1:0d:83:c7:f2:68:70:f0:23
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Basic Constraints: 
                CA:FALSE
            Netscape Comment: 
                OpenSSL Generated Certificate
            X509v3 Subject Key Identifier: 
                76:22:99:CD:3D:BA:90:62:0F:BE:E7:5B:57:8D:31:1D:25:27:C6:6A
            X509v3 Authority Key Identifier: 
                keyid:3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78
                DirName:/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=ROOT CA
                serial:EA:0E:6D:FE:64:B8:68:19

            X509v3 Subject Alternative Name: 
                email:client_email@example.com
            X509v3 Extended Key Usage: 
                TLS Web Client Authentication
            Netscape Cert Type: 
                SSL Client
    Signature Algorithm: sha256WithRSAEncryption
         80:70:19:d2:5c:c1:cf:d2:b6:e5:0e:76:cd:8f:c2:8d:a8:19:
         07:86:22:3f:a4:b1:98:c6:98:c1:dc:f8:99:5b:20:5c:6d:17:
         6b:fa:8b:4c:1b:86:14:b4:71:f7:41:22:03:ca:ec:2c:cd:ae:
         77:93:bd:08:06:8c:3c:06:ce:04:2c:b1:ce:79:20:0d:d5:01:
         1c:bd:66:60:38:db:4f:ad:dc:a6:33:8f:07:af:e6:bd:1c:27:
         4b:93:6a:4f:59:e3:cf:df:ff:87:f1:af:02:ad:50:06:f9:50:
         c7:59:87:bc:0c:e6:66:cd:d1:c8:df:e6:15:b2:21:b3:04:86:
         8c:89
-----BEGIN CERTIFICATE-----
MIIDkDCCAvmgAwIBAgIBAjANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENpdHkxDzANBgNVBAoTBk15
IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMHUk9PVCBDQTAeFw0xNDEy
MDgyMDI1NThaFw0yNDEyMDUyMDI1NThaMGwxCzAJBgNVBAYTAlVTMREwDwYDVQQI
EwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UEChMGTXkgT3JnMRAw
DgYDVQQLEwdNeSBVbml0MRUwEwYDVQQDFAxjbGllbnRfZW1haWwwgZ8wDQYJKoZI
hvcNAQEBBQADgY0AMIGJAoGBAL/zw9dQ1WTW0uNsu35dS0FjdpzEyDOaN+5oJB4m
z95XedbcU7baEsbAlX1puK8dTo+lg4sieOOUzG7+JOIFke0cAbfhU5GqUVN6VW7+
DO/BZnASDIWUlcY+9TVYTT8RsVrW7KH1IcHmH8GRW2eJJSrjhidr2DF78Q2Dx/Jo
cPAjAgMBAAGjggFFMIIBQTAJBgNVHRMEAjAAMCwGCWCGSAGG+EIBDQQfFh1PcGVu
U1NMIEdlbmVyYXRlZCBDZXJ0aWZpY2F0ZTAdBgNVHQ4EFgQUdiKZzT26kGIPvudb
V40xHSUnxmowgZkGA1UdIwSBkTCBjoAUPfD3MD076zpVaPr1Q8nHrOE/EHiha6Rp
MGcxCzAJBgNVBAYTAlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkg
Q2l0eTEPMA0GA1UEChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MRAwDgYDVQQD
EwdST09UIENBggkA6g5t/mS4aBkwIwYDVR0RBBwwGoEYY2xpZW50X2VtYWlsQGV4
YW1wbGUuY29tMBMGA1UdJQQMMAoGCCsGAQUFBwMCMBEGCWCGSAGG+EIBAQQEAwIH
gDANBgkqhkiG9w0BAQsFAAOBgQCAcBnSXMHP0rblDnbNj8KNqBkHhiI/pLGYxpjB
3PiZWyBcbRdr+otMG4YUtHH3QSIDyuwsza53k70IBow8Bs4ELLHOeSAN1QEcvWZg
ONtPrdymM48Hr+a9HCdLk2pPWePP3/+H8a8CrVAG+VDHWYe8DOZmzdHI3+YVsiGz
BIaMiQ==
-----END CERTIFICATE-----
`

	serverCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 7 (0x7)
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Dec  8 20:25:58 2014 GMT
            Not After : Dec  5 20:25:58 2024 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=127.0.0.1
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:e2:50:d9:1c:ff:03:34:0d:f8:b4:0c:08:70:fc:
                    2a:27:2f:42:c9:4b:90:f2:a7:f2:7c:8c:ec:58:a5:
                    0f:49:29:0c:77:b5:aa:0a:aa:b7:71:e7:2d:0e:fb:
                    73:2c:88:de:70:69:df:d1:b0:7f:3b:2d:28:99:2d:
                    f1:43:93:13:aa:c9:98:16:05:05:fb:80:64:7b:11:
                    19:44:b7:5a:8c:83:20:6f:68:73:4f:ec:78:c2:73:
                    de:96:68:30:ce:2a:04:03:22:80:21:26:cc:7e:d6:
                    ec:b5:58:a7:41:bb:ae:fc:2c:29:6a:d1:3a:aa:b9:
                    2f:88:f5:62:d8:8e:69:f4:19
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Basic Constraints: 
                CA:FALSE
            Netscape Comment: 
                OpenSSL Generated Certificate
            X509v3 Subject Key Identifier: 
                36:A1:0C:B2:28:0C:77:6C:7F:96:90:11:CA:19:AF:67:1E:92:17:08
            X509v3 Authority Key Identifier: 
                keyid:3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78
                DirName:/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=ROOT CA
                serial:EA:0E:6D:FE:64:B8:68:19

            X509v3 Subject Alternative Name: 
                <EMPTY>

            X509v3 Extended Key Usage: 
                TLS Web Server Authentication
            Netscape Cert Type: 
                SSL Server
    Signature Algorithm: sha256WithRSAEncryption
         a9:dd:3d:64:e5:e2:fb:7e:2e:ce:52:7a:85:1d:62:0b:ec:ca:
         1d:78:51:d1:f7:13:36:1c:27:3f:69:59:27:5f:89:ac:41:5e:
         65:c6:ae:dc:18:60:18:85:5b:bb:9a:76:93:df:60:47:96:97:
         58:61:34:98:59:46:ea:d4:ad:01:6c:f7:4e:6c:9d:72:26:4d:
         76:21:1b:7a:a1:f0:e6:e6:88:61:68:f5:cc:2e:40:76:f1:57:
         04:5b:9e:d2:88:c8:ac:9e:49:b5:b4:d6:71:c1:fd:d8:b8:0f:
         c7:1a:9c:f3:3f:cc:11:60:ef:54:3a:3d:b8:8d:09:80:fe:be:
         f9:ef
-----BEGIN CERTIFICATE-----
MIIDczCCAtygAwIBAgIBBzANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECBMITXkgU3RhdGUxEDAOBgNVBAcTB015IENpdHkxDzANBgNVBAoTBk15
IE9yZzEQMA4GA1UECxMHTXkgVW5pdDEQMA4GA1UEAxMHUk9PVCBDQTAeFw0xNDEy
MDgyMDI1NThaFw0yNDEyMDUyMDI1NThaMGkxCzAJBgNVBAYTAlVTMREwDwYDVQQI
EwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0eTEPMA0GA1UEChMGTXkgT3JnMRAw
DgYDVQQLEwdNeSBVbml0MRIwEAYDVQQDEwkxMjcuMC4wLjEwgZ8wDQYJKoZIhvcN
AQEBBQADgY0AMIGJAoGBAOJQ2Rz/AzQN+LQMCHD8KicvQslLkPKn8nyM7FilD0kp
DHe1qgqqt3HnLQ77cyyI3nBp39GwfzstKJkt8UOTE6rJmBYFBfuAZHsRGUS3WoyD
IG9oc0/seMJz3pZoMM4qBAMigCEmzH7W7LVYp0G7rvwsKWrROqq5L4j1YtiOafQZ
AgMBAAGjggErMIIBJzAJBgNVHRMEAjAAMCwGCWCGSAGG+EIBDQQfFh1PcGVuU1NM
IEdlbmVyYXRlZCBDZXJ0aWZpY2F0ZTAdBgNVHQ4EFgQUNqEMsigMd2x/lpARyhmv
Zx6SFwgwgZkGA1UdIwSBkTCBjoAUPfD3MD076zpVaPr1Q8nHrOE/EHiha6RpMGcx
CzAJBgNVBAYTAlVTMREwDwYDVQQIEwhNeSBTdGF0ZTEQMA4GA1UEBxMHTXkgQ2l0
eTEPMA0GA1UEChMGTXkgT3JnMRAwDgYDVQQLEwdNeSBVbml0MRAwDgYDVQQDEwdS
T09UIENBggkA6g5t/mS4aBkwCQYDVR0RBAIwADATBgNVHSUEDDAKBggrBgEFBQcD
ATARBglghkgBhvhCAQEEBAMCBkAwDQYJKoZIhvcNAQELBQADgYEAqd09ZOXi+34u
zlJ6hR1iC+zKHXhR0fcTNhwnP2lZJ1+JrEFeZcau3BhgGIVbu5p2k99gR5aXWGE0
mFlG6tStAWz3TmydciZNdiEbeqHw5uaIYWj1zC5AdvFXBFue0ojIrJ5JtbTWccH9
2LgPxxqc8z/MEWDvVDo9uI0JgP6++e8=
-----END CERTIFICATE-----
`
)

func TestX509(t *testing.T) {
	testCases := map[string]struct {
		Insecure bool
		Certs    []*x509.Certificate

		Opts x509.VerifyOptions
		User UserConversion

		ExpectUserName string
		ExpectOK       bool
		ExpectErr      bool
	}{
		"non-tls": {
			Insecure: true,

			ExpectOK:  false,
			ExpectErr: false,
		},

		"tls, no certs": {
			ExpectOK:  false,
			ExpectErr: false,
		},

		"self signed": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, selfSignedCert),
			User:  CommonNameUserConversion,

			ExpectErr: true,
		},

		"server cert": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, serverCert),
			User:  CommonNameUserConversion,

			ExpectErr: true,
		},
		"server cert allowing non-client cert usages": {
			Opts:  x509.VerifyOptions{Roots: getRootCertPool(t)},
			Certs: getCerts(t, serverCert),
			User:  CommonNameUserConversion,

			ExpectUserName: "127.0.0.1",
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"common name": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User:  CommonNameUserConversion,

			ExpectUserName: "client_cn",
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"empty dns": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User:  DNSNameUserConversion,

			ExpectOK:  false,
			ExpectErr: false,
		},
		"dns": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientDNSCert),
			User:  DNSNameUserConversion,

			ExpectUserName: "client_dns.example.com",
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"empty email": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User:  EmailAddressUserConversion,

			ExpectOK:  false,
			ExpectErr: false,
		},
		"email": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientEmailCert),
			User:  EmailAddressUserConversion,

			ExpectUserName: "client_email@example.com",
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"custom conversion error": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User: UserConversionFunc(func(chain []*x509.Certificate) (user.Info, bool, error) {
				return nil, false, errors.New("custom error")
			}),

			ExpectOK:  false,
			ExpectErr: true,
		},
		"custom conversion success": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User: UserConversionFunc(func(chain []*x509.Certificate) (user.Info, bool, error) {
				return &user.DefaultInfo{Name: "custom"}, true, nil
			}),

			ExpectUserName: "custom",
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"future cert": {
			Opts: x509.VerifyOptions{
				CurrentTime: time.Now().Add(time.Duration(-100 * time.Hour * 24 * 365)),
				Roots:       getRootCertPool(t),
			},
			Certs: getCerts(t, clientCNCert),
			User:  CommonNameUserConversion,

			ExpectOK:  false,
			ExpectErr: true,
		},
		"expired cert": {
			Opts: x509.VerifyOptions{
				CurrentTime: time.Now().Add(time.Duration(100 * time.Hour * 24 * 365)),
				Roots:       getRootCertPool(t),
			},
			Certs: getCerts(t, clientCNCert),
			User:  CommonNameUserConversion,

			ExpectOK:  false,
			ExpectErr: true,
		},
	}

	for k, testCase := range testCases {
		req, _ := http.NewRequest("GET", "/", nil)
		if !testCase.Insecure {
			req.TLS = &tls.ConnectionState{PeerCertificates: testCase.Certs}
		}

		a := New(testCase.Opts, testCase.User)

		user, ok, err := a.AuthenticateRequest(req)

		if testCase.ExpectErr && err == nil {
			t.Errorf("%s: Expected error, got none", k)
			continue
		}
		if !testCase.ExpectErr && err != nil {
			t.Errorf("%s: Got unexpected error: %v", k, err)
			continue
		}

		if testCase.ExpectOK != ok {
			t.Errorf("%s: Expected ok=%v, got %v", k, testCase.ExpectOK, ok)
			continue
		}

		if testCase.ExpectOK {
			if testCase.ExpectUserName != user.GetName() {
				t.Errorf("%s: Expected user.name=%v, got %v", k, testCase.ExpectUserName, user.GetName())
				continue
			}
		}
	}
}

func getDefaultVerifyOptions(t *testing.T) x509.VerifyOptions {
	options := DefaultVerifyOptions()
	options.Roots = getRootCertPool(t)
	return options
}

func getRootCertPool(t *testing.T) *x509.CertPool {
	pool := x509.NewCertPool()
	pool.AddCert(getCert(t, rootCACert))
	return pool
}

func getCert(t *testing.T, pemData string) *x509.Certificate {
	pemBlock, _ := pem.Decode([]byte(pemData))
	cert, err := x509.ParseCertificate(pemBlock.Bytes)
	if err != nil {
		t.Fatalf("Error parsing cert: %v", err)
		return nil
	}
	return cert
}

func getCerts(t *testing.T, pemData ...string) []*x509.Certificate {
	certs := make([]*x509.Certificate, 0)
	for _, pemData := range pemData {
		certs = append(certs, getCert(t, pemData))
	}
	return certs
}
