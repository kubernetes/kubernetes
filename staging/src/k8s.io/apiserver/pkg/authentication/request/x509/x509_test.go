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
	"io/ioutil"
	"net/http"
	"reflect"
	"sort"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
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

	/*
	   openssl genrsa -out ca.key 4096
	   openssl req -new -x509 -days 36500 \
	       -sha256 -key ca.key -extensions v3_ca \
	       -out ca.crt \
	       -subj "/C=US/ST=My State/L=My City/O=My Org/O=My Org 1/O=My Org 2/CN=ROOT CA WITH GROUPS"
	   openssl x509 -in ca.crt -text
	*/

	// A certificate with multiple organizations.
	caWithGroups = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            bc:57:6d:0b:7c:ff:cb:52
    Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, O=My Org 1, O=My Org 2, CN=ROOT CA WITH GROUPS
        Validity
            Not Before: Aug 10 19:22:03 2016 GMT
            Not After : Jul 17 19:22:03 2116 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, O=My Org 1, O=My Org 2, CN=ROOT CA WITH GROUPS
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (4096 bit)
                Modulus:
                    00:ba:3a:40:34:1a:ba:13:87:0d:c9:c7:bf:e5:8e:
                    6a:c7:d5:0f:8f:e3:e1:ac:9e:a5:fd:35:e1:39:52:
                    1d:22:77:c1:d2:3f:74:02:2e:23:c6:c1:fc:cd:30:
                    b4:33:e7:12:04:6f:90:27:e1:be:8e:ec:c8:dc:87:
                    91:da:7d:5b:8a:1f:41:fb:62:24:d0:26:98:c6:f7:
                    f8:ca:8a:56:15:c4:b3:5f:43:86:28:f6:4d:fc:e4:
                    03:52:1d:2b:25:f7:19:5c:13:c3:0e:04:91:06:f3:
                    29:b6:3f:8b:86:6d:b5:8e:43:2d:69:4e:60:53:5b:
                    75:8f:e7:d2:57:8c:db:bb:a1:0b:d7:c7:62:41:bc:
                    f2:87:be:66:bb:b9:bf:8b:85:97:19:98:18:50:7b:
                    ee:31:88:47:99:c1:04:e4:12:d2:a6:e2:bf:61:33:
                    82:11:79:c3:d5:39:7c:1c:15:9e:d2:61:f7:16:9f:
                    97:f1:39:05:8f:b9:f8:e0:5b:16:ca:da:bf:10:45:
                    10:0f:14:f9:67:10:66:77:05:f3:fe:21:d6:69:fb:
                    1e:dc:fd:f7:97:40:db:0d:59:99:8a:9d:e4:31:a3:
                    b9:c2:4d:ff:85:ae:ea:da:18:d8:c7:a5:b7:ea:f3:
                    a8:38:a5:44:1f:3b:23:71:fc:4c:5b:bd:36:6f:e0:
                    28:6d:f3:be:e8:c9:74:64:af:89:54:b3:12:c8:2d:
                    27:2d:1c:22:23:81:bd:69:b7:8b:76:63:e1:bf:80:
                    a1:ba:d6:c6:fc:aa:37:2e:44:94:4b:4c:3f:c4:f2:
                    c3:f8:25:54:ab:1f:0f:4c:19:2f:9c:b6:46:09:db:
                    26:52:b4:03:0a:35:75:53:94:33:5d:22:29:48:4a:
                    61:9c:d0:5a:6d:91:f5:18:bb:93:99:30:02:5c:6d:
                    7c:3f:4d:5a:ea:6f:ee:f7:7a:f9:07:9d:fe:e0:6f:
                    75:02:4a:ef:1e:25:c2:d5:8d:2c:57:a2:95:a7:df:
                    37:4f:32:60:94:09:85:4d:a7:67:05:e9:29:db:45:
                    a8:89:ec:1e:e9:3a:49:92:23:17:5b:4a:9c:b8:0d:
                    6f:2a:54:ba:47:45:f8:d3:34:30:e8:db:48:6d:c7:
                    82:08:01:d5:93:6a:08:7c:4b:43:78:04:df:57:b7:
                    fe:e3:d7:4c:ec:9c:dc:2d:0b:8c:e4:6f:aa:e2:30:
                    66:74:16:10:b9:44:c9:1e:73:53:86:25:25:cc:60:
                    3a:94:79:18:f1:c9:31:b0:e1:ca:b9:21:44:75:0a:
                    6c:e4:58:c1:37:ee:69:28:d1:d4:b8:78:21:64:ea:
                    27:d3:67:25:cf:3a:82:8d:de:27:51:b4:33:a2:85:
                    db:07:89
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Subject Key Identifier: 
                AB:3A:46:07:46:0C:68:F0:64:C7:73:A8:7C:8A:20:66:A8:DA:1C:E4
            X509v3 Authority Key Identifier: 
                keyid:AB:3A:46:07:46:0C:68:F0:64:C7:73:A8:7C:8A:20:66:A8:DA:1C:E4

            X509v3 Basic Constraints: 
                CA:TRUE
    Signature Algorithm: sha256WithRSAEncryption
         1c:af:04:c9:10:f2:43:03:b6:24:2e:20:2e:47:46:4d:7f:b9:
         fa:1c:ea:8d:f0:30:a5:42:93:fe:e0:55:4a:b5:8b:4d:30:f4:
         e1:04:1f:20:ec:a1:27:ab:1f:b2:9d:da:58:2e:04:5c:b6:7c:
         69:8c:00:59:42:4f:cc:c7:3c:d4:f7:30:84:2a:14:8e:5d:3a:
         20:91:63:5c:ac:5c:e7:0c:78:fc:28:f3:f9:24:de:3d:30:e3:
         64:ca:5d:a6:86:30:76:5e:53:a4:99:77:a4:7a:c5:52:62:cd:
         f9:79:42:69:57:1b:79:25:c5:51:45:42:ed:ae:9c:bc:f2:4c:
         4d:9d:3a:17:73:b1:d2:94:ab:61:4a:90:fa:59:f1:96:c7:7c:
         26:5b:0c:75:4b:94:6f:76:ac:6c:70:8f:68:5c:e3:e7:7b:b9:
         38:c2:0f:f2:e3:2d:96:ec:79:fa:bf:df:33:02:f2:67:a1:19:
         d1:7d:ed:c4:3b:14:b8:1f:53:c5:6a:52:ad:19:2d:4c:43:19:
         c7:d3:14:75:7f:e7:18:40:38:79:b7:2c:ce:91:6f:cd:16:e3:
         d9:8f:87:be:bc:c0:c0:53:1a:93:d6:ff:a9:17:c0:d9:6f:6a:
         cc:0b:57:37:b8:da:30:98:4a:fc:e5:e9:dc:49:1a:33:35:f0:
         e9:9a:a7:a2:fd:6a:13:9e:85:df:66:a8:15:3f:94:30:4b:ca:
         61:72:7e:1a:b1:83:88:65:21:e8:f6:58:4a:22:48:b5:29:3d:
         00:6c:3e:a2:e5:bd:a5:a3:d9:5a:4d:a9:cb:2a:f8:47:ca:72:
         ea:9d:e1:87:e1:d1:75:5d:07:36:ba:ab:fd:7f:5f:d3:66:d0:
         41:86:7c:6b:1e:a7:7c:9f:dc:26:7a:37:70:54:1e:7c:b3:66:
         7f:f1:99:93:f4:8a:aa:81:02:e9:bf:5d:a5:90:94:82:6e:2a:
         a6:c8:e1:77:df:66:59:d8:6c:b1:55:a0:77:d6:53:6b:78:aa:
         4b:0d:fc:34:06:5c:52:4e:e6:5e:c7:94:13:19:70:e8:2b:00:
         6d:ea:90:b9:f4:6f:74:3f:cc:e7:1d:3e:22:ec:66:cb:84:19:
         7a:40:3c:7e:38:77:b4:4e:da:8c:4b:af:dc:c2:23:28:9d:60:
         a5:4f:5a:c8:9e:17:df:b9:9d:92:bc:d3:c0:20:12:ec:22:d4:
         e8:d4:97:9f:da:3c:35:a0:e9:a3:8c:d1:42:7c:c1:27:1f:8a:
         9b:5b:03:3d:2b:9b:df:25:b6:a8:a7:5a:48:0f:e8:1f:26:4b:
         0e:3c:a2:50:0a:cd:02:33:4c:e4:7a:c9:2d:b8:b8:bf:80:5a:
         6e:07:49:c4:c3:23:a0:2e
-----BEGIN CERTIFICATE-----
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
PKJQCs0CM0zkesktuLi/gFpuB0nEwyOgLg==
-----END CERTIFICATE-----`
)

func TestX509(t *testing.T) {
	multilevelOpts := DefaultVerifyOptions()
	multilevelOpts.Roots = x509.NewCertPool()
	multilevelOpts.Roots.AddCert(getCertsFromFile(t, "root")[0])

	testCases := map[string]struct {
		Insecure bool
		Certs    []*x509.Certificate

		Opts x509.VerifyOptions
		User UserConversion

		ExpectUserName string
		ExpectGroups   []string
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
			ExpectGroups:   []string{"My Org"},
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"common name": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User:  CommonNameUserConversion,

			ExpectUserName: "client_cn",
			ExpectGroups:   []string{"My Org"},
			ExpectOK:       true,
			ExpectErr:      false,
		},
		"ca with multiple organizations": {
			Opts: x509.VerifyOptions{
				Roots: getRootCertPoolFor(t, caWithGroups),
			},
			Certs: getCerts(t, caWithGroups),
			User:  CommonNameUserConversion,

			ExpectUserName: "ROOT CA WITH GROUPS",
			ExpectGroups:   []string{"My Org", "My Org 1", "My Org 2"},
			ExpectOK:       true,
			ExpectErr:      false,
		},

		"custom conversion error": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User: UserConversionFunc(func(chain []*x509.Certificate) (*authenticator.Response, bool, error) {
				return nil, false, errors.New("custom error")
			}),

			ExpectOK:  false,
			ExpectErr: true,
		},
		"custom conversion success": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User: UserConversionFunc(func(chain []*x509.Certificate) (*authenticator.Response, bool, error) {
				return &authenticator.Response{User: &user.DefaultInfo{Name: "custom"}}, true, nil
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

		"multi-level, valid": {
			Opts:  multilevelOpts,
			Certs: getCertsFromFile(t, "client-valid", "intermediate"),
			User:  CommonNameUserConversion,

			ExpectUserName: "My Client",
			ExpectOK:       true,
			ExpectErr:      false,
		},
		"multi-level, expired": {
			Opts:  multilevelOpts,
			Certs: getCertsFromFile(t, "client-expired", "intermediate"),
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

		// this effectively tests the simple dynamic verify function.
		a := New(testCase.Opts, testCase.User)

		resp, ok, err := a.AuthenticateRequest(req)

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
			if testCase.ExpectUserName != resp.User.GetName() {
				t.Errorf("%s: Expected user.name=%v, got %v", k, testCase.ExpectUserName, resp.User.GetName())
			}

			groups := resp.User.GetGroups()
			sort.Strings(testCase.ExpectGroups)
			sort.Strings(groups)
			if !reflect.DeepEqual(testCase.ExpectGroups, groups) {
				t.Errorf("%s: Expected user.groups=%v, got %v", k, testCase.ExpectGroups, groups)
			}
		}
	}
}

func TestX509Verifier(t *testing.T) {
	multilevelOpts := DefaultVerifyOptions()
	multilevelOpts.Roots = x509.NewCertPool()
	multilevelOpts.Roots.AddCert(getCertsFromFile(t, "root")[0])

	testCases := map[string]struct {
		Insecure bool
		Certs    []*x509.Certificate

		Opts x509.VerifyOptions

		AllowedCNs sets.String

		ExpectOK  bool
		ExpectErr bool
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

			ExpectErr: true,
		},

		"server cert disallowed": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, serverCert),

			ExpectErr: true,
		},
		"server cert allowing non-client cert usages": {
			Opts:  x509.VerifyOptions{Roots: getRootCertPool(t)},
			Certs: getCerts(t, serverCert),

			ExpectOK:  true,
			ExpectErr: false,
		},

		"valid client cert": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),

			ExpectOK:  true,
			ExpectErr: false,
		},
		"valid client cert with wrong CN": {
			Opts:       getDefaultVerifyOptions(t),
			AllowedCNs: sets.NewString("foo", "bar"),
			Certs:      getCerts(t, clientCNCert),

			ExpectOK:  false,
			ExpectErr: true,
		},
		"valid client cert with right CN": {
			Opts:       getDefaultVerifyOptions(t),
			AllowedCNs: sets.NewString("client_cn"),
			Certs:      getCerts(t, clientCNCert),

			ExpectOK:  true,
			ExpectErr: false,
		},

		"future cert": {
			Opts: x509.VerifyOptions{
				CurrentTime: time.Now().Add(-100 * time.Hour * 24 * 365),
				Roots:       getRootCertPool(t),
			},
			Certs: getCerts(t, clientCNCert),

			ExpectOK:  false,
			ExpectErr: true,
		},
		"expired cert": {
			Opts: x509.VerifyOptions{
				CurrentTime: time.Now().Add(100 * time.Hour * 24 * 365),
				Roots:       getRootCertPool(t),
			},
			Certs: getCerts(t, clientCNCert),

			ExpectOK:  false,
			ExpectErr: true,
		},

		"multi-level, valid": {
			Opts:  multilevelOpts,
			Certs: getCertsFromFile(t, "client-valid", "intermediate"),

			ExpectOK:  true,
			ExpectErr: false,
		},
		"multi-level, expired": {
			Opts:  multilevelOpts,
			Certs: getCertsFromFile(t, "client-expired", "intermediate"),

			ExpectOK:  false,
			ExpectErr: true,
		},
	}

	for k, testCase := range testCases {
		req, _ := http.NewRequest("GET", "/", nil)
		if !testCase.Insecure {
			req.TLS = &tls.ConnectionState{PeerCertificates: testCase.Certs}
		}

		authCall := false
		auth := authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			authCall = true
			return &authenticator.Response{User: &user.DefaultInfo{Name: "innerauth"}}, true, nil
		})

		a := NewVerifier(testCase.Opts, auth, testCase.AllowedCNs)

		resp, ok, err := a.AuthenticateRequest(req)

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
			if !authCall {
				t.Errorf("%s: Expected inner auth called, wasn't", k)
				continue
			}
			if "innerauth" != resp.User.GetName() {
				t.Errorf("%s: Expected user.name=%v, got %v", k, "innerauth", resp.User.GetName())
				continue
			}
		} else {
			if authCall {
				t.Errorf("%s: Expected inner auth not to be called, was", k)
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
	return getRootCertPoolFor(t, rootCACert)
}

func getRootCertPoolFor(t *testing.T, certs ...string) *x509.CertPool {
	pool := x509.NewCertPool()
	for _, cert := range certs {
		pool.AddCert(getCert(t, cert))
	}
	return pool
}

func getCertsFromFile(t *testing.T, names ...string) []*x509.Certificate {
	certs := []*x509.Certificate{}
	for _, name := range names {
		filename := "testdata/" + name + ".pem"
		data, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatalf("error reading %s: %v", filename, err)
		}
		certs = append(certs, getCert(t, string(data)))
	}
	return certs
}

func getCert(t *testing.T, pemData string) *x509.Certificate {
	t.Helper()

	pemBlock, _ := pem.Decode([]byte(pemData))
	cert, err := x509.ParseCertificate(pemBlock.Bytes)
	if err != nil {
		t.Fatalf("Error parsing cert: %v", err)
		return nil
	}
	return cert
}

func getCerts(t *testing.T, pemData ...string) []*x509.Certificate {
	certs := []*x509.Certificate{}
	for _, pemData := range pemData {
		certs = append(certs, getCert(t, pemData))
	}
	return certs
}

func TestCertificateIdentifier(t *testing.T) {
	tt := []struct {
		name               string
		cert               *x509.Certificate
		expectedIdentifier string
	}{
		{
			name:               "client cert",
			cert:               getCert(t, clientCNCert),
			expectedIdentifier: "SN=1, SKID=E7:FB:1F:45:F0:71:77:AF:8C:10:4A:0A:42:03:F5:1F:1F:07:CF:DF, AKID=3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78",
		},
		{
			name: "nil serial",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.SerialNumber = nil
				return c
			}(),
			expectedIdentifier: "SN=<nil>, SKID=E7:FB:1F:45:F0:71:77:AF:8C:10:4A:0A:42:03:F5:1F:1F:07:CF:DF, AKID=3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78",
		},
		{
			name: "empty SKID",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.SubjectKeyId = nil
				return c
			}(),
			expectedIdentifier: "SN=1, SKID=, AKID=3D:F0:F7:30:3D:3B:EB:3A:55:68:FA:F5:43:C9:C7:AC:E1:3F:10:78",
		},
		{
			name: "empty AKID",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.AuthorityKeyId = nil
				return c
			}(),
			expectedIdentifier: "SN=1, SKID=E7:FB:1F:45:F0:71:77:AF:8C:10:4A:0A:42:03:F5:1F:1F:07:CF:DF, AKID=",
		},
		{
			name:               "self-signed",
			cert:               getCert(t, selfSignedCert),
			expectedIdentifier: "SN=14307769263086146430, SKID=7C:AB:02:A8:45:3F:B0:28:2F:71:91:52:A2:71:EE:D9:40:2B:43:71, AKID=7C:AB:02:A8:45:3F:B0:28:2F:71:91:52:A2:71:EE:D9:40:2B:43:71",
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			got := certificateIdentifier(tc.cert)
			if got != tc.expectedIdentifier {
				t.Errorf("expected %q, got %q", tc.expectedIdentifier, got)
			}
		})
	}
}
