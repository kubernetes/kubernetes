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
	"os"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

const (

	/*

	   > rootCACert

	   openssl genrsa -out root.key 1024 && \
	   openssl rsa -in ./root.key -outform PEM -pubout -out ./root.pub && \
	   CONFIG="[ v3_req ]\n" && \
	   CONFIG="${CONFIG}subjectKeyIdentifier=hash\n" && \
	   CONFIG="${CONFIG}authorityKeyIdentifier=keyid:always,issuer\n" && \
	   CONFIG="${CONFIG}basicConstraints=CA:TRUE\n" && \
	   CONFIG="${CONFIG}keyUsage=keyCertSign,cRLSign\n" && \
	   openssl req -new -x509 -days 36500 \
	   	-sha1 -key root.key \
	   	-out root.crt \
	   	-subj "/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=ROOT CA" \
	   	-config <(printf "${CONFIG}") \
	   	-extensions v3_req \
	   	&& \
	   openssl x509 -in root.crt -text


	   > output

	   Certificate:
	       Data:
	           Version: 3 (0x2)
	           Serial Number:
	               2d:73:1a:2e:d7:8b:89:20:83:9c:42:9a:6e:f7:f5:f6:a1:ec:af:8c
	           Signature Algorithm: sha1WithRSAEncryption
	           Issuer: C = US, ST = My State, L = My City, O = My Org, OU = My Unit, CN = ROOT CA
	           Validity
	               Not Before: May  2 05:43:51 2024 GMT
	               Not After : Apr  8 05:43:51 2124 GMT
	           Subject: C = US, ST = My State, L = My City, O = My Org, OU = My Unit, CN = ROOT CA
	           Subject Public Key Info:
	               Public Key Algorithm: rsaEncryption
	                   Public-Key: (1024 bit)
	                   Modulus:
	                       00:a8:c3:dc:de:1a:f6:3e:95:97:2a:d5:bf:8b:72:
	                       93:06:85:72:4b:36:2a:d9:63:a8:9c:fb:80:3e:9b:
	                       2f:84:c6:57:d2:ff:33:13:bf:32:e9:90:66:db:0a:
	                       9a:05:c1:e3:c1:09:bb:25:75:b2:d7:fc:9c:09:86:
	                       80:15:b0:6c:67:c5:1a:e9:76:01:32:40:22:58:ec:
	                       4e:a1:b7:c5:05:01:49:55:d8:4f:4b:88:1d:bf:66:
	                       d3:de:58:4a:e7:26:b6:bf:af:33:d8:57:42:f1:bc:
	                       34:67:44:88:b4:31:f6:4a:4a:b3:1e:c2:ca:6b:4b:
	                       2e:5a:32:23:9b:1b:3f:97:35
	                   Exponent: 65537 (0x10001)
	           X509v3 extensions:
	               X509v3 Subject Key Identifier:
	                   D3:07:CD:72:E6:BE:0A:5A:D8:E9:60:20:AF:C2:F2:36:7E:33:62:0B
	               X509v3 Authority Key Identifier:
	                   D3:07:CD:72:E6:BE:0A:5A:D8:E9:60:20:AF:C2:F2:36:7E:33:62:0B
	               X509v3 Basic Constraints:
	                   CA:TRUE
	               X509v3 Key Usage:
	                   Certificate Sign, CRL Sign
	       Signature Algorithm: sha1WithRSAEncryption
	       Signature Value:
	           4a:54:07:46:71:c1:b2:a2:d3:32:e7:df:49:8c:af:87:46:ab:
	           81:11:c6:c5:4b:be:0b:0c:ea:7e:5f:38:14:79:43:92:f9:bb:
	           82:6f:f6:06:a6:43:19:e2:7c:52:66:36:13:6f:0f:73:16:3d:
	           79:5f:f9:a6:c8:4c:18:f9:ff:20:2b:de:7f:15:e0:ab:ae:44:
	           fa:65:7a:86:8a:df:d0:63:82:b1:5c:f3:f8:5c:05:97:4e:1f:
	           09:d6:d9:55:e7:36:fc:08:3e:3f:66:99:68:b6:31:44:0f:63:
	           20:6a:b2:81:50:39:19:d0:47:de:20:94:f0:a2:2c:eb:69:93:
	           93:a3
	   -----BEGIN CERTIFICATE-----
	   MIICtjCCAh+gAwIBAgIULXMaLteLiSCDnEKabvf19qHsr4wwDQYJKoZIhvcNAQEF
	   BQAwZzELMAkGA1UEBhMCVVMxETAPBgNVBAgMCE15IFN0YXRlMRAwDgYDVQQHDAdN
	   eSBDaXR5MQ8wDQYDVQQKDAZNeSBPcmcxEDAOBgNVBAsMB015IFVuaXQxEDAOBgNV
	   BAMMB1JPT1QgQ0EwIBcNMjQwNTAyMDU0MzUxWhgPMjEyNDA0MDgwNTQzNTFaMGcx
	   CzAJBgNVBAYTAlVTMREwDwYDVQQIDAhNeSBTdGF0ZTEQMA4GA1UEBwwHTXkgQ2l0
	   eTEPMA0GA1UECgwGTXkgT3JnMRAwDgYDVQQLDAdNeSBVbml0MRAwDgYDVQQDDAdS
	   T09UIENBMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCow9zeGvY+lZcq1b+L
	   cpMGhXJLNirZY6ic+4A+my+ExlfS/zMTvzLpkGbbCpoFwePBCbsldbLX/JwJhoAV
	   sGxnxRrpdgEyQCJY7E6ht8UFAUlV2E9LiB2/ZtPeWErnJra/rzPYV0LxvDRnRIi0
	   MfZKSrMewsprSy5aMiObGz+XNQIDAQABo10wWzAdBgNVHQ4EFgQU0wfNcua+ClrY
	   6WAgr8LyNn4zYgswHwYDVR0jBBgwFoAU0wfNcua+ClrY6WAgr8LyNn4zYgswDAYD
	   VR0TBAUwAwEB/zALBgNVHQ8EBAMCAQYwDQYJKoZIhvcNAQEFBQADgYEASlQHRnHB
	   sqLTMuffSYyvh0argRHGxUu+Cwzqfl84FHlDkvm7gm/2BqZDGeJ8UmY2E28PcxY9
	   eV/5pshMGPn/ICvefxXgq65E+mV6horf0GOCsVzz+FwFl04fCdbZVec2/Ag+P2aZ
	   aLYxRA9jIGqygVA5GdBH3iCU8KIs62mTk6M=
	   -----END CERTIFICATE-----


	*/

	rootCACert = `-----BEGIN CERTIFICATE-----
MIICtjCCAh+gAwIBAgIUXipc16GmHC8Q64wKx+gegIcA0wAwDQYJKoZIhvcNAQEF
BQAwZzELMAkGA1UEBhMCVVMxETAPBgNVBAgMCE15IFN0YXRlMRAwDgYDVQQHDAdN
eSBDaXR5MQ8wDQYDVQQKDAZNeSBPcmcxEDAOBgNVBAsMB015IFVuaXQxEDAOBgNV
BAMMB1JPT1QgQ0EwIBcNMjQxMDA2MjAzNTIwWhgPMjEyNDA5MTIyMDM1MjBaMGcx
CzAJBgNVBAYTAlVTMREwDwYDVQQIDAhNeSBTdGF0ZTEQMA4GA1UEBwwHTXkgQ2l0
eTEPMA0GA1UECgwGTXkgT3JnMRAwDgYDVQQLDAdNeSBVbml0MRAwDgYDVQQDDAdS
T09UIENBMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCt8DHt/ni9y/6lqWss
uv2eFvW6N9RvYhxRmuuxQK74F5/VRAfhEMvDOU+woG/HBXMyPOgLL1uWt4dk3DGu
WYNwYP2oN6D04KkWYgcxwYFjcduzWxynr5zT1T2B3bxZFMkvqshyrHWD38Vge080
NU3Pns7Z53AZu673srH+OSU8WwIDAQABo10wWzAdBgNVHQ4EFgQUSHB11O1rSTtT
2+mm+ZxVklG9luYwHwYDVR0jBBgwFoAUSHB11O1rSTtT2+mm+ZxVklG9luYwDAYD
VR0TBAUwAwEB/zALBgNVHQ8EBAMCAQYwDQYJKoZIhvcNAQEFBQADgYEAj/zGCbq+
POo9thqGg2i2/bzHzAr4X9ylJaeM8oaBhk0pvliTcWGb/usjqwWpcXIqHY8jjBrN
GFJEH6elL1Q63W+JCwWS14i2jQExjPk7/AWLBv/J7XqgiUhPfF/P9iQp+lGcInNR
6TGXeFKLtsrySVfQ4TvEW1zNJj9qJ819YwU=
-----END CERTIFICATE-----
`

	/*

	   > selfSignedCert

	   openssl genrsa -out selfsign.key 1024 && \
	   openssl req -new -x509 -days 36500 \
	   	-sha1 -key selfsign.key  \
	   	-out selfsign.crt \
	   	-subj "/C=US/ST=My State/L=My City/O=My Org/O=My Unit/CN=self1" \
	   	&& \
	   openssl x509 -in selfsign.crt -text


	   > output

	   Certificate:
	       Data:
	           Version: 3 (0x2)
	           Serial Number:
	               72:ae:28:f9:b7:7f:16:0a:89:a7:9c:a1:a3:88:15:4b:20:eb:f5:b2
	           Signature Algorithm: sha1WithRSAEncryption
	           Issuer: C = US, ST = My State, L = My City, O = My Org, O = My Unit, CN = self1
	           Validity
	               Not Before: May  2 00:25:12 2024 GMT
	               Not After : Apr  8 00:25:12 2124 GMT
	           Subject: C = US, ST = My State, L = My City, O = My Org, O = My Unit, CN = self1
	           Subject Public Key Info:
	               Public Key Algorithm: rsaEncryption
	                   Public-Key: (1024 bit)
	                   Modulus:
	                       00:94:91:e3:8a:4d:dd:f6:27:e9:71:9c:d2:f2:64:
	                       b9:af:ce:05:9d:82:a2:98:a9:15:40:8b:ff:a2:5c:
	                       72:53:e8:d0:af:73:c6:76:4d:c7:6a:6e:9f:5d:a7:
	                       e2:f6:aa:6a:18:2b:c3:ee:3b:64:19:16:5d:94:0b:
	                       f2:f7:90:43:9a:5d:ce:7e:07:4d:b9:df:be:f0:39:
	                       98:a4:41:eb:d3:17:90:12:d9:bc:d7:7f:a4:66:98:
	                       c3:91:17:30:5d:7b:c4:12:2b:a9:a9:48:ca:a3:14:
	                       3a:36:ad:23:58:cf:88:b9:30:9a:b4:e6:8a:35:a1:
	                       ce:80:02:4a:aa:24:2b:7b:79
	                   Exponent: 65537 (0x10001)
	           X509v3 extensions:
	               X509v3 Subject Key Identifier:
	                   56:A5:55:02:8C:97:FD:1E:A0:B8:DE:EF:5E:95:F0:AC:A6:23:6F:16
	               X509v3 Authority Key Identifier:
	                   56:A5:55:02:8C:97:FD:1E:A0:B8:DE:EF:5E:95:F0:AC:A6:23:6F:16
	               X509v3 Basic Constraints: critical
	                   CA:TRUE
	       Signature Algorithm: sha1WithRSAEncryption
	       Signature Value:
	           5e:84:19:68:a2:f3:41:c5:f5:57:2f:1b:e5:14:4d:8c:50:ee:
	           5f:f4:aa:ec:4f:6a:06:4b:af:f3:2a:14:cc:0f:7b:a1:17:de:
	           cc:da:f8:fb:c3:04:c7:a7:60:98:76:5c:32:82:5c:ec:95:a0:
	           51:74:12:12:c0:7a:8b:68:bc:8b:47:47:db:95:20:34:be:69:
	           d2:fc:d5:d7:e7:4b:7c:e1:f3:bc:72:3c:b1:f5:d4:db:71:ad:
	           d8:a7:ad:ab:91:68:c9:16:0a:e9:76:ed:87:0f:83:24:cd:ab:
	           c7:a4:16:3f:c6:7c:99:18:bb:b1:12:11:a4:a5:99:af:17:11:
	           e7:b1
	   -----BEGIN CERTIFICATE-----
	   MIICqDCCAhGgAwIBAgIUcq4o+bd/FgqJp5yho4gVSyDr9bIwDQYJKoZIhvcNAQEF
	   BQAwZTELMAkGA1UEBhMCVVMxETAPBgNVBAgMCE15IFN0YXRlMRAwDgYDVQQHDAdN
	   eSBDaXR5MQ8wDQYDVQQKDAZNeSBPcmcxEDAOBgNVBAoMB015IFVuaXQxDjAMBgNV
	   BAMMBXNlbGYxMCAXDTI0MDUwMjAwMjUxMloYDzIxMjQwNDA4MDAyNTEyWjBlMQsw
	   CQYDVQQGEwJVUzERMA8GA1UECAwITXkgU3RhdGUxEDAOBgNVBAcMB015IENpdHkx
	   DzANBgNVBAoMBk15IE9yZzEQMA4GA1UECgwHTXkgVW5pdDEOMAwGA1UEAwwFc2Vs
	   ZjEwgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGBAJSR44pN3fYn6XGc0vJkua/O
	   BZ2CopipFUCL/6JcclPo0K9zxnZNx2pun12n4vaqahgrw+47ZBkWXZQL8veQQ5pd
	   zn4HTbnfvvA5mKRB69MXkBLZvNd/pGaYw5EXMF17xBIrqalIyqMUOjatI1jPiLkw
	   mrTmijWhzoACSqokK3t5AgMBAAGjUzBRMB0GA1UdDgQWBBRWpVUCjJf9HqC43u9e
	   lfCspiNvFjAfBgNVHSMEGDAWgBRWpVUCjJf9HqC43u9elfCspiNvFjAPBgNVHRMB
	   Af8EBTADAQH/MA0GCSqGSIb3DQEBBQUAA4GBAF6EGWii80HF9VcvG+UUTYxQ7l/0
	   quxPagZLr/MqFMwPe6EX3sza+PvDBMenYJh2XDKCXOyVoFF0EhLAeotovItHR9uV
	   IDS+adL81dfnS3zh87xyPLH11NtxrdinrauRaMkWCul27YcPgyTNq8ekFj/GfJkY
	   u7ESEaSlma8XEeex
	   -----END CERTIFICATE-----


	*/

	selfSignedCert = `-----BEGIN CERTIFICATE-----
MIICqDCCAhGgAwIBAgIUcq4o+bd/FgqJp5yho4gVSyDr9bIwDQYJKoZIhvcNAQEF
BQAwZTELMAkGA1UEBhMCVVMxETAPBgNVBAgMCE15IFN0YXRlMRAwDgYDVQQHDAdN
eSBDaXR5MQ8wDQYDVQQKDAZNeSBPcmcxEDAOBgNVBAoMB015IFVuaXQxDjAMBgNV
BAMMBXNlbGYxMCAXDTI0MDUwMjAwMjUxMloYDzIxMjQwNDA4MDAyNTEyWjBlMQsw
CQYDVQQGEwJVUzERMA8GA1UECAwITXkgU3RhdGUxEDAOBgNVBAcMB015IENpdHkx
DzANBgNVBAoMBk15IE9yZzEQMA4GA1UECgwHTXkgVW5pdDEOMAwGA1UEAwwFc2Vs
ZjEwgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGBAJSR44pN3fYn6XGc0vJkua/O
BZ2CopipFUCL/6JcclPo0K9zxnZNx2pun12n4vaqahgrw+47ZBkWXZQL8veQQ5pd
zn4HTbnfvvA5mKRB69MXkBLZvNd/pGaYw5EXMF17xBIrqalIyqMUOjatI1jPiLkw
mrTmijWhzoACSqokK3t5AgMBAAGjUzBRMB0GA1UdDgQWBBRWpVUCjJf9HqC43u9e
lfCspiNvFjAfBgNVHSMEGDAWgBRWpVUCjJf9HqC43u9elfCspiNvFjAPBgNVHRMB
Af8EBTADAQH/MA0GCSqGSIb3DQEBBQUAA4GBAF6EGWii80HF9VcvG+UUTYxQ7l/0
quxPagZLr/MqFMwPe6EX3sza+PvDBMenYJh2XDKCXOyVoFF0EhLAeotovItHR9uV
IDS+adL81dfnS3zh87xyPLH11NtxrdinrauRaMkWCul27YcPgyTNq8ekFj/GfJkY
u7ESEaSlma8XEeex
-----END CERTIFICATE-----
`

	/*

	   > clientCNCert

	   openssl genrsa -out client.key 1024 && \
	   openssl rsa -in ./client.key -outform PEM \
	   	-pubout -out ./client.pub && \
	   openssl req -key ./client.key -new\
	          	-sha1 -out ./client.csr \
	          	-subj "/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=client_cn/UID=test123" \
	   	&& \
	   EXTFILE="subjectKeyIdentifier=hash\n" && \
	   EXTFILE="${EXTFILE}authorityKeyIdentifier=keyid,issuer\n" && \
	   EXTFILE="${EXTFILE}basicConstraints=CA:FALSE\n" && \
	   EXTFILE="${EXTFILE}subjectAltName=email:copy\n" && \
	   EXTFILE="${EXTFILE}extendedKeyUsage=clientAuth\n" && \
	   openssl  x509 -req -days 36500 \
	   	-in ./client.csr \
	   	-extfile <(printf "${EXTFILE}") \
	   	-CA ./root.crt \
	   	-CAkey ./root.key \
	   	-set_serial 1 \
	          	-sha256 \
	   	-out ./client.crt \
	   	&& \
	   openssl x509 -in client.crt -text

	   > output

	   is below

	*/

	clientCNCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 1 (0x1)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Oct  6 20:50:43 2024 GMT
            Not After : Sep 12 20:50:43 2124 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=client_cn, UID=test123
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:c3:5e:95:bd:b9:3e:19:bb:eb:06:ac:72:9c:37:
                    c3:7f:dd:2a:c1:da:be:d2:c1:a7:18:9c:cc:3c:2d:
                    31:af:b2:f9:36:cc:70:64:fe:de:6d:6c:ad:e1:e7:
                    c5:5e:6f:80:a8:01:a7:0b:ac:a5:13:da:0a:91:ab:
                    d4:a2:50:73:d1:37:1e:73:ac:48:07:31:91:1d:13:
                    7d:f9:cf:b2:fa:ac:81:71:2b:35:a2:af:53:70:b3:
                    19:ae:42:ef:04:12:98:5d:e8:d2:ff:be:25:34:5a:
                    5e:b4:e2:1a:78:4f:f7:43:32:50:db:7b:91:25:4c:
                    4c:35:1d:30:f7:56:9b:42:1f
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Subject Key Identifier:
                EF:BB:14:56:15:4F:11:41:74:84:D9:6F:4E:87:3F:80:D7:DC:63:AA
            X509v3 Authority Key Identifier:
                48:70:75:D4:ED:6B:49:3B:53:DB:E9:A6:F9:9C:55:92:51:BD:96:E6
            X509v3 Basic Constraints:
                CA:FALSE
            X509v3 Subject Alternative Name:
                <EMPTY>

            X509v3 Extended Key Usage:
                TLS Web Client Authentication
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        86:5c:79:fc:0c:31:6d:34:f7:0c:e5:1e:14:6e:60:44:08:f8:
        3c:01:3c:34:3a:d0:38:7f:02:f4:11:49:15:59:36:25:c5:0d:
        97:20:a3:93:2a:91:01:15:e5:c4:87:0a:f5:d4:70:43:4e:f8:
        94:2a:74:65:72:85:1a:ed:53:ce:fe:3d:4b:d1:ae:ad:da:02:
        69:a8:ea:e8:ad:27:9e:00:c9:a1:32:f4:96:a8:69:e8:26:ab:
        50:67:7b:bb:64:a5:f8:fb:90:8a:18:eb:cc:3c:0e:b9:1c:11:
        88:92:4e:96:10:09:63:65:c7:d0:1a:7b:a4:95:8d:48:d2:cb:
        76:46
-----BEGIN CERTIFICATE-----
MIICzzCCAjigAwIBAgIBATANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECAwITXkgU3RhdGUxEDAOBgNVBAcMB015IENpdHkxDzANBgNVBAoMBk15
IE9yZzEQMA4GA1UECwwHTXkgVW5pdDEQMA4GA1UEAwwHUk9PVCBDQTAgFw0yNDEw
MDYyMDUwNDNaGA8yMTI0MDkxMjIwNTA0M1owgYIxCzAJBgNVBAYTAlVTMREwDwYD
VQQIDAhNeSBTdGF0ZTEQMA4GA1UEBwwHTXkgQ2l0eTEPMA0GA1UECgwGTXkgT3Jn
MRAwDgYDVQQLDAdNeSBVbml0MRIwEAYDVQQDDAljbGllbnRfY24xFzAVBgoJkiaJ
k/IsZAEBDAd0ZXN0MTIzMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDDXpW9
uT4Zu+sGrHKcN8N/3SrB2r7SwacYnMw8LTGvsvk2zHBk/t5tbK3h58Veb4CoAacL
rKUT2gqRq9SiUHPRNx5zrEgHMZEdE335z7L6rIFxKzWir1NwsxmuQu8EEphd6NL/
viU0Wl604hp4T/dDMlDbe5ElTEw1HTD3VptCHwIDAQABo20wazAdBgNVHQ4EFgQU
77sUVhVPEUF0hNlvToc/gNfcY6owHwYDVR0jBBgwFoAUSHB11O1rSTtT2+mm+ZxV
klG9luYwCQYDVR0TBAIwADAJBgNVHREEAjAAMBMGA1UdJQQMMAoGCCsGAQUFBwMC
MA0GCSqGSIb3DQEBCwUAA4GBAIZcefwMMW009wzlHhRuYEQI+DwBPDQ60Dh/AvQR
SRVZNiXFDZcgo5MqkQEV5cSHCvXUcENO+JQqdGVyhRrtU87+PUvRrq3aAmmo6uit
J54AyaEy9Jaoaegmq1Bne7tkpfj7kIoY68w8DrkcEYiSTpYQCWNlx9Aae6SVjUjS
y3ZG
-----END CERTIFICATE-----`

	/*

	   > serverCert

	   openssl genrsa -out server.key 1024 && \
	   openssl rsa -in ./server.key -outform PEM \
	   	-pubout -out ./server.pub && \
	   openssl req -key ./server.key -new\
	          	-sha1 -out ./server.csr \
	          	-subj "/C=US/ST=My State/L=My City/O=My Org/OU=My Unit/CN=127.0.0.1" \
	   	&& \
	   EXTFILE="subjectKeyIdentifier=hash\n" && \
	   EXTFILE="${EXTFILE}authorityKeyIdentifier=keyid,issuer\n" && \
	   EXTFILE="${EXTFILE}basicConstraints=CA:FALSE\n" && \
	   EXTFILE="${EXTFILE}subjectAltName=email:copy\n" && \
	   EXTFILE="${EXTFILE}extendedKeyUsage=serverAuth\n" && \
	   openssl  x509 -req -days 36500 \
	   	-in ./server.csr \
	   	-extfile <(printf "${EXTFILE}") \
	   	-CA ./root.crt \
	   	-CAkey ./root.key \
	   	-set_serial 7 \
	          	-sha256 \
	   	-out ./server.crt \
	   	&& \
	   openssl x509 -in server.crt -text

	   > output

	   is below

	*/

	serverCert = `Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 7 (0x7)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=ROOT CA
        Validity
            Not Before: Oct  6 20:38:02 2024 GMT
            Not After : Sep 12 20:38:02 2124 GMT
        Subject: C=US, ST=My State, L=My City, O=My Org, OU=My Unit, CN=127.0.0.1
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (1024 bit)
                Modulus:
                    00:b6:d5:2f:a6:7a:78:5d:40:a6:0d:76:6f:e9:9d:
                    54:6d:d9:e9:d6:32:00:f2:8a:fb:da:87:be:05:07:
                    b4:58:ab:88:25:f8:38:e7:50:25:23:47:99:8f:3c:
                    ff:8a:cc:61:7c:21:db:39:c9:81:f6:0c:f2:22:a8:
                    19:65:7a:ae:c6:32:74:63:4d:a5:14:fa:b5:04:ab:
                    a4:83:c5:0f:26:38:b3:65:9d:68:bb:4f:55:e4:0b:
                    e5:71:49:dd:5b:b8:a0:ed:7d:13:6f:29:03:44:20:
                    d0:2d:9c:44:e4:0e:8b:d7:71:79:fe:35:cd:6c:7c:
                    79:a4:01:08:ae:9e:95:46:d9
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Subject Key Identifier:
                CA:72:DA:A3:17:BB:56:CC:14:A9:BA:12:F2:88:7F:F4:15:69:33:CB
            X509v3 Authority Key Identifier:
                48:70:75:D4:ED:6B:49:3B:53:DB:E9:A6:F9:9C:55:92:51:BD:96:E6
            X509v3 Basic Constraints:
                CA:FALSE
            X509v3 Subject Alternative Name:
                <EMPTY>

            X509v3 Extended Key Usage:
                TLS Web Server Authentication
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        08:14:37:cd:ec:d6:4e:81:d2:d7:09:ba:5a:50:84:6a:1b:f2:
        02:49:44:94:5d:e3:41:48:09:dc:88:0b:37:d6:e9:c7:b6:4b:
        42:58:b3:cb:81:5b:a6:0d:78:47:1b:4a:5a:5f:d5:14:4c:37:
        bd:b6:64:c4:d5:ac:17:d0:6c:2d:f5:1b:aa:d8:de:27:f1:1e:
        26:42:dd:45:90:ef:97:0b:e6:c9:01:c5:4b:7c:c3:81:18:c6:
        28:d9:8a:f5:a5:8c:b4:ec:75:c2:b8:43:83:d0:db:09:e1:58:
        a6:2a:65:52:97:0b:d0:d6:c7:43:8f:10:63:23:b4:ce:c9:15:
        4d:4a
-----BEGIN CERTIFICATE-----
MIICtTCCAh6gAwIBAgIBBzANBgkqhkiG9w0BAQsFADBnMQswCQYDVQQGEwJVUzER
MA8GA1UECAwITXkgU3RhdGUxEDAOBgNVBAcMB015IENpdHkxDzANBgNVBAoMBk15
IE9yZzEQMA4GA1UECwwHTXkgVW5pdDEQMA4GA1UEAwwHUk9PVCBDQTAgFw0yNDEw
MDYyMDM4MDJaGA8yMTI0MDkxMjIwMzgwMlowaTELMAkGA1UEBhMCVVMxETAPBgNV
BAgMCE15IFN0YXRlMRAwDgYDVQQHDAdNeSBDaXR5MQ8wDQYDVQQKDAZNeSBPcmcx
EDAOBgNVBAsMB015IFVuaXQxEjAQBgNVBAMMCTEyNy4wLjAuMTCBnzANBgkqhkiG
9w0BAQEFAAOBjQAwgYkCgYEAttUvpnp4XUCmDXZv6Z1Ubdnp1jIA8or72oe+BQe0
WKuIJfg451AlI0eZjzz/isxhfCHbOcmB9gzyIqgZZXquxjJ0Y02lFPq1BKukg8UP
JjizZZ1ou09V5AvlcUndW7ig7X0TbykDRCDQLZxE5A6L13F5/jXNbHx5pAEIrp6V
RtkCAwEAAaNtMGswHQYDVR0OBBYEFMpy2qMXu1bMFKm6EvKIf/QVaTPLMB8GA1Ud
IwQYMBaAFEhwddTta0k7U9vppvmcVZJRvZbmMAkGA1UdEwQCMAAwCQYDVR0RBAIw
ADATBgNVHSUEDDAKBggrBgEFBQcDATANBgkqhkiG9w0BAQsFAAOBgQAIFDfN7NZO
gdLXCbpaUIRqG/ICSUSUXeNBSAnciAs31unHtktCWLPLgVumDXhHG0paX9UUTDe9
tmTE1awX0Gwt9Ruq2N4n8R4mQt1FkO+XC+bJAcVLfMOBGMYo2Yr1pYy07HXCuEOD
0NsJ4VimKmVSlwvQ1sdDjxBjI7TOyRVNSg==
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

		ExpectOK       bool
		ExpectResponse *authenticator.Response
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

			ExpectOK: true,
			ExpectResponse: &authenticator.Response{
				User: &user.DefaultInfo{
					Name:   "127.0.0.1",
					Groups: []string{"My Org"},
					Extra: map[string][]string{
						user.CredentialIDKey: {"X509SHA256=04adf2b65e6325a8c467256eb3a9a373d818398d9a1f1d9eca1cbc2c237fe75f"},
					},
				},
			},
			ExpectErr: false,
		},

		"common name and UserID": {
			Opts:  getDefaultVerifyOptions(t),
			Certs: getCerts(t, clientCNCert),
			User:  CommonNameUserConversion,

			ExpectOK: true,
			ExpectResponse: &authenticator.Response{
				User: &user.DefaultInfo{
					Name:   "client_cn",
					Groups: []string{"My Org"},
					UID:    "test123",
					Extra: map[string][]string{
						user.CredentialIDKey: {"X509SHA256=f3beda2d496e82c49b5349069860b543f9a116732af4029ac791166a4e48826b"},
					},
				},
			},
			ExpectErr: false,
		},
		"ca with multiple organizations": {
			Opts: x509.VerifyOptions{
				Roots: getRootCertPoolFor(t, caWithGroups),
			},
			Certs: getCerts(t, caWithGroups),
			User:  CommonNameUserConversion,

			ExpectOK: true,
			ExpectResponse: &authenticator.Response{
				User: &user.DefaultInfo{
					Name:   "ROOT CA WITH GROUPS",
					Groups: []string{"My Org", "My Org 1", "My Org 2"},
					Extra: map[string][]string{
						user.CredentialIDKey: {"X509SHA256=6f337bb6576b6f942bd5ac5256f621e352aa7b34d971bda9b8f8981f51bba456"},
					},
				},
			},
			ExpectErr: false,
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

			ExpectOK: true,
			ExpectResponse: &authenticator.Response{
				User: &user.DefaultInfo{
					Name: "custom",
				},
			},
			ExpectErr: false,
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

			ExpectOK: true,
			ExpectResponse: &authenticator.Response{
				User: &user.DefaultInfo{
					Name: "My Client",
					Extra: map[string][]string{
						user.CredentialIDKey: {"X509SHA256=794b0529fd1a72d55d52d98be9bab5b822d16f9ae86c4373fa7beee3cafe8582"},
					},
				},
			},
			ExpectErr: false,
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
		t.Run(k, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "/", nil)
			if !testCase.Insecure {
				req.TLS = &tls.ConnectionState{PeerCertificates: testCase.Certs}
			}

			// this effectively tests the simple dynamic verify function.
			a := New(testCase.Opts, testCase.User)

			resp, ok, err := a.AuthenticateRequest(req)

			if testCase.ExpectErr && err == nil {
				t.Fatalf("Expected error, got none")
			}
			if !testCase.ExpectErr && err != nil {
				t.Fatalf("Got unexpected error: %v", err)
			}

			if testCase.ExpectOK != ok {
				t.Fatalf("Expected ok=%v, got %v", testCase.ExpectOK, ok)
			}

			if testCase.ExpectOK {
				sort.Strings(testCase.ExpectResponse.User.GetGroups())
				sort.Strings(resp.User.GetGroups())
				if diff := cmp.Diff(testCase.ExpectResponse, resp); diff != "" {
					t.Errorf("Bad response; diff (-want +got)\n%s", diff)
				}
			}
		})
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
		data, err := os.ReadFile(filename)
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
			expectedIdentifier: "SN=1, SKID=EF:BB:14:56:15:4F:11:41:74:84:D9:6F:4E:87:3F:80:D7:DC:63:AA, AKID=48:70:75:D4:ED:6B:49:3B:53:DB:E9:A6:F9:9C:55:92:51:BD:96:E6",
		},
		{
			name: "nil serial",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.SerialNumber = nil
				return c
			}(),
			expectedIdentifier: "SN=<nil>, SKID=EF:BB:14:56:15:4F:11:41:74:84:D9:6F:4E:87:3F:80:D7:DC:63:AA, AKID=48:70:75:D4:ED:6B:49:3B:53:DB:E9:A6:F9:9C:55:92:51:BD:96:E6",
		},
		{
			name: "empty SKID",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.SubjectKeyId = nil
				return c
			}(),
			expectedIdentifier: "SN=1, SKID=, AKID=48:70:75:D4:ED:6B:49:3B:53:DB:E9:A6:F9:9C:55:92:51:BD:96:E6",
		},
		{
			name: "empty AKID",
			cert: func() *x509.Certificate {
				c := getCert(t, clientCNCert)
				c.AuthorityKeyId = nil
				return c
			}(),
			expectedIdentifier: "SN=1, SKID=EF:BB:14:56:15:4F:11:41:74:84:D9:6F:4E:87:3F:80:D7:DC:63:AA, AKID=",
		},
		{
			name:               "self-signed",
			cert:               getCert(t, selfSignedCert),
			expectedIdentifier: "SN=654708847004117259890317394342561449606220871090, SKID=56:A5:55:02:8C:97:FD:1E:A0:B8:DE:EF:5E:95:F0:AC:A6:23:6F:16, AKID=56:A5:55:02:8C:97:FD:1E:A0:B8:DE:EF:5E:95:F0:AC:A6:23:6F:16",
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
