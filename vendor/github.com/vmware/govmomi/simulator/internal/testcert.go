/*
Copyright (c) 2014-2022 VMware, Inc. All Rights Reserved.

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

package internal

// LocalhostCert is a PEM-encoded TLS cert with SAN IPs
// "127.0.0.1" and "[::1]", expiring at Jan 29 16:00:00 2084 GMT.
// generated from src/crypto/tls:
// go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
//
// Please note the certificate was updated to 2048 bits due to macOS now
// requiring RSA keys of at least 2048 bits. For more information, please see:
// https://developer.apple.com/documentation/security/preventing_insecure_network_connections
var LocalhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDOjCCAiKgAwIBAgIRAK6d/JpGL75P2wOSYc6WalEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzCCASIwDQYJKoZIhvcNAQEBBQADggEP
ADCCAQoCggEBAK9lInM4OAKI4z+wDNkvcZC6S6exFSOysp7NPJyaEAhW93kPY7gO
f6H5aP3V3YU0vYpCSnz/UhyDD+/knBof1J3Do7FVwYtC293vrtXffNtAvygfJodW
1dPllp17ZJbq76ei9oWq1Y5Hox/sVYmNVNztvBfK1mtpS8z8Qrk1LWCyLiDHkvDA
hCy2OjuaopxC6qQejdWT1PxwbqptuLVakQmecpiFrupy8DTG0x0rxxdMdAATywhY
Gm49A/FroagZ6HMz3bm39we/w6VIx3pX1lbUUyrfjvBgfUlRwxyZABBj2STGsOQJ
a451eEcESXcSEWzjGjUQ1Wf+zzxr2GAHmI8CAwEAAaOBiDCBhTAOBgNVHQ8BAf8E
BAMCAqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUwAwEB/zAdBgNV
HQ4EFgQUjtR2VSuxchTxe0UNDVqWDNMR37AwLgYDVR0RBCcwJYILZXhhbXBsZS5j
b22HBH8AAAGHEAAAAAAAAAAAAAAAAAAAAAEwDQYJKoZIhvcNAQELBQADggEBACK7
1L15IeYPiQHFDml3EgDoRnd/tAaZP9nUZIIPLRgUnQITNAtjFgBvqneQZwlQtco2
s8YXSivBQiATlBEtGBzVxv36R4tTXldIgJxaCUxxZtqALLwyGqSaI/cwE0pWa6Z0
Op2wkzUmoQ5rRrJfRM+C6HR/+lWkNtHRzaUFOSlxNJbPo53K53OoDriwEc1PvEYP
wFeUXwTzCZ68pAlWUmDKCyp+lPhjIt2Gznig+BSPCNJqmwKM76oFyywi3HIP56rD
/cwUtoplF68uVuD8HXb1ggGsqtGiAT4GLT8tU5w+BtK8ZIs/LK7mdi7W8aIOhUUH
l1lgeV3oEQue3A7SogA=
-----END CERTIFICATE-----`)

// LocalhostKey is the private key for localhostCert.
var LocalhostKey = []byte(`-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCvZSJzODgCiOM/
sAzZL3GQukunsRUjsrKezTycmhAIVvd5D2O4Dn+h+Wj91d2FNL2KQkp8/1Icgw/v
5JwaH9Sdw6OxVcGLQtvd767V33zbQL8oHyaHVtXT5Zade2SW6u+novaFqtWOR6Mf
7FWJjVTc7bwXytZraUvM/EK5NS1gsi4gx5LwwIQstjo7mqKcQuqkHo3Vk9T8cG6q
bbi1WpEJnnKYha7qcvA0xtMdK8cXTHQAE8sIWBpuPQPxa6GoGehzM925t/cHv8Ol
SMd6V9ZW1FMq347wYH1JUcMcmQAQY9kkxrDkCWuOdXhHBEl3EhFs4xo1ENVn/s88
a9hgB5iPAgMBAAECggEAEpeS3knQThx6kk60HfWUgTXuPRldV0pi+shgq2z9VBT7
6J5EAMewqdfJVFbuQ2eCy/wY70UVTCZscw51qaNEI3EQkgS4Hm345n64tr0Y/BjR
6ovaxq/ivLJyk8D3ubOvscJphWPFfW6EkSa5LnqHy197972tmvcvbMw0unMzmzM7
DenXdoIJQu1SqLiLUiDXEfkvCReekqhe1jATCwTzIBTCnTWxgI4Ox2qsBaxuwrnl
D1GpWy4sh8NpDB0EBwdrjAOmLDOyvsy2X65DIlHS/k7901tvzyNjRrsr2Ig0sAz4
w0ke6CUKQ2B+Pqn3p6bvxRYMP08+ZjlQpPuU4RrxGQKBgQDd3HCrZCgUJAGcfzYX
ZzSmSoxB9+sEuVUZGU+ICMPlG0Dd8aEkXIyDjGMWsLFMIvzNBf4wB1FwdLaCJu6y
0PbX3KVfg/Yc4lvYUuQ+1nD/3gm2hE46lZuSfbmExH5SQVLSbSQf9S/5BTHAWQO9
PNie71AZ8fO5YDBM18tq2V7dBQKBgQDKYk1+Zup5p0BeRyCNnVYnpngO+pAZZmld
gYiRn8095QJ/Y+yV0kYc2+0+5f6St462E+PVH3tX1l9OG3ujVQnWfu2iSsRJRMUf
0blxqTWvqdcGi8SLpVjkrHn30scFNWmojhJv3k42H3nUMC1+WU3rp2f7+W58afyd
NY9x4sqzgwKBgQCoeMq9+3JLyQPIOPl0UBS06gsT1RUMI0gxpPy1yiInich6QRAi
snypMCPWiRo5PKBHd/OLuSLoiFhHARVliDTJum2B2I09Zc5kuJ1F8kUgpxUtGc7l
wdG/LeWAok1iXORtkh9KfT+Ok5kx/OZP/zJnjkZ/TTHMZPSIhZ2cZ7AXmQKBgHMP
HjWNtyKApsSytVwtpgyWxMznQMNgCOkjOoxoCJx2tUvNeHTY/glsM14+DdRFzTnQ
5weEhXAzrS1PzKPYNeafdOR+k0eAdH2Zk09+PspmyZusHIqz72zabeEqEQHyEubE
FtFI1rhIfs/WsBaUGQuvuhtz/I95BiguiiXaJRmXAoGADwcO6YXoWXga07gGRwZP
LYKwt5wBh13LAGbSsUyCSK5FG6ZrTmzaFdAGV1U4wc/wgiIgv33m8BG4Ikxvpa0r
Wg3dbhBx9Oya8QWIMBPk72KKEzsSDfi+Cn52ZmxTkWbBDCnkRhG77Ooi8vJ3dhq4
fHeAu1F9OwF83SBi1oNySd8=
-----END PRIVATE KEY-----`)
