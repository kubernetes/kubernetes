// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package option

import (
	"testing"

	"crypto/tls"
	"math/big"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal"
	"google.golang.org/grpc"
)

// Below is a dummy certificate/key pair taken from
// https://golang.org/src/crypto/tls/tls_test.go
const certPEM = `-----BEGIN CERTIFICATE-----
MIIB0zCCAX2gAwIBAgIJAI/M7BYjwB+uMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMTIwOTEyMjE1MjAyWhcNMTUwOTEyMjE1MjAyWjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBANLJ
hPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wok/4xIA+ui35/MmNa
rtNuC+BdZ1tMuVCPFZcCAwEAAaNQME4wHQYDVR0OBBYEFJvKs8RfJaXTH08W+SGv
zQyKn0H8MB8GA1UdIwQYMBaAFJvKs8RfJaXTH08W+SGvzQyKn0H8MAwGA1UdEwQF
MAMBAf8wDQYJKoZIhvcNAQEFBQADQQBJlffJHybjDGxRMqaRmDhX0+6v02TUKZsW
r5QuVbpQhH6u+0UgcW0jp9QwpxoPTLTWGXEWBBBurxFwiCBhkQ+V
-----END CERTIFICATE-----
-----BEGIN PRIVATE KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END PRIVATE KEY-----
`

// Check that the slice passed into WithScopes is copied.
func TestCopyScopes(t *testing.T) {
	o := &internal.DialSettings{}

	scopes := []string{"a", "b"}
	WithScopes(scopes...).Apply(o)

	// Modify after using.
	scopes[1] = "c"

	if o.Scopes[0] != "a" || o.Scopes[1] != "b" {
		t.Errorf("want ['a', 'b'], got %+v", o.Scopes)
	}
}

func TestApply(t *testing.T) {
	conn := &grpc.ClientConn{}
	opts := []ClientOption{
		WithEndpoint("https://example.com:443"),
		WithScopes("a"), // the next WithScopes should overwrite this one
		WithScopes("https://example.com/auth/helloworld", "https://example.com/auth/otherthing"),
		WithGRPCConn(conn),
		WithUserAgent("ua"),
		WithCredentialsFile("service-account.json"),
		WithCredentialsJSON([]byte(`{some: "json"}`)),
		WithCredentials(&google.DefaultCredentials{ProjectID: "p"}),
		WithAPIKey("api-key"),
		WithAudiences("https://example.com/"),
		WithQuotaProject("user-project"),
		WithRequestReason("Request Reason"),
		WithTelemetryDisabled(),
	}
	var got internal.DialSettings
	for _, opt := range opts {
		opt.Apply(&got)
	}
	want := internal.DialSettings{
		Scopes:            []string{"https://example.com/auth/helloworld", "https://example.com/auth/otherthing"},
		UserAgent:         "ua",
		Endpoint:          "https://example.com:443",
		GRPCConn:          conn,
		Credentials:       &google.DefaultCredentials{ProjectID: "p"},
		CredentialsFile:   "service-account.json",
		CredentialsJSON:   []byte(`{some: "json"}`),
		APIKey:            "api-key",
		Audiences:         []string{"https://example.com/"},
		QuotaProject:      "user-project",
		RequestReason:     "Request Reason",
		TelemetryDisabled: true,
	}
	if !cmp.Equal(got, want, cmpopts.IgnoreUnexported(grpc.ClientConn{})) {
		t.Errorf(cmp.Diff(got, want, cmpopts.IgnoreUnexported(grpc.ClientConn{})))
	}
}

func mockClientCertSource(info *tls.CertificateRequestInfo) (*tls.Certificate, error) {
	cert, _ := tls.X509KeyPair([]byte(certPEM), []byte(certPEM))
	return &cert, nil
}

func TestApplyClientCertSource(t *testing.T) {
	opts := []ClientOption{
		WithClientCertSource(mockClientCertSource),
	}
	var got internal.DialSettings
	for _, opt := range opts {
		opt.Apply(&got)
	}
	want := internal.DialSettings{
		ClientCertSource: mockClientCertSource,
	}

	// Functions cannot be compared in Golang for equality, so we will compare the output of the functions instead.
	certGot, err := got.ClientCertSource(nil)
	if err != nil {
		t.Error(err)
	}
	certWant, err := want.ClientCertSource(nil)
	if err != nil {
		t.Error(err)
	}
	if !cmp.Equal(certGot, certWant, cmpopts.IgnoreUnexported(big.Int{}), cmpopts.IgnoreFields(tls.Certificate{}, "Leaf")) {
		t.Errorf(cmp.Diff(certGot, certWant, cmpopts.IgnoreUnexported(big.Int{}), cmpopts.IgnoreFields(tls.Certificate{}, "Leaf")))
	}
}
