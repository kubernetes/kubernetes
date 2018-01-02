package fixchain

import (
	"errors"
	"fmt"
	"testing"

	"github.com/google/certificate-transparency/go/x509"
)

func TestEqual(t *testing.T) {
	equalTests := []struct {
		e        *FixError
		f        *FixError
		expEqual bool
	}{
		{
			&FixError{},
			&FixError{},
			true,
		},
		{
			&FixError{Type: PostFailed},
			&FixError{},
			false,
		},
		{
			&FixError{Type: PostFailed},
			&FixError{Type: LogPostFailed},
			false,
		},
		{
			&FixError{Cert: GetTestCertificateFromPEM(t, googleLeaf)},
			&FixError{},
			false,
		},
		{
			&FixError{Cert: GetTestCertificateFromPEM(t, googleLeaf)},
			&FixError{Cert: GetTestCertificateFromPEM(t, megaLeaf)},
			false,
		},
		{
			&FixError{
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
			},
			&FixError{},
			false,
		},
		{ // Chains with only one cert different.
			&FixError{
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
			},
			&FixError{
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, comodoRoot),
				},
			},
			false,
		},
		{ // Completely different chains.
			&FixError{
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
			},
			&FixError{
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, megaLeaf),
					GetTestCertificateFromPEM(t, comodoIntermediate),
					GetTestCertificateFromPEM(t, comodoRoot),
				},
			},
			false,
		},
		{
			&FixError{URL: "https://www.test.com"},
			&FixError{},
			false,
		},
		{
			&FixError{URL: "https://www.test.com"},
			&FixError{URL: "https://www.test1.com"},
			false,
		},
		{
			&FixError{Bad: []byte(googleLeaf)},
			&FixError{},
			false,
		},
		{
			&FixError{Bad: []byte(googleLeaf)},
			&FixError{Bad: []byte(megaLeaf)},
			false,
		},
		{
			&FixError{Error: errors.New("Error1")},
			&FixError{},
			false,
		},
		{
			&FixError{Error: errors.New("Error1")},
			&FixError{Error: errors.New("Error2")},
			false,
		},
		{
			&FixError{Code: 200},
			&FixError{},
			false,
		},
		{
			&FixError{Code: 200},
			&FixError{Code: 502},
			false,
		},
		{
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			&FixError{},
			false,
		},
		{
			&FixError{},
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			false,
		},
		{
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			true,
		},
		{
			&FixError{
				Type: PostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Post Failed"),
				Code:  404,
			},
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, megaLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, megaLeaf),
					GetTestCertificateFromPEM(t, comodoIntermediate),
					GetTestCertificateFromPEM(t, comodoRoot),
				},
				URL:   "https://www.test1.com",
				Bad:   GetTestCertificateFromPEM(t, megaLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  502,
			},
			false,
		},
		{ // nil test
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			nil,
			false,
		},
	}

	for i, test := range equalTests {
		if test.e.Equal(test.f) != test.expEqual {
			t.Errorf("#%d: expected FixError.Equal() to return %t, returned %t", i, test.expEqual, !test.expEqual)
		}
	}
}

func TestTypeString(t *testing.T) {
	typeStringTests := []struct {
		ferr     FixError
		expected string
	}{
		{
			FixError{Type: None},
			"None",
		},
		{
			FixError{Type: ParseFailure},
			"ParseFailure",
		},
		{
			FixError{Type: CannotFetchURL},
			"CannotFetchURL",
		},
		{
			FixError{Type: FixFailed},
			"FixFailed",
		},
		{
			FixError{Type: PostFailed},
			"PostFailed",
		},
		{
			FixError{Type: LogPostFailed},
			"LogPostFailed",
		},
		{
			FixError{Type: VerifyFailed},
			"VerifyFailed",
		},
		{
			FixError{},
			"None",
		},
	}

	for i, test := range typeStringTests {
		if got, want := test.ferr.TypeString(), test.expected; got != want {
			t.Errorf("#%d: TypeString() returned %s, expected %s.", i, got, want)
		}
	}
}

func TestString(t *testing.T) {
	stringTests := []struct {
		ferr *FixError
		str  string
	}{
		{
			&FixError{Type: None},
			"None\n",
		},
		{
			&FixError{
				Type: LogPostFailed,
				Cert: GetTestCertificateFromPEM(t, googleLeaf),
				Chain: []*x509.Certificate{
					GetTestCertificateFromPEM(t, googleLeaf),
					GetTestCertificateFromPEM(t, thawteIntermediate),
					GetTestCertificateFromPEM(t, verisignRoot),
				},
				URL:   "https://www.test.com",
				Error: errors.New("Log Post Failed"),
				Code:  404,
			},
			"LogPostFailed\n" +
				"Status Code: 404\n" +
				"Error: Log Post Failed\n" +
				"URL: https://www.test.com\n" +
				"Cert: " + googleLeaf +
				"Chain: " + googleLeaf + thawteIntermediate + verisignRoot,
		},
	}

	for i, test := range stringTests {
		if got, want := test.ferr.String(), test.str; got != want {
			t.Errorf("#%d: String() returned %s, expected %s.", i, got, want)
		}
	}
}

func TestMarshalJSON(t *testing.T) {
	marshalJSONTests := []*FixError{
		&FixError{},
		&FixError{
			Type: LogPostFailed,
			Cert: GetTestCertificateFromPEM(t, googleLeaf),
			Chain: []*x509.Certificate{
				GetTestCertificateFromPEM(t, googleLeaf),
				GetTestCertificateFromPEM(t, thawteIntermediate),
				GetTestCertificateFromPEM(t, verisignRoot),
			},
			URL:   "https://www.test.com",
			Bad:   GetTestCertificateFromPEM(t, googleLeaf).Raw,
			Error: errors.New("Log Post Failed"),
			Code:  404,
		},
	}

	for i, test := range marshalJSONTests {
		b, err := test.MarshalJSON()
		if err != nil {
			t.Errorf("#%d: Error marshalling json: %s", i, err.Error())
		}

		ferr, err := UnmarshalJSON(b)
		if err != nil {
			t.Errorf("#%d: Error unmarshalling json: %s", i, err.Error())
		}

		if !test.Equal(ferr) {
			t.Errorf("#%d: Original FixError does not match marshalled-then-unmarshalled FixError", i)
		}
	}
}

func TestDumpPEM(t *testing.T) {
	dumpPEMTests := []string{googleLeaf}

	for i, test := range dumpPEMTests {
		cert := GetTestCertificateFromPEM(t, test)
		p := dumpPEM(cert.Raw)
		certFromPEM := GetTestCertificateFromPEM(t, p)
		if !cert.Equal(certFromPEM) {
			t.Errorf("#%d: cert from output of dumpPEM() does not match original", i)
		}
	}
}

func TestDumpChainPEM(t *testing.T) {
	dumpChainPEMTests := []struct {
		chain    []string
		expected string
	}{
		{
			[]string{googleLeaf, thawteIntermediate},
			fmt.Sprintf("%s%s", googleLeaf, thawteIntermediate),
		},
	}

	for i, test := range dumpChainPEMTests {
		chain := extractTestChain(t, i, test.chain)
		if got := dumpChainPEM(chain); got != test.expected {
			t.Errorf("#%d: dumpChainPEM() returned %s, expected %s", i, got, test.expected)
		}
	}
}
