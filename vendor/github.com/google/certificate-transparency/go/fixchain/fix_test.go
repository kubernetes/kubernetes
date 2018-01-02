package fixchain

import (
	"net/http"
	"testing"

	"github.com/google/certificate-transparency/go/x509"
)

var constructChainTests = []fixTest{
	// constructChain()
	{ // Correct chain returns chain
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function: "constructChain",
		expectedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{
		cert:  testLeaf,
		chain: []string{testIntermediate2, testIntermediate1, testRoot},
		roots: []string{testRoot},

		function: "constructChain",
		expectedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
			{"Leaf", "Intermediate2", "Intermediate1", "CA", "CA"},
		},
	},
	{ // No roots results in an error
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},

		function:     "constructChain",
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // Incomplete chain results in an error
		cert:  googleLeaf,
		roots: []string{verisignRoot},

		function:     "constructChain",
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // The wrong intermediate and root results in an error
		cert:  megaLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "constructChain",
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // The wrong root results in an error
		cert:  megaLeaf,
		chain: []string{comodoIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "constructChain",
		expectedErrs: []errorType{VerifyFailed},
	},
}

var fixChainTests = []fixTest{
	// fixChain()
	{ // Correct chain returns chain
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function: "fixChain",
		expectedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{ // No roots results in an error
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},

		function:     "fixChain",
		expectedErrs: []errorType{FixFailed},
	},
	{ // No roots where chain that will be built contains a loop results in error
		cert:  testC,
		chain: []string{testB, testA},

		function:     "fixChain",
		expectedErrs: []errorType{FixFailed},
	},
	{ // Incomplete chain returns fixed chain
		cert:  googleLeaf,
		roots: []string{verisignRoot},

		function: "fixChain",
		expectedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{
		cert:  testLeaf,
		chain: []string{testIntermediate2},
		roots: []string{testRoot},

		function: "fixChain",
		expectedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
	},
	{
		cert:  testLeaf,
		chain: []string{testIntermediate1},
		roots: []string{testRoot},

		function: "fixChain",
		expectedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
	},
	{
		cert:  testLeaf,
		roots: []string{testRoot},

		function: "fixChain",
		expectedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
	},
	{ // The wrong intermediate and root results in an error
		cert:  megaLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "fixChain",
		expectedErrs: []errorType{FixFailed},
	},
	{ // The wrong root results in an error
		cert:  megaLeaf,
		chain: []string{comodoIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "fixChain",
		expectedErrs: []errorType{FixFailed},
	},
	// TODO(katjoyce): Add test where cert has multiple URLs in AIA extension.
}

func setUpFix(t *testing.T, i int, ft *fixTest) *toFix {
	// Create & populate toFix to test from fixTest info
	fix := &toFix{
		cert:  GetTestCertificateFromPEM(t, ft.cert),
		chain: newDedupedChain(extractTestChain(t, i, ft.chain)),
		roots: extractTestRoots(t, i, ft.roots),
		cache: newURLCache(&http.Client{Transport: &testRoundTripper{}}, false),
	}

	intermediates := x509.NewCertPool()
	for j, cert := range ft.chain {
		ok := intermediates.AppendCertsFromPEM([]byte(cert))
		if !ok {
			t.Errorf("#%d: Failed to parse intermediate #%d", i, j)
		}
	}

	fix.opts = &x509.VerifyOptions{
		Intermediates:     intermediates,
		Roots:             fix.roots,
		DisableTimeChecks: true,
		KeyUsages:         []x509.ExtKeyUsage{x509.ExtKeyUsageAny},
	}

	return fix
}

func testFixChainFunctions(t *testing.T, i int, ft *fixTest) {
	fix := setUpFix(t, i, ft)

	var chains [][]*x509.Certificate
	var ferrs []*FixError
	switch ft.function {
	case "constructChain":
		chains, ferrs = fix.constructChain()
	case "fixChain":
		chains, ferrs = fix.fixChain()
	case "handleChain":
		chains, ferrs = fix.handleChain()
	}

	matchTestChainList(t, i, ft.expectedChains, chains)
	matchTestErrorList(t, i, ft.expectedErrs, ferrs)
}

func TestFixChainFunctions(t *testing.T) {
	var allTests []fixTest
	allTests = append(allTests, constructChainTests...)
	allTests = append(allTests, fixChainTests...)
	allTests = append(allTests, handleChainTests...)
	for i, ft := range allTests {
		testFixChainFunctions(t, i, &ft)
	}
}

func TestFix(t *testing.T) {
	for i, test := range handleChainTests {
		chains, ferrs := Fix(GetTestCertificateFromPEM(t, test.cert),
			extractTestChain(t, i, test.chain),
			extractTestRoots(t, i, test.roots),
			&http.Client{Transport: &testRoundTripper{}})

		matchTestChainList(t, i, test.expectedChains, chains)
		matchTestErrorList(t, i, test.expectedErrs, ferrs)
	}
}
