package fixchain

import (
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/google/certificate-transparency/go/x509"
)

var newFixAndLogTests = []fixAndLogTest{
	// Tests that add chains to the FixAndLog one at a time using QueueChain()
	{ // Full chain successfully logged.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		function: "QueueChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{ // Chain without the root successfully logged.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate},

		function: "QueueChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{ // Chain to wrong root results in error.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{megaLeaf, comodoIntermediate, comodoRoot},

		function:     "QueueChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // Chain without correct root containing loop results in error.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testC, testB, testA},

		function:     "QueueChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // Incomplete chain successfully logged.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf},

		function: "QueueChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testLeaf},

		function: "QueueChain",
		expLoggedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // Garbled chain (with a leaf that has no chain to our roots) results in an error.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{megaLeaf, googleLeaf, thawteIntermediate, verisignRoot},

		function:     "QueueChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // Garbled chain (with a leaf that has a chain to our roots) successfully logged.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testLeaf, megaLeaf, googleLeaf, thawteIntermediate, comodoRoot},

		function: "QueueChain",
		expLoggedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	// Tests that add chains to the FixAndLog using QueueAllCertsInChain()
	{ // Full chain successfully logged.
		// Note:  Verifying a root to itself results in an error.
		// This is not an issue as the root will already be known to the log, and chains
		// aren't required to contain the root.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
			{"Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
			{"Thawte", "VeriSign"},
		},
	},
	{ // Chain to wrong root results errors.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{megaLeaf, comodoIntermediate, comodoRoot},

		function: "QueueAllCertsInChain",
		expectedErrs: []errorType{
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
		},
	},
	{ // Chain without correct root containing loop results in error.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testC, testB, testA},

		function: "QueueAllCertsInChain",
		expectedErrs: []errorType{
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
		},
	},
	{ // Incomplete chain successfully logged.
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testLeaf},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // Garbled chain (with a leaf that has no chain to our roots)
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{megaLeaf, googleLeaf, thawteIntermediate, verisignRoot},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
			{"Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
		},
	},
	{ // Garbled chain (with a leaf that has a chain to our roots)
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{testLeaf, megaLeaf, googleLeaf, thawteIntermediate, comodoRoot},

		function: "QueueAllCertsInChain",
		expLoggedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
			{"Google", "Thawte", "VeriSign"},
			{"Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{
			VerifyFailed,
			VerifyFailed, FixFailed,
			VerifyFailed, FixFailed,
		},
	},
}

func TestNewFixAndLog(t *testing.T) {
	// Test that expected chains are logged when adding a chain using QueueChain()
	for i, test := range newFixAndLogTests {
		seen := make([]bool, len(test.expLoggedChains))
		errors := make(chan *FixError)
		fl := NewFixAndLog(1, 1, errors, &http.Client{Transport: &testRoundTripper{t: t, test: &test, testIndex: i, seen: seen}}, test.url, newNilLimiter(), false)

		var wg sync.WaitGroup
		wg.Add(1)
		go testErrors(t, i, test.expectedErrs, errors, &wg)
		switch test.function {
		case "QueueChain":
			fl.QueueChain(extractTestChain(t, i, test.chain))
		case "QueueAllCertsInChain":
			fl.QueueAllCertsInChain(extractTestChain(t, i, test.chain))
		}
		fl.Wait()
		close(errors)
		wg.Wait()

		// Check that no chains that were expected to be logged were not.
		for j, val := range seen {
			if !val {
				t.Errorf("#%d: Expected chain was not logged: %s", i, strings.Join(test.expLoggedChains[j], " -> "))
			}
		}
	}
}

var fixAndLogQueueTests = []fixAndLogTest{
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		expectedCert:  "Google",
		expectedChain: []string{"Google", "Thawte", "VeriSign"},
		expectedRoots: []string{verisignRoot, testRoot},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, googleLeaf, thawteIntermediate, verisignRoot},

		expectedCert:  "Google",
		expectedChain: []string{"Google", "Thawte", "VeriSign"},
		expectedRoots: []string{verisignRoot, testRoot},
	},
	{ // Test passing a nil chain to FixAndLog.QueueChain()
		url: "https://ct.googleapis.com/pilot",
	},
}

func testQueueAllCertsInChain(t *testing.T, i int, test *fixAndLogTest, fl *FixAndLog) {
	defer fl.wg.Done()
	seen := make([]bool, len(test.expectedChain))
NextToFix:
	for fix := range fl.fixer.toFix {
		// Check fix.chain is the chain that's expected.
		matchTestChain(t, i, test.expectedChain, fix.chain.certs)
		//Check fix.roots are the roots that are expected for the given url.
		matchTestRoots(t, i, test.expectedRoots, fix.roots)
		for j, expCert := range test.expectedChain {
			if seen[j] {
				continue
			}
			if strings.Contains(nameToKey(&fix.cert.Subject), expCert) {
				seen[j] = true
				continue NextToFix
			}
		}
		t.Errorf("#%d: Queued certificate %s was not expected", i, nameToKey(&fix.cert.Subject))
	}
	for j, val := range seen {
		if !val {
			t.Errorf("#%d: Expected certificate %s was not queued", i, test.expectedChain[j])
		}
	}
}

func TestQueueAllCertsInChain(t *testing.T) {
	for i, test := range fixAndLogQueueTests {
		f := &Fixer{toFix: make(chan *toFix)}
		l := &Logger{
			url:           test.url,
			client:        &http.Client{Transport: &testRoundTripper{}},
			postCertCache: newLockedMap(),
		}
		fl := &FixAndLog{fixer: f, chains: make(chan []*x509.Certificate), logger: l, done: newLockedMap()}

		fl.wg.Add(1)
		go testQueueAllCertsInChain(t, i, &test, fl)
		fl.QueueAllCertsInChain(extractTestChain(t, i, test.chain))
		fl.Wait()
	}
}

func testFixAndLogQueueChain(t *testing.T, i int, test *fixAndLogTest, fl *FixAndLog) {
	defer fl.wg.Done()

	fix, ok := <-fl.fixer.toFix
	if ok {
		// Check fix.cert is the cert that's expected.
		if !strings.Contains(nameToKey(&fix.cert.Subject), test.expectedCert) {
			t.Errorf("#%d: Expected cert does not match queued cert", i)
		}

		// Check fix.chain is the chain that's expected.
		matchTestChain(t, i, test.expectedChain, fix.chain.certs)

		//Check fix.roots are the roots that are expected for the given url.
		matchTestRoots(t, i, test.expectedRoots, fix.roots)
	}
}

func TestFixAndLogQueueChain(t *testing.T) {
	for i, test := range fixAndLogQueueTests {
		f := &Fixer{toFix: make(chan *toFix)}
		l := &Logger{
			url:           test.url,
			client:        &http.Client{Transport: &testRoundTripper{}},
			postCertCache: newLockedMap(),
		}
		fl := &FixAndLog{fixer: f, chains: make(chan []*x509.Certificate), logger: l, done: newLockedMap()}

		fl.wg.Add(1)
		go testFixAndLogQueueChain(t, i, &test, fl)
		fl.QueueChain(extractTestChain(t, i, test.chain))
		fl.Wait()
	}
}
