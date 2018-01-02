package fixchain

import (
	"net/http"
	"sync"
	"testing"

	"github.com/google/certificate-transparency/go/x509"
)

// NewFixer() test
func TestNewFixer(t *testing.T) {
	chains := make(chan []*x509.Certificate)
	errors := make(chan *FixError)

	var expectedChains [][]string
	var expectedErrs []errorType
	for _, test := range handleChainTests {
		expectedChains = append(expectedChains, test.expectedChains...)
		expectedErrs = append(expectedErrs, test.expectedErrs...)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go testChains(t, 0, expectedChains, chains, &wg)
	go testErrors(t, 0, expectedErrs, errors, &wg)

	f := NewFixer(10, chains, errors, &http.Client{Transport: &testRoundTripper{}}, false)
	for _, test := range handleChainTests {
		f.QueueChain(GetTestCertificateFromPEM(t, test.cert),
			extractTestChain(t, 0, test.chain), extractTestRoots(t, 0, test.roots))
	}
	f.Wait()

	close(chains)
	close(errors)
	wg.Wait()
}

// Fixer.fixServer() test
func TestFixServer(t *testing.T) {
	cache := &urlCache{cache: newLockedCache(), client: &http.Client{Transport: &testRoundTripper{}}}
	f := &Fixer{cache: cache}

	var wg sync.WaitGroup
	fixServerTests := handleChainTests

	// Pass chains to be fixed one at a time to fixServer and check the chain
	// and errors produced are correct.
	for i, fst := range fixServerTests {
		chains := make(chan []*x509.Certificate)
		errors := make(chan *FixError)
		f.toFix = make(chan *toFix)
		f.chains = chains
		f.errors = errors

		wg.Add(2)
		go testChains(t, i, fst.expectedChains, chains, &wg)
		go testErrors(t, i, fst.expectedErrs, errors, &wg)

		f.wg.Add(1)
		go f.fixServer()
		f.QueueChain(GetTestCertificateFromPEM(t, fst.cert),
			extractTestChain(t, i, fst.chain), extractTestRoots(t, i, fst.roots))
		f.Wait()

		close(chains)
		close(errors)
		wg.Wait()
	}

	// Pass multiple chains to be fixed to fixServer and check the chain and
	// errors produced are correct.
	chains := make(chan []*x509.Certificate)
	errors := make(chan *FixError)
	f.toFix = make(chan *toFix)
	f.chains = chains
	f.errors = errors

	var expectedChains [][]string
	var expectedErrs []errorType
	for _, fst := range fixServerTests {
		expectedChains = append(expectedChains, fst.expectedChains...)
		expectedErrs = append(expectedErrs, fst.expectedErrs...)
	}

	i := len(fixServerTests)
	wg.Add(2)
	go testChains(t, i, expectedChains, chains, &wg)
	go testErrors(t, i, expectedErrs, errors, &wg)

	f.wg.Add(1)
	go f.fixServer()
	for _, fst := range fixServerTests {
		f.QueueChain(GetTestCertificateFromPEM(t, fst.cert),
			extractTestChain(t, i, fst.chain), extractTestRoots(t, i, fst.roots))
	}
	f.Wait()

	close(chains)
	close(errors)
	wg.Wait()
}

// Fixer.updateCounters() tests
func TestUpdateCounters(t *testing.T) {
	counterTests := []struct {
		errors           []errorType
		reconstructed    uint32
		notReconstructed uint32
		fixed            uint32
		notFixed         uint32
	}{
		{[]errorType{}, 1, 0, 0, 0},
		{[]errorType{VerifyFailed}, 0, 1, 1, 0},
		{[]errorType{VerifyFailed, FixFailed}, 0, 1, 0, 1},

		{[]errorType{ParseFailure}, 1, 0, 0, 0},
		{[]errorType{ParseFailure, VerifyFailed}, 0, 1, 1, 0},
		{[]errorType{ParseFailure, VerifyFailed, FixFailed}, 0, 1, 0, 1},
	}

	for i, test := range counterTests {
		f := &Fixer{}
		var ferrs []*FixError
		for _, err := range test.errors {
			ferrs = append(ferrs, &FixError{Type: err})
		}
		f.updateCounters(ferrs)

		if f.reconstructed != test.reconstructed {
			t.Errorf("#%d: Incorrect value for reconstructed, wanted %d, got %d", i, test.reconstructed, f.reconstructed)
		}
		if f.notReconstructed != test.notReconstructed {
			t.Errorf("#%d: Incorrect value for notReconstructed, wanted %d, got %d", i, test.notReconstructed, f.notReconstructed)
		}
		if f.fixed != test.fixed {
			t.Errorf("#%d: Incorrect value for fixed, wanted %d, got %d", i, test.fixed, f.fixed)
		}
		if f.notFixed != test.notFixed {
			t.Errorf("#%d: Incorrect value for notFixed, wanted %d, got %d", i, test.notFixed, f.notFixed)
		}
	}
}

// Fixer.QueueChain() tests
type fixerQueueTest struct {
	cert  string
	chain []string
	roots []string

	dchain []string
}

var fixerQueueTests = []fixerQueueTest{
	{
		cert:  googleLeaf,
		chain: []string{verisignRoot, thawteIntermediate},
		roots: []string{verisignRoot},

		dchain: []string{"VeriSign", "Thawte"},
	},
	{
		cert:  googleLeaf,
		chain: []string{verisignRoot, verisignRoot, thawteIntermediate},
		roots: []string{verisignRoot},

		dchain: []string{"VeriSign", "Thawte"},
	},
	{
		cert:  googleLeaf,
		roots: []string{verisignRoot},

		dchain: []string{},
	},
}

func testFixerQueueChain(t *testing.T, i int, qt *fixerQueueTest, f *Fixer) {
	defer f.wg.Done()
	fix := <-f.toFix
	// Check the deduped chain
	matchTestChain(t, i, qt.dchain, fix.chain.certs)
}

func TestFixerQueueChain(t *testing.T) {
	ch := make(chan *toFix)
	defer close(ch)
	f := &Fixer{toFix: ch}

	for i, qt := range fixerQueueTests {
		f.wg.Add(1)
		go testFixerQueueChain(t, i, &qt, f)
		chain := extractTestChain(t, i, qt.chain)
		roots := extractTestRoots(t, i, qt.roots)
		f.QueueChain(GetTestCertificateFromPEM(t, qt.cert), chain, roots)
		f.wg.Wait()
	}
}
