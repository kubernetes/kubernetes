package fixchain

import (
	"net/http"
	"sync"
	"testing"
)

// NewLogger() test
func TestNewLogger(t *testing.T) {
	// Test single chain posts.
	for i, test := range postTests {
		errors := make(chan *FixError)
		var wg sync.WaitGroup
		wg.Add(1)
		go testErrors(t, i, test.expectedErrs, errors, &wg)

		l := NewLogger(1, test.url, errors, &http.Client{Transport: &postTestRoundTripper{t: t, test: &test, testIndex: i}}, newNilLimiter(), false)

		l.QueueChain(extractTestChain(t, i, test.chain))
		l.Wait()

		close(l.errors)
		wg.Wait()

		// Check logger caching.
		if test.chain != nil {
			if test.ferr.Type == None && !l.postCertCache.get(hash(GetTestCertificateFromPEM(t, test.chain[0]))) {
				t.Errorf("#%d: leaf certificate not cached", i)
			}
			if !l.postChainCache.get(hashChain(extractTestChain(t, i, test.chain))) {
				t.Errorf("#%d: chain not cached", i)
			}
		}
	}
}

// NewLogger() test
func TestNewLoggerCaching(t *testing.T) {
	// Test logging multiple chains by looking at caching.
	newLoggerTest := struct {
		url          string
		chains       [][]string
		expectedErrs []errorType
	}{
		"https://ct.googleapis.com/pilot",
		[][]string{
			{googleLeaf, thawteIntermediate, verisignRoot},
			{googleLeaf, thawteIntermediate, verisignRoot},
			{googleLeaf, thawteIntermediate},
			{testLeaf, testIntermediate2, testIntermediate1, testRoot},
		},
		[]errorType{},
	}

	errors := make(chan *FixError)
	var wg sync.WaitGroup
	wg.Add(1)
	go testErrors(t, 0, newLoggerTest.expectedErrs, errors, &wg)

	l := NewLogger(5, newLoggerTest.url, errors, &http.Client{Transport: &newLoggerTestRoundTripper{}}, newNilLimiter(), false)

	for _, chain := range newLoggerTest.chains {
		l.QueueChain(extractTestChain(t, 0, chain))
	}
	l.Wait()
	close(l.errors)
	wg.Wait()

	// Check logger caching.
	seen := make(map[[hashSize]byte]bool)
	for i, chain := range newLoggerTest.chains {
		leafHash := hash(GetTestCertificateFromPEM(t, chain[0]))
		if !l.postCertCache.get(leafHash) {
			t.Errorf("Chain %d: leaf certificate not cached", i)
		}
		if !seen[leafHash] && !l.postChainCache.get(hashChain(extractTestChain(t, 0, chain))) {
			t.Errorf("Chain %d: chain not cached", i)
		}
		seen[leafHash] = true
	}
}

// Logger.postServer() test
func TestPostServer(t *testing.T) {
	for i, test := range postTests {
		errors := make(chan *FixError)
		l := &Logger{
			url:            test.url,
			client:         &http.Client{Transport: &postTestRoundTripper{t: t, test: &test, testIndex: i}},
			toPost:         make(chan *toPost),
			errors:         errors,
			limiter:        newNilLimiter(),
			postCertCache:  newLockedMap(),
			postChainCache: newLockedMap(),
		}
		var wg sync.WaitGroup
		wg.Add(1)
		go testErrors(t, i, test.expectedErrs, errors, &wg)

		go l.postServer()
		l.QueueChain(extractTestChain(t, i, test.chain))
		l.Wait()

		close(l.errors)
		wg.Wait()
	}
}

// Logger.IsPosted() test
func TestIsPosted(t *testing.T) {
	isPostedTests := []struct {
		cert     string
		expected bool
	}{
		{
			googleLeaf,
			true,
		},
		{
			megaLeaf,
			true,
		},
		{
			testLeaf,
			false,
		},
		{
			testC,
			false,
		},
	}

	l := &Logger{postCertCache: newLockedMap()}
	l.postCertCache.set(hash(GetTestCertificateFromPEM(t, googleLeaf)), true)
	l.postCertCache.set(hash(GetTestCertificateFromPEM(t, megaLeaf)), true)
	l.postCertCache.set(hash(GetTestCertificateFromPEM(t, testLeaf)), false)

	for i, test := range isPostedTests {
		if l.IsPosted(GetTestCertificateFromPEM(t, test.cert)) != test.expected {
			t.Errorf("#%d: received %t, expected %t", i, !test.expected, test.expected)
		}
	}
}

// Logger.QueueChain() tests
type loggerQueueTest struct {
	chain         []string
	expectedChain []string
}

var loggerQueueTests = []loggerQueueTest{
	{
		chain:         []string{googleLeaf, thawteIntermediate, verisignRoot},
		expectedChain: []string{"Google", "Thawte", "VeriSign"},
	},
	{ // Add the same chain a second time to test chain caching.
		// Note that if chain caching isn't working correctly, the test will hang.
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},
	},
}

func testLoggerQueueChain(t *testing.T, i int, qt *loggerQueueTest, l *Logger) {
	defer l.wg.Done()
	if qt.expectedChain != nil {
		post := <-l.toPost
		matchTestChain(t, i, qt.expectedChain, post.chain)
		l.wg.Done() // Required as logger wg is incremented internally every time a toPost is added to the queue.
	}
}

func TestLoggerQueueChain(t *testing.T) {
	ch := make(chan *toPost)
	defer close(ch)
	l := &Logger{toPost: ch, postCertCache: newLockedMap(), postChainCache: newLockedMap()}

	for i, qt := range loggerQueueTests {
		l.wg.Add(1)
		go testLoggerQueueChain(t, i, &qt, l)
		chain := extractTestChain(t, i, qt.chain)
		l.QueueChain(chain)
		l.wg.Wait()
	}
}

// Logger.RootCerts() test
func TestRootCerts(t *testing.T) {
	rootCertsTests := []struct {
		url           string
		expectedRoots []string
	}{
		{
			"https://ct.googleapis.com/pilot",
			[]string{verisignRoot, comodoRoot}, // These are not the actual roots for the pilot CT log, this is just for testing purposes.
		},
	}

	for i, test := range rootCertsTests {
		l := &Logger{
			url:    test.url,
			client: &http.Client{Transport: &rootCertsTestRoundTripper{}},
		}
		roots := l.RootCerts()
		matchTestRoots(t, i, test.expectedRoots, roots)
	}
}
