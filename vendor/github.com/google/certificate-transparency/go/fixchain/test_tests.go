package fixchain

type fixTest struct {
	cert  string
	chain []string
	roots []string

	function       string
	expectedChains [][]string
	expectedErrs   []errorType
}

var handleChainTests = []fixTest{
	// handleChain()
	{ // Correct chain returns chain
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function: "handleChain",
		expectedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
	},
	{ // No roots results in an error
		cert:  googleLeaf,
		chain: []string{thawteIntermediate, verisignRoot},

		function:     "handleChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // No roots where chain that will be built contains a loop results in error
		cert:  testC,
		chain: []string{testB, testA},

		function:     "handleChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // Incomplete chain returns a fixed chain
		cert:  googleLeaf,
		roots: []string{verisignRoot},

		function: "handleChain",
		expectedChains: [][]string{
			{"Google", "Thawte", "VeriSign"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{
		cert:  testLeaf,
		roots: []string{testRoot},

		function: "handleChain",
		expectedChains: [][]string{
			{"Leaf", "Intermediate2", "Intermediate1", "CA"},
		},
		expectedErrs: []errorType{VerifyFailed},
	},
	{ // The wrong intermediate and root results in an error
		cert:  megaLeaf,
		chain: []string{thawteIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "handleChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
	{ // The wrong root results in an error
		cert:  megaLeaf,
		chain: []string{comodoIntermediate, verisignRoot},
		roots: []string{verisignRoot},

		function:     "handleChain",
		expectedErrs: []errorType{VerifyFailed, FixFailed},
	},
}

type postTest struct {
	url   string
	chain []string

	urlScheme string
	urlHost   string
	urlPath   string

	ferr         *FixError
	expectedErrs []errorType
}

var postTests = []postTest{
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		urlScheme: "https",
		urlHost:   "ct.googleapis.com",
		urlPath:   "/pilot/ct/v1/add-chain",

		ferr: &FixError{Type: None},
	},
	{ // Empty chain
		url: "https://ct.googleapis.com/pilot",

		urlScheme: "https",
		urlHost:   "ct.googleapis.com",
		urlPath:   "/pilot/ct/v1/add-chain",

		ferr: &FixError{Type: None},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		ferr:         &FixError{Type: PostFailed},
		expectedErrs: []errorType{PostFailed},
	},
	{
		url:   "https://ct.googleapis.com/pilot",
		chain: []string{googleLeaf, thawteIntermediate, verisignRoot},

		ferr:         &FixError{Type: LogPostFailed},
		expectedErrs: []errorType{LogPostFailed},
	},
}

type fixAndLogTest struct {
	url   string
	chain []string

	// Expected items that will be queued to be fixed then logged
	expectedCert  string
	expectedChain []string
	expectedRoots []string

	function        string
	expLoggedChains [][]string
	expectedErrs    []errorType
}
