package bundler

// This test file contains tests on checking the correctness of BundleFromRemote
import (
	"flag"
	"testing"

	"github.com/cloudflare/cfssl/ubiquity"
)

var shouldTestSNI bool

func init() {
	flag.BoolVar(&shouldTestSNI, "test-sni", false, "run the SNI tests")
	flag.Parse()
}

// remoteTest defines a test case for BundleFromRemote. Hostname and ip are the test inputs.
// bundlerConstructor points the bundler ctor and errorCallback handles the error checking.
type remoteTest struct {
	hostname           string
	ip                 string
	bundlerConstructor func(*testing.T) (b *Bundler)
	errorCallback      func(*testing.T, error)
	bundleCallback     func(*testing.T, *Bundle)
}

const (
	ValidSSLSite           = "google.com"
	SelfSignedSSLSite      = "cacert.org"
	MismatchedHostnameSite = "www.capitol.state.tx.us"
	ECCCertSite            = "benflare.us"
	InvalidSite            = "cloudflare1337.com"
	ValidSNI               = "alice.sni.velox.ch"
	ValidSNIWildcard       = "cloudflare.sni.velox.ch"
	SNISANWildcard         = "*.sni.velox.ch"
	ValidSNIIP             = "85.25.46.13"
	InvalidIP              = "300.300.300.300"
)

func getBundleHostnameChecker(hostname string) func(*testing.T, *Bundle) {
	return func(t *testing.T, bundle *Bundle) {
		if bundle == nil {
			t.Fatalf("Nil bundle returned")
		}
		var found = false
		for _, h := range bundle.Hostnames {
			if h == hostname {
				found = true
			}
		}
		if !found {
			t.Errorf("hostname expected but not found: %s", hostname)
		}
	}
}

// test cases of BundleFromRemote
var remoteTests = []remoteTest{
	{
		hostname:           ValidSSLSite,
		bundlerConstructor: newBundler,
		errorCallback:      nil,
	},
	{
		hostname:           SelfSignedSSLSite,
		bundlerConstructor: newBundler,
		errorCallback:      ExpectErrorMessages([]string{`"code":12`}), // only check it is a 12xx error
	},
	{
		hostname:           MismatchedHostnameSite,
		bundlerConstructor: newBundler,
		errorCallback:      ExpectErrorMessages([]string{`"code":12`}), // only check it is a 12xx error
	},
	{
		hostname:           InvalidSite,
		bundlerConstructor: newBundler,
		errorCallback:      ExpectErrorMessages([]string{`"code":6000`, "dial tcp: lookup cloudflare1337.com"}),
	},
	{
		hostname:           InvalidIP,
		bundlerConstructor: newBundler,
		errorCallback:      ExpectErrorMessages([]string{`"code":6000`, "dial tcp: lookup 300.300.300.300"}),
	},
	{
		ip:                 InvalidIP,
		bundlerConstructor: newBundler,
		errorCallback:      ExpectErrorMessages([]string{`"code":6000`, "dial tcp: lookup 300.300.300.300"}),
	},
}

// TestBundleFromRemote goes through the test cases defined in remoteTests and run them through. See above for test case definitions.
func TestBundleFromRemote(t *testing.T) {
	for _, bf := range []BundleFlavor{Ubiquitous, Optimal} {
		for _, test := range remoteTests {
			b := test.bundlerConstructor(t)
			bundle, err := b.BundleFromRemote(test.hostname, test.ip, bf)
			if test.errorCallback != nil {
				test.errorCallback(t, err)
			} else {
				if err != nil {
					t.Fatal("expected no error. but an error occurred", err.Error())
				}
				if test.bundleCallback != nil {
					test.bundleCallback(t, bundle)
				}
			}
		}
	}
}

var remoteSNITests = []remoteTest{
	{
		hostname:           ValidSNI,
		bundlerConstructor: newBundler,
		errorCallback:      nil,
		bundleCallback:     getBundleHostnameChecker(ValidSNI),
	},
	{
		hostname:           ValidSNIWildcard,
		bundlerConstructor: newBundler,
		errorCallback:      nil,
		bundleCallback:     getBundleHostnameChecker(SNISANWildcard),
	},
	{
		hostname:           ValidSNI,
		ip:                 ValidSNIIP,
		bundlerConstructor: newBundler,
		errorCallback:      nil,
		bundleCallback:     getBundleHostnameChecker(ValidSNI),
	},
	{
		hostname:           ValidSNIWildcard,
		ip:                 ValidSNIIP,
		bundlerConstructor: newBundler,
		errorCallback:      nil,
		bundleCallback:     getBundleHostnameChecker(SNISANWildcard),
	},
}

// TestBundleFromRemoteSNI goes through the test cases defined in remoteSNITests and run them through. See above for test case definitions.
func TestBundleFromRemoteSNI(t *testing.T) {
	if !shouldTestSNI {
		t.Skip()
	}
	for _, bf := range []BundleFlavor{Ubiquitous, Optimal} {
		for _, test := range remoteSNITests {
			b := test.bundlerConstructor(t)
			bundle, err := b.BundleFromRemote(test.hostname, test.ip, bf)
			if test.errorCallback != nil {
				test.errorCallback(t, err)
			} else {
				if err != nil {
					t.Errorf("expected no error. but an error occurred: %s", err.Error())
				}
				if test.bundleCallback != nil {
					test.bundleCallback(t, bundle)
				}
			}
		}
	}
}

func TestBundleFromRemoteFlavor(t *testing.T) {
	b := newBundler(t)
	ubiquity.Platforms = nil
	ubiquity.LoadPlatforms(testMetadata)

	bundle, err := b.BundleFromRemote(ECCCertSite, "", Ubiquitous)
	if err != nil {
		t.Fatalf("expected no error. but an error occurred: %s", err.Error())
	}
	if len(bundle.Chain) != 3 {
		t.Error("expected 3-cert bundle. Got ", len(bundle.Chain))
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Error("expected no untrusted platforms. Got ", bundle.Status.Untrusted)
	}

	bundle, err = b.BundleFromRemote(ECCCertSite, "", Optimal)
	if err != nil {
		t.Errorf("expected no error. but an error occurred: %s", err.Error())
	}
	if len(bundle.Chain) != 2 {
		t.Error("expected 2-cert bundle. Got ", len(bundle.Chain))
	}

}
