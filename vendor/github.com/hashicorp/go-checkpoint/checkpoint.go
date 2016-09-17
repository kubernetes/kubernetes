// checkpoint is a package for checking version information and alerts
// for a HashiCorp product.
package checkpoint

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	mrand "math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/hashicorp/go-cleanhttp"
)

var magicBytes [4]byte = [4]byte{0x35, 0x77, 0x69, 0xFB}

// CheckParams are the parameters for configuring a check request.
type CheckParams struct {
	// Product and version are used to lookup the correct product and
	// alerts for the proper version. The version is also used to perform
	// a version check.
	Product string
	Version string

	// Arch and OS are used to filter alerts potentially only to things
	// affecting a specific os/arch combination. If these aren't specified,
	// they'll be automatically filled in.
	Arch string
	OS   string

	// Signature is some random signature that should be stored and used
	// as a cookie-like value. This ensures that alerts aren't repeated.
	// If the signature is changed, repeat alerts may be sent down. The
	// signature should NOT be anything identifiable to a user (such as
	// a MAC address). It should be random.
	//
	// If SignatureFile is given, then the signature will be read from this
	// file. If the file doesn't exist, then a random signature will
	// automatically be generated and stored here. SignatureFile will be
	// ignored if Signature is given.
	Signature     string
	SignatureFile string

	// CacheFile, if specified, will cache the result of a check. The
	// duration of the cache is specified by CacheDuration, and defaults
	// to 48 hours if not specified. If the CacheFile is newer than the
	// CacheDuration, than the Check will short-circuit and use those
	// results.
	//
	// If the CacheFile directory doesn't exist, it will be created with
	// permissions 0755.
	CacheFile     string
	CacheDuration time.Duration

	// Force, if true, will force the check even if CHECKPOINT_DISABLE
	// is set. Within HashiCorp products, this is ONLY USED when the user
	// specifically requests it. This is never automatically done without
	// the user's consent.
	Force bool
}

// CheckResponse is the response for a check request.
type CheckResponse struct {
	Product             string
	CurrentVersion      string `json:"current_version"`
	CurrentReleaseDate  int    `json:"current_release_date"`
	CurrentDownloadURL  string `json:"current_download_url"`
	CurrentChangelogURL string `json:"current_changelog_url"`
	ProjectWebsite      string `json:"project_website"`
	Outdated            bool   `json:"outdated"`
	Alerts              []*CheckAlert
}

// CheckAlert is a single alert message from a check request.
//
// These never have to be manually constructed, and are typically populated
// into a CheckResponse as a result of the Check request.
type CheckAlert struct {
	ID      int
	Date    int
	Message string
	URL     string
	Level   string
}

// Check checks for alerts and new version information.
func Check(p *CheckParams) (*CheckResponse, error) {
	if disabled := os.Getenv("CHECKPOINT_DISABLE"); disabled != "" && !p.Force {
		return &CheckResponse{}, nil
	}

	// If we have a cached result, then use that
	if r, err := checkCache(p.Version, p.CacheFile, p.CacheDuration); err != nil {
		return nil, err
	} else if r != nil {
		defer r.Close()
		return checkResult(r)
	}

	var u url.URL

	if p.Arch == "" {
		p.Arch = runtime.GOARCH
	}
	if p.OS == "" {
		p.OS = runtime.GOOS
	}

	// If we're given a SignatureFile, then attempt to read that.
	signature := p.Signature
	if p.Signature == "" && p.SignatureFile != "" {
		var err error
		signature, err = checkSignature(p.SignatureFile)
		if err != nil {
			return nil, err
		}
	}

	v := u.Query()
	v.Set("version", p.Version)
	v.Set("arch", p.Arch)
	v.Set("os", p.OS)
	v.Set("signature", signature)

	u.Scheme = "https"
	u.Host = "checkpoint-api.hashicorp.com"
	u.Path = fmt.Sprintf("/v1/check/%s", p.Product)
	u.RawQuery = v.Encode()

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Accept", "application/json")
	req.Header.Add("User-Agent", "HashiCorp/go-checkpoint")

	client := cleanhttp.DefaultClient()
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Unknown status: %d", resp.StatusCode)
	}

	var r io.Reader = resp.Body
	if p.CacheFile != "" {
		// Make sure the directory holding our cache exists.
		if err := os.MkdirAll(filepath.Dir(p.CacheFile), 0755); err != nil {
			return nil, err
		}

		// We have to cache the result, so write the response to the
		// file as we read it.
		f, err := os.Create(p.CacheFile)
		if err != nil {
			return nil, err
		}

		// Write the cache header
		if err := writeCacheHeader(f, p.Version); err != nil {
			f.Close()
			os.Remove(p.CacheFile)
			return nil, err
		}

		defer f.Close()
		r = io.TeeReader(r, f)
	}

	return checkResult(r)
}

// CheckInterval is used to check for a response on a given interval duration.
// The interval is not exact, and checks are randomized to prevent a thundering
// herd. However, it is expected that on average one check is performed per
// interval. The returned channel may be closed to stop background checks.
func CheckInterval(p *CheckParams, interval time.Duration, cb func(*CheckResponse, error)) chan struct{} {
	doneCh := make(chan struct{})

	if disabled := os.Getenv("CHECKPOINT_DISABLE"); disabled != "" {
		return doneCh
	}

	go func() {
		for {
			select {
			case <-time.After(randomStagger(interval)):
				resp, err := Check(p)
				cb(resp, err)
			case <-doneCh:
				return
			}
		}
	}()

	return doneCh
}

// randomStagger returns an interval that is between 3/4 and 5/4 of
// the given interval. The expected value is the interval.
func randomStagger(interval time.Duration) time.Duration {
	stagger := time.Duration(mrand.Int63()) % (interval / 2)
	return 3*(interval/4) + stagger
}

func checkCache(current string, path string, d time.Duration) (io.ReadCloser, error) {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist, not a problem
			return nil, nil
		}

		return nil, err
	}

	if d == 0 {
		d = 48 * time.Hour
	}

	if fi.ModTime().Add(d).Before(time.Now()) {
		// Cache is busted, delete the old file and re-request. We ignore
		// errors here because re-creating the file is fine too.
		os.Remove(path)
		return nil, nil
	}

	// File looks good so far, open it up so we can inspect the contents.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	// Check the signature of the file
	var sig [4]byte
	if err := binary.Read(f, binary.LittleEndian, sig[:]); err != nil {
		f.Close()
		return nil, err
	}
	if !reflect.DeepEqual(sig, magicBytes) {
		// Signatures don't match. Reset.
		f.Close()
		return nil, nil
	}

	// Check the version. If it changed, then rewrite
	var length uint32
	if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
		f.Close()
		return nil, err
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(f, data); err != nil {
		f.Close()
		return nil, err
	}
	if string(data) != current {
		// Version changed, reset
		f.Close()
		return nil, nil
	}

	return f, nil
}

func checkResult(r io.Reader) (*CheckResponse, error) {
	var result CheckResponse
	dec := json.NewDecoder(r)
	if err := dec.Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func checkSignature(path string) (string, error) {
	_, err := os.Stat(path)
	if err == nil {
		// The file exists, read it out
		sigBytes, err := ioutil.ReadFile(path)
		if err != nil {
			return "", err
		}

		// Split the file into lines
		lines := strings.SplitN(string(sigBytes), "\n", 2)
		if len(lines) > 0 {
			return strings.TrimSpace(lines[0]), nil
		}
	}

	// If this isn't a non-exist error, then return that.
	if !os.IsNotExist(err) {
		return "", err
	}

	// The file doesn't exist, so create a signature.
	var b [16]byte
	n := 0
	for n < 16 {
		n2, err := rand.Read(b[n:])
		if err != nil {
			return "", err
		}

		n += n2
	}
	signature := fmt.Sprintf(
		"%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])

	// Make sure the directory holding our signature exists.
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", err
	}

	// Write the signature
	if err := ioutil.WriteFile(path, []byte(signature+"\n\n"+userMessage+"\n"), 0644); err != nil {
		return "", err
	}

	return signature, nil
}

func writeCacheHeader(f io.Writer, v string) error {
	// Write our signature first
	if err := binary.Write(f, binary.LittleEndian, magicBytes); err != nil {
		return err
	}

	// Write out our current version length
	var length uint32 = uint32(len(v))
	if err := binary.Write(f, binary.LittleEndian, length); err != nil {
		return err
	}

	_, err := f.Write([]byte(v))
	return err
}

// userMessage is suffixed to the signature file to provide feedback.
var userMessage = `
This signature is a randomly generated UUID used to de-duplicate
alerts and version information. This signature is random, it is
not based on any personally identifiable information. To create
a new signature, you can simply delete this file at any time.
See the documentation for the software using Checkpoint for more
information on how to disable it.
`
