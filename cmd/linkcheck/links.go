/*
Copyright 2015 The Kubernetes Authors.

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

// This tool extracts the links from types.go and .md files, visits the link and
// checks the status code of the response.
// Usage:
// $ linkcheck --root-dir=${ROOT}

package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/mvdan/xurls"
	flag "github.com/spf13/pflag"
)

var (
	rootDir    = flag.String("root-dir", "", "Root directory containing documents to be processed.")
	fileSuffix = flag.StringSlice("file-suffix", []string{"types.go", ".md"}, "suffix of files to be checked")
	// URLs matching the patterns in the regWhiteList won't be checked. Patterns
	// of dummy URLs should be added to the list to avoid false alerts. Also,
	// patterns of URLs that we don't care about can be added here to improve
	// efficiency.
	regWhiteList = []*regexp.Regexp{
		regexp.MustCompile(`https://kubernetes-site\.appspot\.com`),
		// skip url that doesn't start with an English alphabet, e.g., URLs with IP addresses.
		regexp.MustCompile(`https?://[^A-Za-z].*`),
		regexp.MustCompile(`https?://localhost.*`),
	}
	// URLs listed in the fullURLWhiteList won't be checked. This separated from
	// the RegWhiteList to improve efficiency. This list includes dummy URLs that
	// are hard to be generalized by a regex, and URLs that will cause false alerts.
	fullURLWhiteList = map[string]struct{}{
		"http://github.com/some/repo.git": {},
		// This URL returns 404 when visited by this tool, but it works fine if visited by a browser.
		"http://stackoverflow.com/questions/ask?tags=kubernetes":                                            {},
		"https://github.com/$YOUR_GITHUB_USERNAME/kubernetes.git":                                           {},
		"https://github.com/$YOUR_GITHUB_USERNAME/kubernetes":                                               {},
		"http://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/darwin/amd64/kubectl": {},
		// It seems this server expects certain User-Agent value, it works fine with Chrome, but returns 404 if we issue a plain cURL to it.
		"http://supervisord.org/":         {},
		"http://kubernetes.io/vX.Y/docs":  {},
		"http://kubernetes.io/vX.Y/docs/": {},
		"http://kubernetes.io/vX.Y/":      {},
	}

	visitedURLs    = map[string]struct{}{}
	htmlpreviewReg = regexp.MustCompile(`https://htmlpreview\.github\.io/\?`)
	httpOrhttpsReg = regexp.MustCompile(`https?.*`)
)

func newWalkFunc(invalidLink *bool, client *http.Client) filepath.WalkFunc {
	return func(filePath string, info os.FileInfo, err error) error {
		hasSuffix := false
		for _, suffix := range *fileSuffix {
			hasSuffix = hasSuffix || strings.HasSuffix(info.Name(), suffix)
		}
		if !hasSuffix {
			return nil
		}

		fileBytes, err := ioutil.ReadFile(filePath)
		if err != nil {
			return err
		}
		foundInvalid := false
		allURLs := xurls.Strict.FindAll(fileBytes, -1)
		fmt.Fprintf(os.Stdout, "\nChecking file %s\n", filePath)
	URL:
		for _, URL := range allURLs {
			// Don't check non http/https URL
			if !httpOrhttpsReg.Match(URL) {
				continue
			}
			for _, whiteURL := range regWhiteList {
				if whiteURL.Match(URL) {
					continue URL
				}
			}
			if _, found := fullURLWhiteList[string(URL)]; found {
				continue
			}
			// remove the htmlpreview Prefix
			processedURL := htmlpreviewReg.ReplaceAll(URL, []byte{})

			// check if we have visited the URL.
			if _, found := visitedURLs[string(processedURL)]; found {
				continue
			}
			visitedURLs[string(processedURL)] = struct{}{}

			retry := 0
			const maxRetry int = 3
			backoff := 100
			for retry < maxRetry {
				fmt.Fprintf(os.Stdout, "Visiting %s\n", string(processedURL))
				// Use verb HEAD to increase efficiency. However, some servers
				// do not handle HEAD well, so we need to try a GET to avoid
				// false alert.
				resp, err := client.Head(string(processedURL))
				// URLs with mock host or mock port will cause error. If we report
				// the error here, people need to add the mock URL to the white
				// list every time they add a mock URL, which will be a maintenance
				// nightmare. Hence, we decide to only report 404 to catch the
				// cases where host and port are legit, but path is not, which
				// is the most common mistake in our docs.
				if err != nil {
					break
				}
				if resp.StatusCode == http.StatusTooManyRequests {
					retryAfter := resp.Header.Get("Retry-After")
					if seconds, err := strconv.Atoi(retryAfter); err != nil {
						backoff = seconds + 10
					}
					fmt.Fprintf(os.Stderr, "Got %d visiting %s, retry after %d seconds.\n", resp.StatusCode, string(URL), backoff)
					time.Sleep(time.Duration(backoff) * time.Second)
					backoff *= 2
					retry++
				} else if resp.StatusCode == http.StatusNotFound {
					// We only check for 404 error for now. 401, 403 errors are hard to handle.

					// We need to try a GET to avoid false alert.
					resp, err = client.Get(string(processedURL))
					if err != nil {
						break
					}
					if resp.StatusCode != http.StatusNotFound {
						continue URL
					}

					foundInvalid = true
					fmt.Fprintf(os.Stderr, "Failed: in file %s, Got %d visiting %s\n", filePath, resp.StatusCode, string(URL))
					break
				} else {
					break
				}
			}
			if retry == maxRetry {
				foundInvalid = true
				fmt.Fprintf(os.Stderr, "Failed: in file %s, still got 429 visiting %s after %d retries\n", filePath, string(URL), maxRetry)
			}
		}
		if foundInvalid {
			*invalidLink = true
		}
		return nil
	}
}

func main() {
	flag.Parse()

	if *rootDir == "" {
		flag.Usage()
		os.Exit(2)
	}
	client := http.Client{
		Timeout: time.Duration(5 * time.Second),
	}
	invalidLink := false
	if err := filepath.Walk(*rootDir, newWalkFunc(&invalidLink, &client)); err != nil {
		fmt.Fprintf(os.Stderr, "Fail: %v.\n", err)
		os.Exit(2)
	}
	if invalidLink {
		os.Exit(1)
	}
}
