/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"regexp"
)

var (
	successEndpoints = []string{
		// Discovery
		"http://169.254.169.254",
		"http://metadata.google.internal",
		"http://169.254.169.254/",
		"http://metadata.google.internal/",
		"http://metadata.google.internal/0.1",
		"http://metadata.google.internal/0.1/",
		"http://metadata.google.internal/0.1/meta-data",
		"http://metadata.google.internal/computeMetadata",
		"http://metadata.google.internal/computeMetadata/v1beta1",
		"http://metadata.google.internal/computeMetadata/v1",
		// Allowed API versions.
		"http://metadata.google.internal/0.1/meta-data/",
		"http://metadata.google.internal/computeMetadata/v1beta1/",
		"http://metadata.google.internal/computeMetadata/v1/",
		// Service account token endpoints.
		"http://metadata.google.internal/0.1/meta-data/service-accounts/default/acquire",
		"http://metadata.google.internal/computeMetadata/v1beta1/instance/service-accounts/default/token",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
		// Params that contain 'recursive' as substring.
		"http://metadata.google.internal/computeMetadata/v1/instance/?nonrecursive=true",
		"http://metadata.google.internal/computeMetadata/v1/instance/?something=other&nonrecursive=true",
	}
	noKubeEnvEndpoints = []string{
		// Check that these don't get a recursive result.
		"http://metadata.google.internal/computeMetadata/v1/instance/?recursive%3Dtrue",   // urlencoded
		"http://metadata.google.internal/computeMetadata/v1/instance/?re%08ecursive=true", // backspaced
	}
	failureEndpoints = []string{
		// Other API versions.
		"http://metadata.google.internal/0.2/",
		"http://metadata.google.internal/computeMetadata/v2/",
		// kube-env.
		"http://metadata.google.internal/0.1/meta-data/attributes/kube-env",
		"http://metadata.google.internal/computeMetadata/v1beta1/instance/attributes/kube-env",
		"http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env",
		// VM identity.
		"http://metadata.google.internal/0.1/meta-data/service-accounts/default/identity",
		"http://metadata.google.internal/computeMetadata/v1beta1/instance/service-accounts/default/identity",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
		// Recursive.
		"http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true",
		"http://metadata.google.internal/computeMetadata/v1/instance/?something=other&recursive=true",
		"http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true&something=other",
		// Other.
		"http://metadata.google.internal/computeMetadata/v1/instance/attributes//kube-env",
		"http://metadata.google.internal/computeMetadata/v1/instance/attributes/../attributes/kube-env",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts//default/identity",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/../service-accounts/default/identity",
	}
)

func main() {
	success := 0
	h := map[string][]string{
		"Metadata-Flavor": {"Google"},
	}
	for _, e := range successEndpoints {
		if err := checkURL(e, h, 200, ""); err != nil {
			log.Printf("Wrong response for %v: %v", e, err)
			success = 1
		}
	}
	for _, e := range noKubeEnvEndpoints {
		if err := checkURL(e, h, 200, "kube-env"); err != nil {
			log.Printf("Wrong response for %v: %v", e, err)
			success = 1
		}
	}
	for _, e := range failureEndpoints {
		if err := checkURL(e, h, 403, ""); err != nil {
			log.Printf("Wrong response for %v: %v", e, err)
			success = 1
		}
	}

	xForwardedForHeader := map[string][]string{
		"X-Forwarded-For": {"Somebody-somewhere"},
	}
	// Check that success endpoints fail if X-Forwarded-For is present.
	for _, e := range successEndpoints {
		if err := checkURL(e, xForwardedForHeader, 403, ""); err != nil {
			log.Printf("Wrong response for %v with X-Forwarded-For: %v", e, err)
			success = 1
		}
	}
	os.Exit(success)
}

// Checks that a URL with the given headers returns the right code, and if s is
// non-empty, checks that the body doesn't contain s.
func checkURL(url string, header http.Header, expectedStatus int, s string) error {
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header = header
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != expectedStatus {
		return fmt.Errorf("unexpected response: got %d, want %d", resp.StatusCode, expectedStatus)
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if s != "" {
		matched, err := regexp.Match(s, body)
		if err != nil {
			return err
		}
		if matched {
			return fmt.Errorf("body incorrectly contained %q: got %v", s, string(body))
		}
	}
	return nil
}
