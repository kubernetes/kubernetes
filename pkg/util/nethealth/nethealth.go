/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// The nethealth binary performs a quick HTTP GET download speed check
// Key Features:
//   Shell-script friendly - returns a non-zero exit code on timeout or corruption or bandwidth threshold failures
//   Timeout configurable to abort the test early (for super slow links)
//   Can compare actual bandwidth against a command line minimum bandwidth parameter and return a non-zero exit code.
//   Corruption check - can download a checksum file and compute blob checksum and compare.
//   Configurable object URL for non-GCE environments

package main

import (
	"crypto/sha512"
	"encoding/hex"
	"flag"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"
)

var (
	objectUrl     string
	objectHashUrl string
	objectLength  int64
	timeout       int
	minimum       int64
)

func init() {
	// Defaults to a public bucket with a 64 MB file with random data
	flag.StringVar(&objectUrl, "url", "http://storage.googleapis.com/k8s-bandwidth-test/64MB.bin", "Blob URL")
	flag.StringVar(&objectHashUrl, "hashurl", "http://storage.googleapis.com/k8s-bandwidth-test/sha512.txt", "Blob Hash URL")
	flag.Int64Var(&objectLength, "length", 64*1024*1024, "Expected content length")
	flag.IntVar(&timeout, "timeout", 30, "Maximum Seconds to wait")
	// If the transfer bandwidth is lower than the minimum, process returns non-zero exit status
	flag.Int64Var(&minimum, "minimum", 10, "Minimum bandwidth expected (MiB/sec)")
}

func monitorTimeElapsed(downloadStartTime time.Time, timeoutSeconds time.Duration) {
	for true {
		time.Sleep(1 * time.Second)
		// Check the status of the Get request if possible and check if timeout elapsed
		if time.Since(downloadStartTime) > timeoutSeconds*time.Second {
			log.Fatalf("ERROR: Timeout (%d) seconds occurred before GET finished - declaring TOO SLOW", timeout)
		}
	}
}

func main() {
	flag.Parse()
	// Quick object existence check before a full GET
	res, err := http.Head(objectUrl)
	if err != nil {
		log.Fatalf("Failed to find URL %s (%s)", objectUrl, err)
	}
	if res.ContentLength != objectLength {
		log.Fatalf("Length reported (%d) is not equal to expected length (%d)", res.ContentLength, objectLength)
	}
	log.Printf("HTTP HEAD reports content length: %d - running GET\n", res.ContentLength)
	res, err = http.Head(objectHashUrl)
	if err != nil {
		log.Fatalf("Failed to find hash URL %s (%s)", objectHashUrl, err)
	}
	/* Now, setup a Client with a transport with compression disabled and timeouts enabled */
	tr := &http.Transport{
		DisableCompression: true,
	}
	downloadStartTime := time.Now()
	go monitorTimeElapsed(downloadStartTime, time.Duration(timeout))
	client := &http.Client{Transport: tr}
	res, err = client.Get(objectUrl)
	if err != nil {
		log.Fatalf("Failure (%s) while reading %s", err, objectUrl)
	}
	if res.ContentLength != objectLength {
		log.Fatalf("Length reported (%d) is not equal to expected length (%d)", res.ContentLength, objectLength)
	}
	blobData, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		log.Fatal("Failed to read full content", err)
	}
	elapsedMs := int64(time.Since(downloadStartTime) / time.Millisecond)
	bandwidth := (res.ContentLength * 1000) / (elapsedMs * 1024)

	log.Printf("DOWNLOAD: %d bytes %d ms Bandwidth ~ %d KiB/sec\n", res.ContentLength, elapsedMs, bandwidth)
	// Check if this bandwidth exceeds minimum expected
	if minimum*1024 > bandwidth {
		log.Fatalf("ERROR: Minimum bandwidth guarantee of %d MiB/sec not met - network connectivity is slow", minimum)
	}
	// Perform SHA512 hash and compare against the expected hash to check for corruption.
	res, err = client.Get(objectHashUrl)
	if err != nil {
		log.Fatalf("Failure (%s) while reading %s", err, objectHashUrl)
	}
	content, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		log.Fatal("Failed to read full content of hash file", err)
	}
	parts := strings.Split(string(content), " ")
	if len(parts) <= 1 {
		log.Fatalf("Could not parse SHA hash file contents (%s)", content)
	}
	hash := parts[1]
	hash = strings.Trim(hash, "\n ")
	sumBytes := sha512.Sum512(blobData)
	sumString := hex.EncodeToString(sumBytes[0:])
	if strings.Compare(sumString, hash) == 0 {
		log.Println("Hash Matches expected value")
	} else {
		log.Fatalf("ERROR: Hash Mismatch - Computed hash = '%s' Expected hash = '%s'", sumString, hash)
	}
}
