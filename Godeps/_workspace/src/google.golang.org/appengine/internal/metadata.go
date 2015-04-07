// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

// This file has code for accessing metadata.
//
// References:
//	https://cloud.google.com/compute/docs/metadata

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
)

const (
	metadataHost = "metadata"
	metadataPath = "/computeMetadata/v1/"
)

var (
	metadataRequestHeaders = http.Header{
		"X-Google-Metadata-Request": []string{"True"},
	}
)

// TODO(dsymonds): Do we need to support default values, like Python?
func mustGetMetadata(key string) []byte {
	b, err := getMetadata(key)
	if err != nil {
		log.Fatalf("Metadata fetch failed: %v", err)
	}
	return b
}

func getMetadata(key string) ([]byte, error) {
	// TODO(dsymonds): May need to use url.Parse to support keys with query args.
	req := &http.Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Host:   metadataHost,
			Path:   metadataPath + key,
		},
		Header: metadataRequestHeaders,
		Host:   metadataHost,
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("metadata server returned HTTP %d", resp.StatusCode)
	}
	return ioutil.ReadAll(resp.Body)
}
