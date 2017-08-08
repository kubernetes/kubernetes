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

package openapi

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"

	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/pkg/version"
)

const openapiFileName = "openapi_cache"

type CachingOpenAPIClient struct {
	version      string
	client       discovery.OpenAPISchemaInterface
	cacheDirName string
}

// NewCachingOpenAPIClient returns a new discovery.OpenAPISchemaInterface
// that will read the openapi spec from a local cache if it exists, and
// if not will then fetch an openapi spec using a client.
// client: used to fetch a new openapi spec if a local cache is not found
// version: the server version and used as part of the cache file location
// cacheDir: the directory under which the cache file will be written
func NewCachingOpenAPIClient(client discovery.OpenAPISchemaInterface, version, cacheDir string) *CachingOpenAPIClient {
	return &CachingOpenAPIClient{
		client:       client,
		version:      version,
		cacheDirName: cacheDir,
	}
}

// OpenAPIData returns an openapi spec.
// It will first attempt to read the spec from a local cache
// If it cannot read a local cache, it will read the file
// using the client and then write the cache.
func (c *CachingOpenAPIClient) OpenAPIData() (Resources, error) {
	// Try to use the cached version
	if c.useCache() {
		doc, err := c.readOpenAPICache()
		if err == nil {
			return NewOpenAPIData(doc)
		}
	}

	// No cached version found, download from server
	s, err := c.client.OpenAPISchema()
	if err != nil {
		glog.V(2).Infof("Failed to download openapi data %v", err)
		return nil, err
	}

	oa, err := NewOpenAPIData(s)
	if err != nil {
		glog.V(2).Infof("Failed to parse openapi data %v", err)
		return nil, err
	}

	// Try to cache the openapi spec
	if c.useCache() {
		err = c.writeToCache(s)
		if err != nil {
			// Just log an message, no need to fail the command since we got the data we need
			glog.V(2).Infof("Unable to cache openapi spec %v", err)
		}
	}

	// Return the parsed data
	return oa, nil
}

// useCache returns true if the client should try to use the cache file
func (c *CachingOpenAPIClient) useCache() bool {
	return len(c.version) > 0 && len(c.cacheDirName) > 0
}

// readOpenAPICache tries to read the openapi spec from the local file cache
func (c *CachingOpenAPIClient) readOpenAPICache() (*openapi_v2.Document, error) {
	// Get the filename to read
	filename := c.openAPICacheFilename()

	// Read the cached file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	doc := &openapi_v2.Document{}
	return doc, proto.Unmarshal(data, doc)
}

// writeToCache tries to write the openapi spec to the local file cache.
// writes the data to a new tempfile, and then links the cache file and the tempfile
func (c *CachingOpenAPIClient) writeToCache(doc *openapi_v2.Document) error {
	// Get the constant filename used to read the cache.
	cacheFile := c.openAPICacheFilename()

	// Binary encode the spec.  This is 10x as fast as using json encoding.  (60ms vs 600ms)
	b, err := proto.Marshal(doc)
	if err != nil {
		return fmt.Errorf("Could not binary encode openapi spec: %v", err)
	}

	// Create a new temp file for the cached openapi spec.
	cacheDir := filepath.Dir(cacheFile)
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return fmt.Errorf("Could not create directory: %v %v", cacheDir, err)
	}
	tmpFile, err := ioutil.TempFile(cacheDir, "openapi")
	if err != nil {
		return fmt.Errorf("Could not create temp cache file: %v %v", cacheFile, err)
	}

	// Write the binary encoded openapi spec to the temp file
	if _, err := io.Copy(tmpFile, bytes.NewBuffer(b)); err != nil {
		return fmt.Errorf("Could not write temp cache file: %v", err)
	}

	// Link the temp cache file to the constant cache filepath
	return linkFiles(tmpFile.Name(), cacheFile)
}

// openAPICacheFilename returns the filename to read the cache from
func (c *CachingOpenAPIClient) openAPICacheFilename() string {
	// Cache using the client and server versions
	return filepath.Join(c.cacheDirName, c.version, version.Get().GitVersion, openapiFileName)
}

// linkFiles links the old file to the new file
func linkFiles(old, new string) error {
	if err := os.Link(old, new); err != nil {
		// If we can't write due to file existing, or permission problems, keep going.
		if os.IsExist(err) || os.IsPermission(err) {
			return nil
		}
		return err
	}
	return nil
}
