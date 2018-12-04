/*
Copyright 2018 The Kubernetes Authors.

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

package testfiles

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"

	"github.com/pkg/errors"
)

// URLFileSource retrieves content from a local cache and/or
// from remote. Both can be enabled separately.
//
// As a special case, files stored in a GitHub repo can
// be referenced with a URL that points to the HTML
// tree view (github.com/.../<repo>/blob/<revision>/<path>).
// URLFileSource will then retrieve the raw file, not the
// HTML page.
type URLFileSource struct {
	// Directory where cached files are stored.
	// Can be relative to the repo root, in which case
	// file content will also be retrieved from bindata
	// if that is enabled, or absolute, in which
	// case it will be read directly.
	//
	// URLs are turned into relative file names
	// by stripping the http[s]:// prefix and then
	// splitting at slashes.
	//
	// Caching is disabled if empty.
	CacheDir string

	// Enables retrieval via HTTP[S] as fallback
	// when the file is not found in the cache or
	// caching is disabled.
	Download bool
}

// ErrNotURL is returned as root cause by CachedFilePath when the testfile
// path is a normal file path.
var ErrNotURL = errors.New("must include scheme")

// ReadTestFile handles all paths that have a explicit scheme. If a cache
// was configured, it tries to get the content from there first before falling
// back to (again optionally) downloading the data.
func (u URLFileSource) ReadTestFile(filePath string) ([]byte, error) {
	fileURL, err := url.Parse(filePath)
	if err != nil {
		return nil, err
	}
	if fileURL.Scheme == "" {
		// Not a URL, defer to other file sources.
		return nil, nil
	}
	if u.CacheDir != "" {
		// Retrieve plain test file using whatever file source(s)
		// are registered.
		data, err := Read(u.createCachedFilePath(fileURL))
		if data != nil || err != nil {
			return data, err
		}
	}
	if u.Download {
		return u.download(filePath)
	}
	return nil, nil
}

func (u URLFileSource) createCachedFilePath(fileURL *url.URL) string {
	return path.Join(u.CacheDir, fileURL.Hostname(), fileURL.EscapedPath(), fileURL.RawQuery)
}

// First occurrence of "/blob/" is expected to separate repo from revision and file path.
var githubBlobRE = regexp.MustCompile(`^(.+?)/blob/([^/]+)/(.*)$`)

func (u URLFileSource) download(rawurl string) ([]byte, error) {
	// Special support for github.com: download raw file content even
	// when URL for HTML view is given.
	fileURL, err := url.Parse(rawurl)
	if err != nil {
		return nil, errors.Wrap(err, rawurl)
	}
	if fileURL.Host == "github.com" {
		parts := githubBlobRE.FindStringSubmatch(fileURL.EscapedPath())
		if parts != nil {
			rawurl = (&url.URL{
				Scheme: fileURL.Scheme,
				User:   fileURL.User,
				Host:   "raw.githubusercontent.com",
				Path:   path.Join(parts[1], parts[2], parts[3]),
			}).String()
		}
	}

	resp, err := http.Get(rawurl)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, errors.Errorf("%d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
	}
	var buffer bytes.Buffer
	if _, err = io.Copy(&buffer, resp.Body); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

// DescribeFiles summarizes from where the source would get content: cache and/or downloading.
func (u URLFileSource) DescribeFiles() string {
	var description string
	if u.CacheDir != "" && u.Download {
		description = fmt.Sprintf("Files referenced by URL are cached in %s and downloaded on demand.", u.CacheDir)
	} else if u.CacheDir != "" {
		description = fmt.Sprintf("Files referenced by URL are cached in %s.", u.CacheDir)
	} else if u.Download {
		description = "Files referenced by URL are downloaded on demand."
	}
	return description
}

// CachedFilePath returns the path where the the file is expected in the cache.
// In contrast to ReadTestFile, a valid URL with scheme must be passed.
func (u URLFileSource) CachedFilePath(rawurl string) (string, error) {
	fileURL, err := url.Parse(rawurl)
	if err != nil {
		return "", errors.Wrap(err, rawurl)
	}
	if fileURL.Scheme == "" {
		return "", errors.Wrap(ErrNotURL, rawurl)
	}
	return u.createCachedFilePath(fileURL), nil
}

// CacheURL downloads the file content and puts it into the cache.
func (u URLFileSource) CacheURL(rawurl string) error {
	data, err := u.download(rawurl)
	if err != nil {
		return err
	}
	if data == nil {
		return errors.New("no data")
	}
	filePath, err := u.CachedFilePath(rawurl)
	if err != nil {
		return err
	}
	dirPath := path.Dir(filePath)
	if err := os.MkdirAll(dirPath, 0755); err != nil && !os.IsExist(err) {
		return err
	}
	if err := ioutil.WriteFile(filePath, data, 0644); err != nil {
		return errors.Wrap(err, filePath)
	}
	return nil
}
