// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compiler

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"sync"

	yaml "gopkg.in/yaml.v3"
)

var verboseReader = false

var fileCache map[string][]byte
var infoCache map[string]*yaml.Node

var fileCacheEnable = true
var infoCacheEnable = true

// These locks are used to synchronize accesses to the fileCache and infoCache
// maps (above). They are global state and can throw thread-related errors
// when modified from separate goroutines. The general strategy is to protect
// all public functions in this file with mutex Lock() calls. As a result, to
// avoid deadlock, these public functions should not call other public
// functions, so some public functions have private equivalents.
// In the future, we might consider replacing the maps with sync.Map and
// eliminating these mutexes.
var fileCacheMutex sync.Mutex
var infoCacheMutex sync.Mutex

func initializeFileCache() {
	if fileCache == nil {
		fileCache = make(map[string][]byte, 0)
	}
}

func initializeInfoCache() {
	if infoCache == nil {
		infoCache = make(map[string]*yaml.Node, 0)
	}
}

// EnableFileCache turns on file caching.
func EnableFileCache() {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	fileCacheEnable = true
}

// EnableInfoCache turns on parsed info caching.
func EnableInfoCache() {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	infoCacheEnable = true
}

// DisableFileCache turns off file caching.
func DisableFileCache() {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	fileCacheEnable = false
}

// DisableInfoCache turns off parsed info caching.
func DisableInfoCache() {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	infoCacheEnable = false
}

// RemoveFromFileCache removes an entry from the file cache.
func RemoveFromFileCache(fileurl string) {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	if !fileCacheEnable {
		return
	}
	initializeFileCache()
	delete(fileCache, fileurl)
}

// RemoveFromInfoCache removes an entry from the info cache.
func RemoveFromInfoCache(filename string) {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	if !infoCacheEnable {
		return
	}
	initializeInfoCache()
	delete(infoCache, filename)
}

// GetInfoCache returns the info cache map.
func GetInfoCache() map[string]*yaml.Node {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	if infoCache == nil {
		initializeInfoCache()
	}
	return infoCache
}

// ClearFileCache clears the file cache.
func ClearFileCache() {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	fileCache = make(map[string][]byte, 0)
}

// ClearInfoCache clears the info cache.
func ClearInfoCache() {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	infoCache = make(map[string]*yaml.Node)
}

// ClearCaches clears all caches.
func ClearCaches() {
	ClearFileCache()
	ClearInfoCache()
}

// FetchFile gets a specified file from the local filesystem or a remote location.
func FetchFile(fileurl string) ([]byte, error) {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	return fetchFile(fileurl)
}

func fetchFile(fileurl string) ([]byte, error) {
	var bytes []byte
	initializeFileCache()
	if fileCacheEnable {
		bytes, ok := fileCache[fileurl]
		if ok {
			if verboseReader {
				log.Printf("Cache hit %s", fileurl)
			}
			return bytes, nil
		}
		if verboseReader {
			log.Printf("Fetching %s", fileurl)
		}
	}
	response, err := http.Get(fileurl)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return nil, fmt.Errorf("Error downloading %s: %s", fileurl, response.Status)
	}
	bytes, err = ioutil.ReadAll(response.Body)
	if fileCacheEnable && err == nil {
		fileCache[fileurl] = bytes
	}
	return bytes, err
}

// ReadBytesForFile reads the bytes of a file.
func ReadBytesForFile(filename string) ([]byte, error) {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	return readBytesForFile(filename)
}

func readBytesForFile(filename string) ([]byte, error) {
	// is the filename a url?
	fileurl, _ := url.Parse(filename)
	if fileurl.Scheme != "" {
		// yes, fetch it
		bytes, err := fetchFile(filename)
		if err != nil {
			return nil, err
		}
		return bytes, nil
	}
	// no, it's a local filename
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return bytes, nil
}

// ReadInfoFromBytes unmarshals a file as a *yaml.Node.
func ReadInfoFromBytes(filename string, bytes []byte) (*yaml.Node, error) {
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	return readInfoFromBytes(filename, bytes)
}

func readInfoFromBytes(filename string, bytes []byte) (*yaml.Node, error) {
	initializeInfoCache()
	if infoCacheEnable {
		cachedInfo, ok := infoCache[filename]
		if ok {
			if verboseReader {
				log.Printf("Cache hit info for file %s", filename)
			}
			return cachedInfo, nil
		}
		if verboseReader {
			log.Printf("Reading info for file %s", filename)
		}
	}
	var info yaml.Node
	err := yaml.Unmarshal(bytes, &info)
	if err != nil {
		return nil, err
	}
	if infoCacheEnable && len(filename) > 0 {
		infoCache[filename] = &info
	}
	return &info, nil
}

// ReadInfoForRef reads a file and return the fragment needed to resolve a $ref.
func ReadInfoForRef(basefile string, ref string) (*yaml.Node, error) {
	fileCacheMutex.Lock()
	defer fileCacheMutex.Unlock()
	infoCacheMutex.Lock()
	defer infoCacheMutex.Unlock()
	initializeInfoCache()
	if infoCacheEnable {
		info, ok := infoCache[ref]
		if ok {
			if verboseReader {
				log.Printf("Cache hit for ref %s#%s", basefile, ref)
			}
			return info, nil
		}
		if verboseReader {
			log.Printf("Reading info for ref %s#%s", basefile, ref)
		}
	}
	basedir, _ := filepath.Split(basefile)
	parts := strings.Split(ref, "#")
	var filename string
	if parts[0] != "" {
		filename = parts[0]
		if _, err := url.ParseRequestURI(parts[0]); err != nil {
			// It is not an URL, so the file is local
			filename = basedir + parts[0]
		}
	} else {
		filename = basefile
	}
	bytes, err := readBytesForFile(filename)
	if err != nil {
		return nil, err
	}
	info, err := readInfoFromBytes(filename, bytes)
	if info != nil && info.Kind == yaml.DocumentNode {
		info = info.Content[0]
	}
	if err != nil {
		log.Printf("File error: %v\n", err)
	} else {
		if info == nil {
			return nil, NewError(nil, fmt.Sprintf("could not resolve %s", ref))
		}
		if len(parts) > 1 {
			path := strings.Split(parts[1], "/")
			for i, key := range path {
				if i > 0 {
					m := info
					if true {
						found := false
						for i := 0; i < len(m.Content); i += 2 {
							if m.Content[i].Value == key {
								info = m.Content[i+1]
								found = true
							}
						}
						if !found {
							infoCache[ref] = nil
							return nil, NewError(nil, fmt.Sprintf("could not resolve %s", ref))
						}
					}
				}
			}
		}
	}
	if infoCacheEnable {
		infoCache[ref] = info
	}
	return info, nil
}
