// Copyright 2017 Google Inc. All Rights Reserved.
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
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
)

var file_cache map[string][]byte
var info_cache map[string]interface{}
var count int64

var VERBOSE_READER = false

func initializeFileCache() {
	if file_cache == nil {
		file_cache = make(map[string][]byte, 0)
	}
}

func initializeInfoCache() {
	if info_cache == nil {
		info_cache = make(map[string]interface{}, 0)
	}
}

func FetchFile(fileurl string) ([]byte, error) {
	initializeFileCache()
	bytes, ok := file_cache[fileurl]
	if ok {
		if VERBOSE_READER {
			log.Printf("Cache hit %s", fileurl)
		}
		return bytes, nil
	}
	log.Printf("Fetching %s", fileurl)
	response, err := http.Get(fileurl)
	if err != nil {
		return nil, err
	} else {
		defer response.Body.Close()
		bytes, err := ioutil.ReadAll(response.Body)
		if err == nil {
			file_cache[fileurl] = bytes
		}
		return bytes, err
	}
}

// read a file and unmarshal it as a yaml.MapSlice
func ReadInfoForFile(filename string) (interface{}, error) {
	initializeInfoCache()
	info, ok := info_cache[filename]
	if ok {
		if VERBOSE_READER {
			log.Printf("Cache hit info for file %s", filename)
		}
		return info, nil
	}
	if VERBOSE_READER {
		log.Printf("Reading info for file %s", filename)
	}

	// is the filename a url?
	fileurl, _ := url.Parse(filename)
	if fileurl.Scheme != "" {
		// yes, fetch it
		bytes, err := FetchFile(filename)
		if err != nil {
			return nil, err
		}
		var info yaml.MapSlice
		err = yaml.Unmarshal(bytes, &info)
		if err != nil {
			return nil, err
		}
		info_cache[filename] = info
		return info, nil
	} else {
		// no, it's a local filename
		bytes, err := ioutil.ReadFile(filename)
		if err != nil {
			log.Printf("File error: %v\n", err)
			return nil, err
		}
		var info yaml.MapSlice
		err = yaml.Unmarshal(bytes, &info)
		if err != nil {
			return nil, err
		}
		info_cache[filename] = info
		return info, nil
	}
}

// read a file and return the fragment needed to resolve a $ref
func ReadInfoForRef(basefile string, ref string) (interface{}, error) {
	initializeInfoCache()
	{
		info, ok := info_cache[ref]
		if ok {
			if VERBOSE_READER {
				log.Printf("Cache hit for ref %s#%s", basefile, ref)
			}
			return info, nil
		}
	}
	if VERBOSE_READER {
		log.Printf("Reading info for ref %s#%s", basefile, ref)
	}
	count = count + 1
	basedir, _ := filepath.Split(basefile)
	parts := strings.Split(ref, "#")
	var filename string
	if parts[0] != "" {
		filename = basedir + parts[0]
	} else {
		filename = basefile
	}
	info, err := ReadInfoForFile(filename)
	if err != nil {
		log.Printf("File error: %v\n", err)
	} else {
		if len(parts) > 1 {
			path := strings.Split(parts[1], "/")
			for i, key := range path {
				if i > 0 {
					m, ok := info.(yaml.MapSlice)
					if ok {
						found := false
						for _, section := range m {
							if section.Key == key {
								info = section.Value
								found = true
							}
						}
						if !found {
							info_cache[ref] = nil
							return nil, NewError(nil, fmt.Sprintf("could not resolve %s", ref))
						}
					}
				}
			}
		}
	}
	info_cache[ref] = info
	return info, nil
}
