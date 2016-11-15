// Copyright 2015 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package libcni

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
)

func ConfFromBytes(bytes []byte) (*NetworkConfig, error) {
	conf := &NetworkConfig{Bytes: bytes}
	if err := json.Unmarshal(bytes, &conf.Network); err != nil {
		return nil, fmt.Errorf("error parsing configuration: %s", err)
	}
	return conf, nil
}

func ConfFromFile(filename string) (*NetworkConfig, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %s", filename, err)
	}
	return ConfFromBytes(bytes)
}

func ConfFiles(dir string) ([]string, error) {
	// In part, adapted from rkt/networking/podenv.go#listFiles
	files, err := ioutil.ReadDir(dir)
	switch {
	case err == nil: // break
	case os.IsNotExist(err):
		return nil, nil
	default:
		return nil, err
	}

	confFiles := []string{}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if filepath.Ext(f.Name()) == ".conf" {
			confFiles = append(confFiles, filepath.Join(dir, f.Name()))
		}
	}
	return confFiles, nil
}

func LoadConf(dir, name string) (*NetworkConfig, error) {
	files, err := ConfFiles(dir)
	switch {
	case err != nil:
		return nil, err
	case len(files) == 0:
		return nil, fmt.Errorf("no net configurations found")
	}
	sort.Strings(files)

	for _, confFile := range files {
		conf, err := ConfFromFile(confFile)
		if err != nil {
			return nil, err
		}
		if conf.Network.Name == name {
			return conf, nil
		}
	}
	return nil, fmt.Errorf(`no net configuration with name "%s" in %s`, name, dir)
}

func InjectConf(original *NetworkConfig, key string, newValue interface{}) (*NetworkConfig, error) {
	config := make(map[string]interface{})
	err := json.Unmarshal(original.Bytes, &config)
	if err != nil {
		return nil, fmt.Errorf("unmarshal existing network bytes: %s", err)
	}

	if key == "" {
		return nil, fmt.Errorf("key value can not be empty")
	}

	if newValue == nil {
		return nil, fmt.Errorf("newValue must be specified")
	}

	config[key] = newValue

	newBytes, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}

	return ConfFromBytes(newBytes)
}
