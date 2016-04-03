// Copyright 2016 The rkt Authors
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

package config

import (
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"sort"
	"strings"
)

type stage1V1JsonParser struct{}

type stage1V1 struct {
	Name     string `json:"name"`
	Version  string `json:"version"`
	Location string `json:"location"`
}

var (
	allowedSchemes = map[string]struct{}{
		"file":   struct{}{},
		"docker": struct{}{},
		"http":   struct{}{},
		"https":  struct{}{},
	}
)

func init() {
	addParser("stage1", "v1", &stage1V1JsonParser{})
	registerSubDir("stage1.d", []string{"stage1"})
}

func (p *stage1V1JsonParser) parse(config *Config, raw []byte) error {
	var stage1 stage1V1
	if err := json.Unmarshal(raw, &stage1); err != nil {
		return err
	}
	if err := p.validateStage1V1(&stage1); err != nil {
		return fmt.Errorf("invalid stage1 configuration: %v", err)
	}
	// At this point either both name and version are specified or
	// neither. The same goes for data in Config.
	if stage1.Name != "" {
		if config.Stage1.Name != "" {
			return fmt.Errorf("name and version of a default stage1 image are already specified")
		}
		config.Stage1.Name = stage1.Name
		config.Stage1.Version = stage1.Version
	}
	if stage1.Location != "" {
		if config.Stage1.Location != "" {
			return fmt.Errorf("location of a default stage1 image is already specified")
		}
		config.Stage1.Location = stage1.Location
	}
	return nil
}

func (p *stage1V1JsonParser) validateStage1V1(stage1 *stage1V1) error {
	if stage1.Name == "" && stage1.Version != "" {
		return fmt.Errorf("default stage1 image version specified, but name is missing")
	}
	if stage1.Name != "" && stage1.Version == "" {
		return fmt.Errorf("default stage1 image name specified, but version is missing")
	}
	if stage1.Location != "" {
		if !filepath.IsAbs(stage1.Location) {
			url, err := url.Parse(stage1.Location)
			if err != nil {
				return fmt.Errorf("default stage1 image location is an invalid URL: %v", err)
			}
			if url.Scheme == "" {
				return fmt.Errorf("default stage1 image location is either a relative path or a URL without scheme")
			}
			if _, ok := allowedSchemes[url.Scheme]; !ok {
				schemes := toArray(allowedSchemes)
				sort.Strings(schemes)
				return fmt.Errorf("default stage1 image location URL has invalid scheme %q, allowed schemes are %q", url.Scheme, strings.Join(schemes, `", "`))
			}
		}
	}
	return nil
}
