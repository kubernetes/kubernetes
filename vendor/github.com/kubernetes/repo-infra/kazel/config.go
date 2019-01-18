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
	"encoding/json"
	"io/ioutil"
)

// Cfg defines the configuration options for kazel.
type Cfg struct {
	GoPrefix string
	// evaluated recursively, defaults to ["."]
	SrcDirs []string
	// regexps that match packages to skip
	SkippedPaths []string
	// regexps that match packages to skip for K8SOpenAPIGen.
	// note that this skips anything matched by SkippedPaths as well.
	SkippedOpenAPIGenPaths []string
	// whether to add "pkg-srcs" and "all-srcs" filegroups
	// note that this operates on the entire tree (not just SrcsDirs) but skips anything matching SkippedPaths
	AddSourcesRules bool
	// whether to have multiple build files in vendor/ or just one.
	VendorMultipleBuildFiles bool
	// whether to manage kubernetes' pkg/generated/openapi.
	K8sOpenAPIGen bool
	// Whether to manage the upstream Go rules provided by bazelbuild/rules_go.
	// If using gazelle, set this to false (or omit).
	ManageGoRules bool
}

// ReadCfg reads and unmarshals the specified json file into a Cfg struct.
func ReadCfg(cfgPath string) (*Cfg, error) {
	b, err := ioutil.ReadFile(cfgPath)
	if err != nil {
		return nil, err
	}
	var cfg Cfg
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	defaultCfg(&cfg)
	return &cfg, nil
}

func defaultCfg(c *Cfg) {
	if len(c.SrcDirs) == 0 {
		c.SrcDirs = []string{"."}
	}
}
