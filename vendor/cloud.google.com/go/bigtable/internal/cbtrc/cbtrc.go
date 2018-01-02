/*
Copyright 2015 Google Inc. All Rights Reserved.

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

// Package cbtrc encapsulates common code for reading .cbtrc files.
package cbtrc

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// Config represents a configuration.
type Config struct {
	Project, Instance string // required
	Creds             string // optional
}

// RegisterFlags registers a set of standard flags for this config.
// It should be called before flag.Parse.
func (c *Config) RegisterFlags() {
	flag.StringVar(&c.Project, "project", c.Project, "project ID")
	flag.StringVar(&c.Instance, "instance", c.Instance, "Cloud Bigtable instance")
	flag.StringVar(&c.Creds, "creds", c.Creds, "if set, use application credentials in this file")
}

// CheckFlags checks that the required config values are set.
func (c *Config) CheckFlags() error {
	var missing []string
	if c.Project == "" {
		missing = append(missing, "-project")
	}
	if c.Instance == "" {
		missing = append(missing, "-instance")
	}
	if len(missing) > 0 {
		return fmt.Errorf("Missing %s", strings.Join(missing, " and "))
	}
	return nil
}

// Filename returns the filename consulted for standard configuration.
func Filename() string {
	// TODO(dsymonds): Might need tweaking for Windows.
	return filepath.Join(os.Getenv("HOME"), ".cbtrc")
}

// Load loads a .cbtrc file.
// If the file is not present, an empty config is returned.
func Load() (*Config, error) {
	filename := Filename()
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		// silent fail if the file isn't there
		if os.IsNotExist(err) {
			return &Config{}, nil
		}
		return nil, fmt.Errorf("Reading %s: %v", filename, err)
	}
	c := new(Config)
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		line := s.Text()
		i := strings.Index(line, "=")
		if i < 0 {
			return nil, fmt.Errorf("Bad line in %s: %q", filename, line)
		}
		key, val := strings.TrimSpace(line[:i]), strings.TrimSpace(line[i+1:])
		switch key {
		default:
			return nil, fmt.Errorf("Unknown key in %s: %q", filename, key)
		case "project":
			c.Project = val
		case "instance":
			c.Instance = val
		case "creds":
			c.Creds = val
		}
	}
	return c, s.Err()
}
