// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.8

package config

import (
	"crypto/tls"
	"io/ioutil"
	"reflect"
	"testing"

	"gopkg.in/yaml.v2"
)

// LoadTLSConfig parses the given YAML file into a tls.Config.
func LoadTLSConfig(filename string) (*tls.Config, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	cfg := TLSConfig{}
	if err = yaml.UnmarshalStrict(content, &cfg); err != nil {
		return nil, err
	}
	return NewTLSConfig(&cfg)
}

var expectedTLSConfigs = []struct {
	filename string
	config   *tls.Config
}{
	{
		filename: "tls_config.empty.good.yml",
		config:   &tls.Config{},
	}, {
		filename: "tls_config.insecure.good.yml",
		config:   &tls.Config{InsecureSkipVerify: true},
	},
}

func TestValidTLSConfig(t *testing.T) {
	for _, cfg := range expectedTLSConfigs {
		got, err := LoadTLSConfig("testdata/" + cfg.filename)
		if err != nil {
			t.Errorf("Error parsing %s: %s", cfg.filename, err)
		}
		// non-nil functions are never equal.
		got.GetClientCertificate = nil
		if !reflect.DeepEqual(got, cfg.config) {
			t.Fatalf("%v: unexpected config result: \n\n%v\n expected\n\n%v", cfg.filename, got, cfg.config)
		}
	}
}
