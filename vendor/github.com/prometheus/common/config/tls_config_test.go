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

package config

import (
	"crypto/tls"
	"io/ioutil"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// LoadTLSConfig parses the given YAML file into a tls.Config.
func LoadTLSConfig(filename string) (*tls.Config, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	cfg := &TLSConfig{}
	if err = yaml.Unmarshal(content, cfg); err != nil {
		return nil, err
	}
	return cfg.GenerateConfig()
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
		cfg.config.BuildNameToCertificate()
		got, err := LoadTLSConfig("testdata/" + cfg.filename)
		if err != nil {
			t.Errorf("Error parsing %s: %s", cfg.filename, err)
		}
		if !reflect.DeepEqual(*got, *cfg.config) {
			t.Fatalf("%s: unexpected config result: \n\n%s\n expected\n\n%s", cfg.filename, got, cfg.config)
		}
	}
}

var expectedTLSConfigErrors = []struct {
	filename string
	errMsg   string
}{
	{
		filename: "tls_config.invalid_field.bad.yml",
		errMsg:   "unknown fields in",
	}, {
		filename: "tls_config.cert_no_key.bad.yml",
		errMsg:   "specified without client key file",
	}, {
		filename: "tls_config.key_no_cert.bad.yml",
		errMsg:   "specified without client cert file",
	},
}

func TestBadTLSConfigs(t *testing.T) {
	for _, ee := range expectedTLSConfigErrors {
		_, err := LoadTLSConfig("testdata/" + ee.filename)
		if err == nil {
			t.Errorf("Expected error parsing %s but got none", ee.filename)
			continue
		}
		if !strings.Contains(err.Error(), ee.errMsg) {
			t.Errorf("Expected error for %s to contain %q but got: %s", ee.filename, ee.errMsg, err)
		}
	}
}
