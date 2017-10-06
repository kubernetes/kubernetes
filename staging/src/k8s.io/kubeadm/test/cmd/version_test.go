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

package kubeadm

import (
	"encoding/json"
	"regexp"
	"testing"

	"github.com/ghodss/yaml"
)

const (
	ShortExpectedRegex  = "^v.+\n$"
	NormalExpectedRegex = "^kubeadm version: &version\\.Info{Major:\".+\", Minor:\".+\", GitVersion:\".+\", GitCommit:\".+\", GitTreeState:\".+\", BuildDate:\".+\", GoVersion:\".+\", Compiler:\".+\", Platform:\".+\"}\n$"
)

var (
	VersionInfo = []string{"major", "minor", "gitVersion", "gitCommit", "gitTreeState", "buildDate", "goVersion", "compiler", "platform"}
)

func TestCmdVersion(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var versionTest = []struct {
		args     string
		regex    string
		expected bool
	}{
		{"--output=valid", "", false},
		{"--output=short", ShortExpectedRegex, true},
		{"", NormalExpectedRegex, true},
	}

	for _, rt := range versionTest {
		stdout, _, actual := RunCmd(*kubeadmPath, "version", rt.args)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdVersion running 'kubeadm version %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}

		if rt.expected {
			matched, err := regexp.MatchString(rt.regex, stdout)
			if err != nil {
				t.Fatalf("encountered an error while trying to match 'kubeadm version %s' stdout: %v", rt.args, err)
			}
			if !matched {
				t.Errorf("'kubeadm version %s' stdout did not match expected regex; wanted: [%q], got: [%s]", rt.args, rt.regex, stdout)
			}
		}
	}
}

func TestCmdVersionOutputJsonOrYaml(t *testing.T) {
	if *kubeadmCmdSkip {
		t.Log("kubeadm cmd tests being skipped")
		t.Skip()
	}

	var versionTest = []struct {
		args     string
		format   string
		expected bool
	}{
		{"--output=json", "json", true},
		{"--output=yaml", "yaml", true},
	}

	for _, rt := range versionTest {
		stdout, _, actual := RunCmd(*kubeadmPath, "version", rt.args)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CmdVersion running 'kubeadm version %s' with an error: %v\n\texpected: %t\n\t  actual: %t",
				rt.args,
				actual,
				rt.expected,
				(actual == nil),
			)
		}

		if rt.expected {
			var obj interface{}
			switch rt.format {
			case "json":
				err := json.Unmarshal([]byte(stdout), &obj)
				if err != nil {
					t.Errorf("failed to parse json from 'kubeadm version %s': %s", rt.args, err)
				}
			case "yaml":
				err := yaml.Unmarshal([]byte(stdout), &obj)
				if err != nil {
					t.Errorf("failed to parse yaml from 'kubeadm version %s': %s", rt.args, err)
				}
			}

			m := obj.(map[string]interface{})
			if m["clientVersion"] == nil {
				t.Errorf("failed to get the infomation of clientVersion from 'kubeadm version %s'", rt.args)
			}
			info := m["clientVersion"].(map[string]interface{})
			for _, key := range VersionInfo {
				if len(info[key].(string)) == 0 {
					t.Errorf("failed to get the infomation of %s from 'kubeadm version %s'", key, rt.args)
				}
			}
		}
	}
}
