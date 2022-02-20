/*
Copyright 2021 The Kubernetes Authors.

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

package load

import (
	"bytes"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/pod-security-admission/admission/api"
)

var defaultConfig = &api.PodSecurityConfiguration{
	Defaults: api.PodSecurityDefaults{
		Enforce: "privileged", EnforceVersion: "latest",
		Warn: "privileged", WarnVersion: "latest",
		Audit: "privileged", AuditVersion: "latest",
	},
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	file, err := ioutil.TempFile("", "podsecurityconfig")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		os.Remove(file.Name())
	})
	if err := ioutil.WriteFile(file.Name(), []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return file.Name()
}

func TestLoadFromFile(t *testing.T) {
	// no file
	{
		config, err := LoadFromFile("")
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, defaultConfig) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(defaultConfig, config))
		}
	}

	// empty file
	{
		config, err := LoadFromFile(writeTempFile(t, ``))
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, defaultConfig) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(defaultConfig, config))
		}
	}

	// valid file
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`
		expect := &api.PodSecurityConfiguration{
			Defaults: api.PodSecurityDefaults{
				Enforce: "baseline", EnforceVersion: "latest",
				Warn: "privileged", WarnVersion: "latest",
				Audit: "privileged", AuditVersion: "latest",
			},
		}

		config, err := LoadFromFile(writeTempFile(t, input))
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, expect) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(expect, config))
		}
	}

	// valid file
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v1beta1",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`
		expect := &api.PodSecurityConfiguration{
			Defaults: api.PodSecurityDefaults{
				Enforce: "baseline", EnforceVersion: "latest",
				Warn: "privileged", WarnVersion: "latest",
				Audit: "privileged", AuditVersion: "latest",
			},
		}

		config, err := LoadFromFile(writeTempFile(t, input))
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, expect) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(expect, config))
		}
	}

	// missing file
	{
		_, err := LoadFromFile(`bogus-missing-pod-security-policy-config-file`)
		if err == nil {
			t.Fatalf("expected err, got none")
		}
		if !strings.Contains(err.Error(), "bogus-missing-pod-security-policy-config-file") {
			t.Fatalf("expected missing file error, got %v", err)
		}
	}

	// invalid content file
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v99",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`

		_, err := LoadFromFile(writeTempFile(t, input))
		if err == nil {
			t.Fatalf("expected err, got none")
		}
		if !strings.Contains(err.Error(), "pod-security.admission.config.k8s.io/v99") {
			t.Fatalf("expected apiVersion error, got %v", err)
		}
	}
}

func TestLoadFromReader(t *testing.T) {
	// no reader
	{
		config, err := LoadFromReader(nil)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, defaultConfig) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(defaultConfig, config))
		}
	}

	// empty reader
	{
		config, err := LoadFromReader(&bytes.Buffer{})
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, defaultConfig) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(defaultConfig, config))
		}
	}

	// valid reader
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`
		expect := &api.PodSecurityConfiguration{
			Defaults: api.PodSecurityDefaults{
				Enforce: "baseline", EnforceVersion: "latest",
				Warn: "privileged", WarnVersion: "latest",
				Audit: "privileged", AuditVersion: "latest",
			},
		}

		config, err := LoadFromReader(bytes.NewBufferString(input))
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, expect) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(expect, config))
		}
	}

	// valid reader
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v1beta1",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`
		expect := &api.PodSecurityConfiguration{
			Defaults: api.PodSecurityDefaults{
				Enforce: "baseline", EnforceVersion: "latest",
				Warn: "privileged", WarnVersion: "latest",
				Audit: "privileged", AuditVersion: "latest",
			},
		}

		config, err := LoadFromReader(bytes.NewBufferString(input))
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config, expect) {
			t.Fatalf("unexpected config:\n%s", cmp.Diff(expect, config))
		}
	}

	// invalid reader
	{
		input := `{
			"apiVersion":"pod-security.admission.config.k8s.io/v99",
			"kind":"PodSecurityConfiguration",
			"defaults":{"enforce":"baseline"}}`

		_, err := LoadFromReader(bytes.NewBufferString(input))
		if err == nil {
			t.Fatalf("expected err, got none")
		}
		if !strings.Contains(err.Error(), "pod-security.admission.config.k8s.io/v99") {
			t.Fatalf("expected apiVersion error, got %v", err)
		}
	}
}

func TestLoadFromData(t *testing.T) {
	testcases := []struct {
		name         string
		data         []byte
		expectErr    string
		expectConfig *api.PodSecurityConfiguration
	}{
		{
			name:         "nil",
			data:         nil,
			expectConfig: defaultConfig,
		},
		{
			name:         "nil",
			data:         []byte{},
			expectConfig: defaultConfig,
		},
		{
			name: "v1alpha1 - json",
			data: []byte(`{
"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1",
"kind":"PodSecurityConfiguration",
"defaults":{"enforce":"baseline"}}`),
			expectConfig: &api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce: "baseline", EnforceVersion: "latest",
					Warn: "privileged", WarnVersion: "latest",
					Audit: "privileged", AuditVersion: "latest",
				},
			},
		},
		{
			name: "v1alpha1 - yaml",
			data: []byte(`
apiVersion: pod-security.admission.config.k8s.io/v1alpha1
kind: PodSecurityConfiguration
defaults:
  enforce: baseline
  enforce-version: v1.7
exemptions:
  usernames: ["alice","bob"]
  namespaces: ["kube-system"]
  runtimeClasses: ["special"]
`),
			expectConfig: &api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce: "baseline", EnforceVersion: "v1.7",
					Warn: "privileged", WarnVersion: "latest",
					Audit: "privileged", AuditVersion: "latest",
				},
				Exemptions: api.PodSecurityExemptions{
					Usernames:      []string{"alice", "bob"},
					Namespaces:     []string{"kube-system"},
					RuntimeClasses: []string{"special"},
				},
			},
		},
		{
			name: "v1beta1 - json",
			data: []byte(`{
"apiVersion":"pod-security.admission.config.k8s.io/v1beta1",
"kind":"PodSecurityConfiguration",
"defaults":{"enforce":"baseline"}}`),
			expectConfig: &api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce: "baseline", EnforceVersion: "latest",
					Warn: "privileged", WarnVersion: "latest",
					Audit: "privileged", AuditVersion: "latest",
				},
			},
		},
		{
			name: "v1beta1 - yaml",
			data: []byte(`
apiVersion: pod-security.admission.config.k8s.io/v1beta1
kind: PodSecurityConfiguration
defaults:
  enforce: baseline
  enforce-version: v1.7
exemptions:
  usernames: ["alice","bob"]
  namespaces: ["kube-system"]
  runtimeClasses: ["special"]
`),
			expectConfig: &api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce: "baseline", EnforceVersion: "v1.7",
					Warn: "privileged", WarnVersion: "latest",
					Audit: "privileged", AuditVersion: "latest",
				},
				Exemptions: api.PodSecurityExemptions{
					Usernames:      []string{"alice", "bob"},
					Namespaces:     []string{"kube-system"},
					RuntimeClasses: []string{"special"},
				},
			},
		},
		{
			name:      "missing apiVersion",
			data:      []byte(`{"kind":"PodSecurityConfiguration"}`),
			expectErr: `'apiVersion' is missing`,
		},
		{
			name:      "missing kind",
			data:      []byte(`{"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1"}`),
			expectErr: `'Kind' is missing`,
		},
		{
			name:      "unknown group",
			data:      []byte(`{"apiVersion":"apps/v1alpha1","kind":"PodSecurityConfiguration"}`),
			expectErr: `apps/v1alpha1`,
		},
		{
			name:      "unknown version",
			data:      []byte(`{"apiVersion":"pod-security.admission.config.k8s.io/v99","kind":"PodSecurityConfiguration"}`),
			expectErr: `pod-security.admission.config.k8s.io/v99`,
		},
		{
			name:      "unknown kind",
			data:      []byte(`{"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1","kind":"SomeConfiguration"}`),
			expectErr: `SomeConfiguration`,
		},
		{
			name: "unknown field",
			data: []byte(`{
"apiVersion":"pod-security.admission.config.k8s.io/v1alpha1",
"kind":"PodSecurityConfiguration",
"deflaults":{"enforce":"baseline"}}`),
			expectErr: `unknown field "deflaults"`,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			config, err := LoadFromData(tc.data)
			if err != nil {
				if len(tc.expectErr) == 0 {
					t.Fatalf("unexpected error: %v", err)
				}
				if !strings.Contains(err.Error(), tc.expectErr) {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if len(tc.expectErr) > 0 {
				t.Fatalf("expected err, got none")
			}

			if !reflect.DeepEqual(config, tc.expectConfig) {
				t.Fatalf("unexpected config:\n%s", cmp.Diff(tc.expectConfig, config))
			}
		})
	}
}
