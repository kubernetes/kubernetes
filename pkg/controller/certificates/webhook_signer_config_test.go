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

package certificates

import (
	"io/ioutil"
	"os"
	"testing"
	"text/template"
	"time"
)

const configFileTmplJSON = `
{
	"retryBackoff": {{ .RetryBackoff }},
	"kubeConfigFile": "{{ .KubeConfigFile }}"
}
`

const configFileTmplInvalidJSON = `
{
	"retryBackoff" {{ .RetryBackoff }}
	"kubeConfigFile": "{{ .KubeConfigFile }}"

`

const configFileTmplYAML = `
retryBackoff: {{ .RetryBackoff }}
kubeConfigFile: {{ .KubeConfigFile }}
`

const configFileTmplMissingRetry = `
kubeConfigFile: {{ .KubeConfigFile }}
`

const configFileTmplMissingKubeConfig = `
retryBackoff: {{ .RetryBackoff }}
`

const configFileTmplKubeConfigNonExistent = `
retryBackoff: {{ .RetryBackoff }}
kubeConfigFile: '/this/does/not/exist'
`

const kubeConfig = `
clusters:
- cluster:
    server: https://127.0.0.1
    name: testcluster
users:
- user:
    username: admin
    password: mypass
`

func TestWebhookSignerFromConfigFile(t *testing.T) {
	cases := []struct {
		configFileTmpl       string
		configRetryBackoff   int64
		expectedRetryBackoff time.Duration
		wantErr              bool
	}{
		{
			configFileTmpl:       configFileTmplJSON,
			configRetryBackoff:   2000,
			expectedRetryBackoff: time.Duration(2) * time.Second,
			wantErr:              false,
		},
		{
			configFileTmpl:       configFileTmplYAML,
			configRetryBackoff:   2000,
			expectedRetryBackoff: time.Duration(2) * time.Second,
			wantErr:              false,
		},
		{
			configFileTmpl:     configFileTmplInvalidJSON,
			configRetryBackoff: 2000,
			wantErr:            true,
		},
		{
			configFileTmpl:       configFileTmplMissingRetry,
			expectedRetryBackoff: time.Duration(500) * time.Millisecond,
			wantErr:              false,
		},
		{
			configFileTmpl:     configFileTmplMissingKubeConfig,
			configRetryBackoff: 2000,
			wantErr:            true,
		},
		{
			configFileTmpl:     configFileTmplKubeConfigNonExistent,
			configRetryBackoff: 2000,
			wantErr:            true,
		},
	}

	kubeConfigFile, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatalf("error creating kubeconfig tempfile: %v", err)
	}

	defer os.Remove(kubeConfigFile.Name())

	if _, err := kubeConfigFile.Write([]byte(kubeConfig)); err != nil {
		t.Fatalf("error writing to kubeconfig tempfile: %v", err)
	}

	if err := kubeConfigFile.Close(); err != nil {
		t.Fatalf("error closing kubeconfig tempfile: %v", err)
	}

	for _, c := range cases {
		configFile, err := ioutil.TempFile("", "webhookconfig")
		if err != nil {
			t.Fatalf("error creating webhook tempfile: %v", err)
		}
		defer os.Remove(configFile.Name())

		tmpl, err := template.New("test").Parse(c.configFileTmpl)
		if err != nil {
			t.Fatalf("error creating template: %v", err)
		}

		data := struct {
			KubeConfigFile string
			RetryBackoff   int64
		}{
			KubeConfigFile: kubeConfigFile.Name(),
			RetryBackoff:   c.configRetryBackoff,
		}

		if err := tmpl.Execute(configFile, data); err != nil {
			t.Fatalf("error executing template: %v", err)
		}

		if err := configFile.Close(); err != nil {
			t.Fatalf("error closing tempfile: %v", err)
		}

		signer, err := NewWebhookSignerFromConfigFile(configFile.Name())

		if !c.wantErr {
			if err != nil {
				t.Errorf("unexpected error during constructions: %v", err)
			}

			if signer.kubeConfigFile != kubeConfigFile.Name() {
				t.Errorf("kubeConfigFile didn't match expected %s: %s", kubeConfigFile.Name(), signer.kubeConfigFile)
			}

			if signer.retryBackoff != c.expectedRetryBackoff {
				t.Errorf("retryBackoff didn't match expected %v: %v", c.expectedRetryBackoff, signer.retryBackoff)
			}
		} else if err == nil {
			t.Error("expected error during construction, but got none")
		}
	}
}

func TestWebhookSignerMissingConfigFile(t *testing.T) {
	_, err := NewWebhookSignerFromConfigFile("/this/does/not/exist")
	if err == nil {
		t.Error("did not encounter an error when NewWebhookSignerFromConfigFile given a non-existent file")
	}
}
