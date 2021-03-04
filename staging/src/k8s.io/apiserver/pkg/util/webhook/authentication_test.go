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

package webhook

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestAuthenticationDetection(t *testing.T) {
	tests := []struct {
		name       string
		kubeconfig clientcmdapi.Config
		serverName string
		expected   rest.Config
	}{
		{
			name:       "empty",
			serverName: "foo.com",
		},
		{
			name:       "fallback to current context",
			serverName: "foo.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"bar.com": {Token: "bar"},
				},
				Contexts: map[string]*clientcmdapi.Context{
					"ctx": {
						AuthInfo: "bar.com",
					},
				},
				CurrentContext: "ctx",
			},
			expected: rest.Config{BearerToken: "bar"},
		},
		{
			name:       "exact match",
			serverName: "foo.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo.com": {Token: "foo"},
					"*.com":   {Token: "foo-star"},
					"bar.com": {Token: "bar"},
				},
			},
			expected: rest.Config{BearerToken: "foo"},
		},
		{
			name:       "partial star match",
			serverName: "foo.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.com":   {Token: "foo-star"},
					"bar.com": {Token: "bar"},
				},
			},
			expected: rest.Config{BearerToken: "foo-star"},
		},
		{
			name:       "full star match",
			serverName: "foo.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*":       {Token: "star"},
					"bar.com": {Token: "bar"},
				},
			},
			expected: rest.Config{BearerToken: "star"},
		},
		{
			name:       "skip bad in cluster config",
			serverName: "kubernetes.default.svc",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*":       {Token: "star"},
					"bar.com": {Token: "bar"},
				},
			},
			expected: rest.Config{BearerToken: "star"},
		},
		{
			name:       "most selective",
			serverName: "one.two.three.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.two.three.com": {Token: "first"},
					"*.three.com":     {Token: "second"},
					"*.com":           {Token: "third"},
				},
			},
			expected: rest.Config{BearerToken: "first"},
		},
		{
			name:       "exact match with default https port",
			serverName: "one.two.three.com:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"one.two.three.com:443": {Token: "exact"},
					"*.two.three.com":       {Token: "first"},
					"*.three.com":           {Token: "second"},
					"*.com":                 {Token: "third"},
					"*":                     {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "exact"},
		},
		{
			name:       "wildcard match with default https port",
			serverName: "one.two.three.com:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.two.three.com:443": {Token: "first-with-port"},
					"*.two.three.com":     {Token: "first"},
					"*.three.com":         {Token: "second"},
					"*.com":               {Token: "third"},
					"*":                   {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-with-port"},
		},
		{
			name:       "wildcard match without default https port",
			serverName: "one.two.three.com:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.two.three.com": {Token: "first"},
					"*.three.com":     {Token: "second"},
					"*.com":           {Token: "third"},
					"*":               {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first"},
		},
		{
			name:       "exact match with non-default https port",
			serverName: "one.two.three.com:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"one.two.three.com:8443": {Token: "exact"},
					"*.two.three.com":        {Token: "first"},
					"*.three.com":            {Token: "second"},
					"*.com":                  {Token: "third"},
					"*":                      {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "exact"},
		},
		{
			name:       "wildcard match with non-default https port",
			serverName: "one.two.three.com:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.two.three.com:8443": {Token: "first-with-port"},
					"one.two.three.com":    {Token: "first-without-port"},
					"*.two.three.com":      {Token: "first"},
					"*.three.com":          {Token: "second"},
					"*.com":                {Token: "third"},
					"*":                    {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-with-port"},
		},
		{
			name:       "wildcard match without non-default https port",
			serverName: "one.two.three.com:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"one.two.three.com": {Token: "first-without-port"},
					"*.two.three.com":   {Token: "first"},
					"*.three.com":       {Token: "second"},
					"*.com":             {Token: "third"},
					"*":                 {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "fallback"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resolver := defaultAuthenticationInfoResolver{kubeconfig: tc.kubeconfig}
			actual, err := resolver.ClientConfigFor(tc.serverName)
			if err != nil {
				t.Fatal(err)
			}
			actual.UserAgent = ""
			actual.Timeout = 0

			if !equality.Semantic.DeepEqual(*actual, tc.expected) {
				t.Errorf("%v", diff.ObjectReflectDiff(tc.expected, *actual))
			}
		})
	}

}
