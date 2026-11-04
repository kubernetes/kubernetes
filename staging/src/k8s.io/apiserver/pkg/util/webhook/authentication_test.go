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

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/equality"
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
			name:       "match with impersonation",
			serverName: "foo.com",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo.com": {
						Token:                "foo",
						Impersonate:          "user-a",
						ImpersonateUID:       "user-a-uid-1111",
						ImpersonateGroups:    []string{"user-a-group1", "user-a-group2"},
						ImpersonateUserExtra: map[string][]string{"foo": {"bar", "baz", "etc"}},
					},
				},
			},
			expected: rest.Config{
				BearerToken: "foo",
				Impersonate: rest.ImpersonationConfig{
					UserName: "user-a",
					UID:      "user-a-uid-1111",
					Groups:   []string{"user-a-group1", "user-a-group2"},
					Extra:    map[string][]string{"foo": {"bar", "baz", "etc"}},
				},
			},
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
		{
			name:       "wildcard match ipv4 with non-default port",
			serverName: "1.2.3.4:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"1.2.3.4":      {Token: "exact-without-port"},
					"*.2.3.4:8443": {Token: "first-with-port"},
					"*.2.3.4":      {Token: "first-without-port"},
					"*.3.4":        {Token: "second-without-port"},
					"*.4":          {Token: "third-without-port"},
					"*":            {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-with-port"},
		},
		{
			name:       "wildcard match ipv4 with default port",
			serverName: "1.2.3.4:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.2.3.4": {Token: "first-without-port"},
					"*.3.4":   {Token: "second-without-port"},
					"*.4":     {Token: "third-without-port"},
					"*":       {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-without-port"},
		},
		{
			name:       "no wildcard match ipv6 with non-default port",
			serverName: "[::ffff:1.2.3.4]:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"::ffff:1.2.3.4": {Token: "exact-without-port"},
					"*.2.3.4:8443":   {Token: "first-with-port"},
					"*.2.3.4":        {Token: "first-without-port"},
					"*.3.4":          {Token: "second-without-port"},
					"*.4":            {Token: "third-without-port"},
					"*":              {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "fallback"},
		},
		{
			name:       "no wildcard match ipv6 with default port",
			serverName: "[::ffff:1.2.3.4]:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.2.3.4": {Token: "first-without-port"},
					"*.3.4":   {Token: "second-without-port"},
					"*.4":     {Token: "third-without-port"},
					"*":       {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "fallback"},
		},
		{
			name:       "explicit wildcard match ipv6 with non-default port",
			serverName: "[::ffff:1.2.3.4]:8443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"::ffff:1.2.3.4": {Token: "exact-without-port"},
					"*.2.3.4]:8443":  {Token: "first-with-port-with-bracket"},
					"*.2.3.4:8443":   {Token: "first-with-port"},
					"*.2.3.4":        {Token: "first-without-port"},
					"*.3.4":          {Token: "second-without-port"},
					"*.4":            {Token: "third-without-port"},
					"*":              {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-with-port-with-bracket"},
		},
		{
			name:       "explicit wildcard match ipv6 with default port",
			serverName: "[::ffff:1.2.3.4]:443",
			kubeconfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"*.2.3.4":  {Token: "first-without-port"},
					"*.2.3.4]": {Token: "first-without-port-with-bracket"},
					"*.3.4":    {Token: "second-without-port"},
					"*.4":      {Token: "third-without-port"},
					"*":        {Token: "fallback"},
				},
			},
			expected: rest.Config{BearerToken: "first-without-port-with-bracket"},
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
				t.Errorf("%v", cmp.Diff(tc.expected, *actual))
			}
		})
	}

}
