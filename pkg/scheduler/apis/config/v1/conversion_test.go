/*
Copyright 2020 The Kubernetes Authors.

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

package v1

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestV1LegacyExtenderToConfigExtenderConversion(t *testing.T) {
	cases := []struct {
		name string
		in   v1.LegacyExtender
		out  config.Extender
		want config.Extender
	}{
		{
			name: "empty extender conversion",
			in:   v1.LegacyExtender{},
			out:  config.Extender{},
			want: config.Extender{},
		},
		{
			name: "fully configured extender conversion",
			in: v1.LegacyExtender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &v1.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      10 * time.Second,
				NodeCacheCapable: true,
				ManagedResources: []v1.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
			out: config.Extender{},
			want: config.Extender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &config.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      metav1.Duration{Duration: 10 * time.Second},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
		},
		{
			name: "clears empty fields",
			in:   v1.LegacyExtender{},
			out: config.Extender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &config.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      metav1.Duration{Duration: 10 * time.Second},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
			want: config.Extender{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := Convert_v1_LegacyExtender_To_config_Extender(&tc.in, &tc.out, nil); err != nil {
				t.Errorf("failed to convert: %+v", err)
			}
			if diff := cmp.Diff(tc.want, tc.out); diff != "" {
				t.Errorf("unexpected conversion (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestConfigExtenderToV1LegacyExtenderConversion(t *testing.T) {
	cases := []struct {
		name string
		in   config.Extender
		out  v1.LegacyExtender
		want v1.LegacyExtender
	}{
		{
			name: "empty extender conversion",
			in:   config.Extender{},
			out:  v1.LegacyExtender{},
			want: v1.LegacyExtender{},
		},
		{
			name: "fully configured extender conversion",
			in: config.Extender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &config.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      metav1.Duration{Duration: 10 * time.Second},
				NodeCacheCapable: true,
				ManagedResources: []config.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
			out: v1.LegacyExtender{},
			want: v1.LegacyExtender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &v1.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      10 * time.Second,
				NodeCacheCapable: true,
				ManagedResources: []v1.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
		},
		{
			name: "clears empty fields",
			in:   config.Extender{},
			out: v1.LegacyExtender{
				URLPrefix:      "/prefix",
				BindVerb:       "bind",
				FilterVerb:     "filter",
				PreemptVerb:    "preempt",
				PrioritizeVerb: "prioritize",
				Weight:         5,
				EnableHTTPS:    true,
				TLSConfig: &v1.ExtenderTLSConfig{
					Insecure:   true,
					ServerName: "server-name",
					CertFile:   "cert-file",
					KeyFile:    "key-file",
					CAFile:     "ca-file",
					CertData:   []byte("cert-data"),
					KeyData:    []byte("key-data"),
					CAData:     []byte("ca-data"),
				},
				HTTPTimeout:      10 * time.Second,
				NodeCacheCapable: true,
				ManagedResources: []v1.ExtenderManagedResource{
					{
						Name:               "managed-resource",
						IgnoredByScheduler: true,
					},
					{
						Name:               "another-resource",
						IgnoredByScheduler: false,
					},
				},
				Ignorable: true,
			},
			want: v1.LegacyExtender{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := Convert_config_Extender_To_v1_LegacyExtender(&tc.in, &tc.out, nil); err != nil {
				t.Errorf("failed to convert: %+v", err)
			}
			if diff := cmp.Diff(tc.want, tc.out); diff != "" {
				t.Errorf("unexpected conversion (-want, +got):\n%s", diff)
			}
		})
	}
}
