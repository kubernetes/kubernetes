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

package policy

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestSysctls(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		allowed       bool
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "forbidden sysctls",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
		},
		{
			name: "forbidden sysctls, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "a"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[1].name", BadValue: "b"},
			},
		},
		{
			name: "new supported sysctls not supported: net.ipv4.ip_local_reserved_ports",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.ip_local_reserved_ports`,
		},
		{
			name: "new supported sysctls not supported: net.ipv4.ip_local_reserved_ports, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.ip_local_reserved_ports`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "net.ipv4.ip_local_reserved_ports"},
			},
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_time",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "7200"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_time`,
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_time, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "7200"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_time`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "net.ipv4.tcp_keepalive_time"},
			},
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_fin_timeout",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_fin_timeout", Value: "60"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_fin_timeout`,
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_fin_timeout, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_fin_timeout", Value: "60"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_fin_timeout`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "net.ipv4.tcp_fin_timeout"},
			},
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_intvl",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_intvl", Value: "75"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_intvl`,
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_intvl, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_intvl", Value: "75"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_intvl`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "net.ipv4.tcp_keepalive_intvl"},
			},
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_probes",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_probes", Value: "9"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_probes`,
		},
		{
			name: "new supported sysctls not supported: net.ipv4.tcp_keepalive_probes, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_probes", Value: "9"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `net.ipv4.tcp_keepalive_probes`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "net.ipv4.tcp_keepalive_probes"},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := sysctlsV1Dot0(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if !tc.allowed {
				if result.Allowed {
					t.Fatal("expected disallowed")
				}
				if e, a := tc.expectReason, result.ForbiddenReason; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if result.ErrList != nil {
					if diff := cmp.Diff(tc.expectErrList, *result.ErrList, cmpOpts...); diff != "" {
						t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
					}
				}
			} else if !result.Allowed {
				t.Fatal("expected allowed")
			}
		})
	}
}

func TestSysctls_1_27(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		allowed       bool
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "forbidden sysctls",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
		},
		{
			name: "forbidden sysctls, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "a"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[1].name", BadValue: "b"},
			},
		},
		{
			name: "new supported sysctls",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"}},
				},
			}},
			allowed: true,
		},
		{
			name: "new supported sysctls, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.ip_local_reserved_ports", Value: "1024-4999"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := sysctlsV1Dot27(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if !tc.allowed {
				if result.Allowed {
					t.Fatal("expected disallowed")
				}
				if e, a := tc.expectReason, result.ForbiddenReason; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if result.ErrList != nil {
					if diff := cmp.Diff(tc.expectErrList, *result.ErrList, cmpOpts...); diff != "" {
						t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
					}
				}
			} else if !result.Allowed {
				t.Fatal("expected allowed")
			}
		})
	}
}

func TestSysctls_1_29(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		allowed       bool
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "forbidden sysctls",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
		},
		{
			name: "forbidden sysctls, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "a"}, {Name: "b"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed:      false,
			expectReason: `forbidden sysctls`,
			expectDetail: `a, b`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[0].name", BadValue: "a"},
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.sysctls[1].name", BadValue: "b"},
			},
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_time",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "7200"}},
				},
			}},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_time, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "7200"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_fin_timeout",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_fin_timeout", Value: "60"}},
				},
			}},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_fin_timeout, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_fin_timeout", Value: "60"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_intvl",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_intvl", Value: "75"}},
				},
			}},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_intvl, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_intvl", Value: "75"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_probes",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_probes", Value: "9"}},
				},
			}},
			allowed: true,
		},
		{
			name: "new supported sysctls: net.ipv4.tcp_keepalive_probes, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					Sysctls: []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_probes", Value: "9"}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := sysctlsV1Dot29(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if !tc.allowed {
				if result.Allowed {
					t.Fatal("expected disallowed")
				}
				if e, a := tc.expectReason, result.ForbiddenReason; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
					t.Errorf("expected\n%s\ngot\n%s", e, a)
				}
				if result.ErrList != nil {
					if diff := cmp.Diff(tc.expectErrList, *result.ErrList, cmpOpts...); diff != "" {
						t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
					}
				}
			} else if !result.Allowed {
				t.Fatal("expected allowed")
			}
		})
	}
}
