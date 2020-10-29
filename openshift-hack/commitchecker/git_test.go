package main

import (
	"testing"
)

func TestUpstreamSummaryPattern(t *testing.T) {
	tt := []struct {
		summary string
		valid   bool
	}{
		{valid: true, summary: "UPSTREAM: 12345: a change"},
		{valid: true, summary: "UPSTREAM: k8s.io/heapster: 12345: a change"},
		{valid: true, summary: "UPSTREAM: <carry>: a change"},
		{valid: true, summary: "UPSTREAM: <drop>: a change"},
		{valid: true, summary: "UPSTREAM: coreos/etcd: <carry>: a change"},
		{valid: true, summary: "UPSTREAM: coreos/etcd: <drop>: a change"},
		{valid: true, summary: "UPSTREAM: revert: 12345: a change"},
		{valid: true, summary: "UPSTREAM: revert: k8s.io/heapster: 12345: a change"},
		{valid: true, summary: "UPSTREAM: revert: <carry>: a change"},
		{valid: true, summary: "UPSTREAM: revert: <drop>: a change"},
		{valid: true, summary: "UPSTREAM: revert: coreos/etcd: <carry>: a change"},
		{valid: true, summary: "UPSTREAM: revert: coreos/etcd: <drop>: a change"},
		{valid: false, summary: "UPSTREAM: whoopsie daisy"},
		{valid: true, summary: "UPSTREAM: gopkg.in/ldap.v2: 51: exposed better API for paged search"},
	}
	for _, tc := range tt {
		t.Run(tc.summary, func(t *testing.T) {
			got := UpstreamSummaryPattern.Match([]byte(tc.summary))

			if tc.valid != got {
				t.Errorf("expected %#v, got %#v", tc.valid, got)
			}
		})
	}
}

func TestBumpPattern(t *testing.T) {
	tt := []struct {
		summary string
		valid   bool
	}{
		{valid: true, summary: "bump(*)"},
		{valid: false, summary: "not a bump"},
	}
	for _, tc := range tt {
		t.Run(tc.summary, func(t *testing.T) {
			got := BumpSummaryPattern.Match([]byte(tc.summary))

			if tc.valid != got {
				t.Errorf("expected %#v, got %#v", tc.valid, got)
			}
		})
	}
}
