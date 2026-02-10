//go:build windows
// +build windows

package util

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This test is intended to PASS after hardening validatePath/validatePayload on Windows.
// Expectations after the fix:
// - drive-absolute keys (C:\...) are rejected before filesystem operations.
// - rooted keys without a volume (\Users\...) are rejected.
// - forward-slash rooted keys (/Users/...) are normalized then rejected.
// - No marker is written outside targetDir and no marker is materialized under targetDir.
func TestAtomicWriter_WindowsPathForms_FindWhereDataLands(t *testing.T) {
	type tc struct {
		name    string
		makeKey func(outsideFile string) string
	}

	cases := []tc{
		{
			name:    "drive-absolute",
			makeKey: func(outsideFile string) string { return outsideFile }, // "C:\Users\...\pwn.txt"
		},
		{
			name: "rooted-backslash-no-volume",
			makeKey: func(outsideFile string) string {
				vol := filepath.VolumeName(outsideFile)     // "C:"
				return strings.TrimPrefix(outsideFile, vol) // "\Users\..."
			},
		},
		{
			name: "rooted-forwardslash-no-volume",
			makeKey: func(outsideFile string) string {
				vol := filepath.VolumeName(outsideFile)
				rooted := strings.TrimPrefix(outsideFile, vol) // "\Users\..."
				return strings.ReplaceAll(rooted, `\`, `/`)    // "/Users/..."
			},
		},
	}

	marker := []byte("H1_POC_MARKER\n")

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			// ✅ Full isolation per-case: independent targetDir avoids any marker_hits bleed.
			targetDir := t.TempDir()

			// A file outside targetDir (but still in temp) to detect any unexpected outside write.
			baseTemp := os.TempDir()
			outsideDir := filepath.Join(baseTemp, "aw_"+c.name+"_"+time.Now().UTC().Format("20060102_150405.000000000"))
			outsideFile := filepath.Join(outsideDir, "pwn.txt")
			if err := os.MkdirAll(outsideDir, 0o755); err != nil {
				t.Fatalf("setup: mkdir outsideDir: %v", err)
			}
			t.Cleanup(func() { _ = os.RemoveAll(outsideDir) })

			w, err := NewAtomicWriter(targetDir, "h1-abswrite-poc")
			if err != nil {
				t.Fatalf("NewAtomicWriter error: %v targetDir=%q", err, targetDir)
			}

			payloadKey := c.makeKey(outsideFile)
			payload := map[string]FileProjection{
				payloadKey: {Data: marker, Mode: int32(0644)},
			}

			writeErr := w.Write(payload, nil)

			// After hardening, ALL these path forms must be rejected.
			if writeErr == nil {
				t.Fatalf("expected invalid-path error, got nil\ncase=%s\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q",
					c.name, targetDir, payloadKey, outsideFile)
			}

			msgLower := strings.ToLower(writeErr.Error())

			// We expect a "must be relative path" class of rejection (from validatePath).
			if !strings.Contains(msgLower, "must be relative path") {
				t.Fatalf("unexpected error text (expected 'must be relative path')\ncase=%s\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile)
			}

			// Defense-in-depth: if we ever see the old signature "C::" or "syntax is incorrect",
			// it likely means the input reached filesystem ops (regression).
			if strings.Contains(msgLower, "c::") || strings.Contains(msgLower, "syntax is incorrect") {
				t.Fatalf("regression: drive/path input appears to have reached filesystem ops\ncase=%s\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile)
			}

			// (1) Ensure no outside write occurred.
			if b, readErr := os.ReadFile(outsideFile); readErr == nil && bytes.Contains(b, marker) {
				t.Fatalf("outside write observed (should never happen)\ncase=%s\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q\noutsideFile_content=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile, string(b))
			}

			// (2) Forensics inside targetDir: ensure marker never materialized anywhere under targetDir.
			var hits []string
			_ = filepath.WalkDir(targetDir, func(p string, d fs.DirEntry, err error) error {
				if err != nil || d.IsDir() {
					return nil
				}
				b, err := os.ReadFile(p)
				if err == nil && bytes.Contains(b, marker) {
					hits = append(hits, p)
				}
				return nil
			})

			if len(hits) > 0 {
				t.Fatalf("marker unexpectedly written under targetDir (should be rejected pre-write)\ncase=%s\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\nmarker_hits=%v",
					c.name, writeErr, targetDir, payloadKey, hits)
			}

			// Keep a helpful breadcrumb in logs without failing.
			t.Logf("OK: rejected invalid payload key (%s). writeErr=%v payloadKey=%q", c.name, writeErr, payloadKey)
		})
	}
}
