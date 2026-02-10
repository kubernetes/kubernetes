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
			// ✅ عزل كامل لكل case: targetDir مستقل (يمنع أي تداخل marker_hits)
			targetDir := t.TempDir()

			// ملف "خارج targetDir" لكن داخل Temp (آمن) — فقط لكشف خارج-الهدف لو حدث
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

			// (A) drive-absolute: دليل DoS حتمي (\C:: ...) — اجعله IMPACT واضح
			if c.name == "drive-absolute" {
				if writeErr == nil {
					t.Fatalf("IMPACT: drive-absolute unexpectedly succeeded (expected deterministic failure)\nwriteErr=<nil>\ntargetDir=%q\npayloadKey=%q",
						targetDir, payloadKey)
				}
				// نتحقق من البصمة المعروفة في اللوج: \C:: + syntax is incorrect
				msg := writeErr.Error()
				if !strings.Contains(msg, "C::") && !strings.Contains(strings.ToLower(msg), "syntax is incorrect") {
					t.Fatalf("IMPACT: drive-absolute failed but missing expected Windows path syntax signature\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q",
						writeErr, targetDir, payloadKey)
				}
				t.Fatalf("IMPACT: drive-absolute path reaches filesystem ops and triggers deterministic Windows path syntax failure (\\C:: ... syntax is incorrect)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q",
					writeErr, targetDir, payloadKey, outsideFile)
			}

			// (B) rooted-forwardslash: غالبًا يجب أن يُرفض (must be relative path)
			if c.name == "rooted-forwardslash-no-volume" && writeErr == nil {
				t.Fatalf("IMPACT: rooted forward-slash path unexpectedly accepted (expected rejection on Windows)\ntargetDir=%q\npayloadKey=%q",
					targetDir, payloadKey)
			}

			// (1) هل حصلت كتابة فعلية خارج targetDir؟ (لو حدث → HIGH-IMPACT)
			if b, readErr := os.ReadFile(outsideFile); readErr == nil && bytes.Contains(b, marker) {
				t.Fatalf("HIGH-IMPACT: outside write observed (%s)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q\noutsideFile_content=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile, string(b))
			}

			// (2) Forensics داخل targetDir: أين انتهى المحتوى؟
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

			// ✅ هذا هو الدليل الأساسي الذي نريده لـ rooted-backslash: كتابة داخل targetDir بمسار مشوّه
			if len(hits) > 0 {
				t.Fatalf("IMPACT: payload accepted and written under targetDir with path confusion (%s)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\nmarker_hits=%v",
					c.name, writeErr, targetDir, payloadKey, hits)
			}

			t.Logf("No marker written (%s). writeErr=%v targetDir=%q payloadKey=%q outsideFile=%q",
				c.name, writeErr, targetDir, payloadKey, outsideFile)
		})
	}
}
