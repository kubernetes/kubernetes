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
	targetDir := t.TempDir()

	type tc struct {
		name    string
		makeKey func(outsideFile string) string
	}

	cases := []tc{
		{
			name:    "drive-absolute",
			makeKey: func(outsideFile string) string { return outsideFile },
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
				rooted := strings.TrimPrefix(outsideFile, vol)
				return strings.ReplaceAll(rooted, `\`, `/`) // "/Users/..."
			},
		},
	}

	marker := []byte("H1_POC_OUTSIDE_WRITE\n")

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			baseTemp := os.TempDir()
			outsideDir := filepath.Join(baseTemp, "aw_"+c.name+"_"+time.Now().UTC().Format("20060102_150405.000000000"))
			outsideFile := filepath.Join(outsideDir, "pwn.txt")
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

			// (1) هل كُتب الملف خارج targetDir فعلاً؟
			if b, readErr := os.ReadFile(outsideFile); readErr == nil && bytes.Contains(b, marker) {
				t.Fatalf("HIGH-IMPACT: outside write observed (%s)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q\noutsideFile_content=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile, string(b))
			}

			// (2) فحص جنائي داخل targetDir: أين انتهى المحتوى؟
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
				t.Fatalf("IMPACT: payload accepted and written under targetDir with path confusion (%s)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\nmarker_hits=%v",
					c.name, writeErr, targetDir, payloadKey, hits)
			}

			t.Logf("No marker written (%s). writeErr=%v targetDir=%q payloadKey=%q outsideFile=%q",
				c.name, writeErr, targetDir, payloadKey, outsideFile)
		})
	}
}
