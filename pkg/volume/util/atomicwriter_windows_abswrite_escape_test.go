//go:build windows
// +build windows

package util

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAtomicWriter_WindowsPathForms_CanWriteOutsideTargetDir(t *testing.T) {
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
			// مثال: \Users\...\Temp\...\pwn.txt  (rooted على نفس الـvolume)
			name: "rooted-backslash-no-volume",
			makeKey: func(outsideFile string) string {
				vol := filepath.VolumeName(outsideFile)     // "C:"
				return strings.TrimPrefix(outsideFile, vol) // "\Users\..."
			},
		},
		{
			// مثال: /Users/...  (بعض المسارات تتعامل معه Go كـ rooted أيضًا على Windows)
			name: "rooted-forwardslash-no-volume",
			makeKey: func(outsideFile string) string {
				vol := filepath.VolumeName(outsideFile)
				rooted := strings.TrimPrefix(outsideFile, vol) // "\Users\..."
				return strings.ReplaceAll(rooted, `\`, `/`)    // "/Users/..."
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			baseTemp := os.TempDir()
			outsideDir := filepath.Join(baseTemp, "aw_"+c.name+"_"+time.Now().UTC().Format("20060102_150405.000000000"))
			outsideFile := filepath.Join(outsideDir, "pwn.txt")

			// تأكد منطقي: outsideFile ليس تحت targetDir
			if rel, err := filepath.Rel(targetDir, outsideFile); err == nil {
				if !strings.HasPrefix(rel, "..") {
					t.Fatalf("sanity failed: outsideFile unexpectedly under targetDir: rel=%q targetDir=%q outsideFile=%q",
						rel, targetDir, outsideFile)
				}
			}

			t.Cleanup(func() { _ = os.RemoveAll(outsideDir) })

			w, err := NewAtomicWriter(targetDir, "h1-abswrite-poc")
			if err != nil {
				t.Fatalf("NewAtomicWriter error: %v targetDir=%q", err, targetDir)
			}

			payloadKey := c.makeKey(outsideFile)
			payload := map[string]FileProjection{
				payloadKey: {
					Data:   []byte("H1_POC_OUTSIDE_WRITE\n"),
					Mode:   int32(0644),
					FsUser: nil,
				},
			}

			writeErr := w.Write(payload, nil)

			// الدليل الحاسم: هل تم إنشاء الملف خارج targetDir؟
			if b, readErr := os.ReadFile(outsideFile); readErr == nil {
				t.Fatalf("HIGH-IMPACT: outside write observed (%s)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q\noutsideFile_content=%q",
					c.name, writeErr, targetDir, payloadKey, outsideFile, string(b))
			}

			t.Logf("No outside write observed (%s). writeErr=%v targetDir=%q payloadKey=%q outsideFile=%q",
				c.name, writeErr, targetDir, payloadKey, outsideFile)
		})
	}
}
