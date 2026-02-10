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

func TestAtomicWriter_AbsolutePath_CanWriteOutsideTargetDir(t *testing.T) {
	targetDir := t.TempDir()

	// خارج targetDir لكن ما زال داخل Temp (آمن)
	baseTemp := os.TempDir()
	outsideDir := filepath.Join(baseTemp, "aw_abswrite_"+time.Now().UTC().Format("20060102_150405.000000000"))
	outsideFile := filepath.Join(outsideDir, "pwn.txt")

	// تأكيد منطقي: outsideFile ليس تحت targetDir
	if rel, err := filepath.Rel(targetDir, outsideFile); err == nil {
		// rel لو كان داخل targetDir غالبًا لا يبدأ بـ ".."
		if !strings.HasPrefix(rel, "..") {
			t.Fatalf("sanity failed: outsideFile unexpectedly under targetDir: rel=%q targetDir=%q outsideFile=%q", rel, targetDir, outsideFile)
		}
	}

	// نظّف بعد الاختبار
	t.Cleanup(func() { _ = os.RemoveAll(outsideDir) })

	w, err := NewAtomicWriter(targetDir, "h1-abswrite-poc")
	if err != nil {
		t.Fatalf("NewAtomicWriter error: %v targetDir=%q", err, targetDir)
	}

	payloadKey := outsideFile // <- هذا هو “absolute path” الذي نريد إثبات أثره
	payload := map[string]FileProjection{
		payloadKey: {
			Data:   []byte("H1_POC_OUTSIDE_WRITE\n"),
			Mode:   int32(0644),
			FsUser: nil,
		},
	}

	// قد يرجع Error لاحقًا بسبب symlink steps… هذا لا يهمنا إن كان “الكتابة” حصلت قبل الفشل
	writeErr := w.Write(payload, nil)

	// الدليل الحاسم: هل الملف اتخلق خارج targetDir؟
	if b, readErr := os.ReadFile(outsideFile); readErr == nil {
		t.Fatalf("HIGH-IMPACT: outside write observed (absolute path accepted)\nwriteErr=%v\ntargetDir=%q\npayloadKey=%q\noutsideFile=%q\noutsideFile_content=%q",
			writeErr, targetDir, payloadKey, outsideFile, string(b))
	}

	t.Logf("No outside write observed. writeErr=%v targetDir=%q payloadKey=%q outsideFile=%q", writeErr, targetDir, payloadKey, outsideFile)
}
