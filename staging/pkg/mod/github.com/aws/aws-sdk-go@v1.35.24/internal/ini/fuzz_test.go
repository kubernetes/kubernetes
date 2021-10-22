// +build fuzz

// fuzz test data is stored in Amazon S3.
package ini_test

import (
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go/internal/ini"
)

// TestFuzz is used to test for crashes and not validity of the input
func TestFuzz(t *testing.T) {
	paths, err := filepath.Glob("testdata/fuzz/*")
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if paths == nil {
		t.Errorf("expected fuzz files, but received none")
	}

	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			ini.OpenFile(path)
		})
	}
}
