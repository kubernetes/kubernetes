package dockerfile

import (
	"testing"

	"github.com/docker/docker/pkg/testutil/tempfile"
	"github.com/stretchr/testify/assert"
)

func TestIsExistingDirectory(t *testing.T) {
	tmpfile := tempfile.NewTempFile(t, "file-exists-test", "something")
	defer tmpfile.Remove()
	tmpdir := tempfile.NewTempDir(t, "dir-exists-test")
	defer tmpdir.Remove()

	var testcases = []struct {
		doc      string
		path     string
		expected bool
	}{
		{
			doc:      "directory exists",
			path:     tmpdir.Path,
			expected: true,
		},
		{
			doc:      "path doesn't exist",
			path:     "/bogus/path/does/not/exist",
			expected: false,
		},
		{
			doc:      "file exists",
			path:     tmpfile.Name(),
			expected: false,
		},
	}

	for _, testcase := range testcases {
		result, err := isExistingDirectory(testcase.path)
		if !assert.NoError(t, err) {
			continue
		}
		assert.Equal(t, testcase.expected, result, testcase.doc)
	}
}
