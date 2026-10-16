//go:build linux

/*
Copyright The Kubernetes Authors.

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

package logs

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/pkg/kubelet/container"
)

// TestCompressLogKeepsLogWhenArchiveIsIncomplete makes the final write of the
// gzip archive fail with ENOSPC, like it would on a full disk, and checks that
// the original log is kept instead of being replaced by a truncated archive.
func TestCompressLogKeepsLogWhenArchiveIsIncomplete(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "0.log.20260101-000000")
	// The log is empty, so nothing is written while copying and the gzip
	// header is only written when the archive is closed.
	require.NoError(t, os.WriteFile(log, nil, 0640))
	require.NoError(t, os.Symlink("/dev/full", log+tmpSuffix))

	c := &containerLogManager{osInterface: container.RealOS{}}
	require.Error(t, c.compressLog(log), "compressing should fail when the archive cannot be written")

	got, err := os.ReadFile(log)
	require.NoError(t, err, "original log should be kept")
	assert.Empty(t, got, "original log should be unchanged")
	_, err = os.Stat(log + compressSuffix)
	assert.True(t, os.IsNotExist(err), "incomplete archive should not be created")
	_, err = os.Lstat(log + tmpSuffix)
	assert.True(t, os.IsNotExist(err), "temporary archive should be removed")
}
