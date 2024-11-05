//go:build !linux

/*
Copyright 2024 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"io"
	"os"

	securejoin "github.com/cyphar/filepath-securejoin"
)

// heuristicsCopyFileLog returns the contents of the given logFile
func heuristicsCopyFileLog(ctx context.Context, w io.Writer, logDir, logFileName string) error {
	logFile, err := securejoin.SecureJoin(logDir, logFileName)
	if err != nil {
		return err
	}
	fInfo, err := os.Stat(logFile)
	if err != nil {
		return err
	}
	// This is to account for the heuristics where logs for service foo
	// could be in /var/log/foo/
	if fInfo.IsDir() {
		return os.ErrNotExist
	}

	f, err := os.Open(logFile)
	if err != nil {
		return err
	}
	// Ignoring errors when closing a file opened read-only doesn't cause data loss
	defer func() { _ = f.Close() }()

	if _, err := io.Copy(w, newReaderCtx(ctx, f)); err != nil {
		return err
	}
	return nil
}
