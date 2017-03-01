/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"
	"fmt"
	"io"
	"math"
)

// DiagnosticLog wraps DiagnosticManager.BrowseLog
type DiagnosticLog struct {
	m DiagnosticManager

	Key  string
	Host *HostSystem

	Start int32
}

// Seek to log position starting at the last nlines of the log
func (l *DiagnosticLog) Seek(ctx context.Context, nlines int32) error {
	h, err := l.m.BrowseLog(ctx, l.Host, l.Key, math.MaxInt32, 0)
	if err != nil {
		return err
	}

	l.Start = h.LineEnd - nlines

	return nil
}

// Copy log starting from l.Start to the given io.Writer
// Returns on error or when end of log is reached.
func (l *DiagnosticLog) Copy(ctx context.Context, w io.Writer) (int, error) {
	const max = 500 // VC max == 500, ESX max == 1000
	written := 0

	for {
		h, err := l.m.BrowseLog(ctx, l.Host, l.Key, l.Start, max)
		if err != nil {
			return 0, err
		}

		for _, line := range h.LineText {
			n, err := fmt.Fprintln(w, line)
			written += n
			if err != nil {
				return written, err
			}
		}

		l.Start += int32(len(h.LineText))

		if l.Start >= h.LineEnd {
			break
		}
	}

	return written, nil
}
