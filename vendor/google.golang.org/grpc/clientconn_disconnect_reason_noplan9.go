//go:build !plan9

/*
 *
 * Copyright 2026 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"context"
	"errors"
	"os"
	"syscall"
)

// disconnectErrorLabel returns the grpc.disconnect_error metric label for a
// transport error, as specified by gRFC A94.
func disconnectErrorLabel(err error) string {
	var sysErr syscall.Errno
	switch {
	case errors.Is(err, context.Canceled):
		return "subchannel shutdown"
	case errors.Is(err, syscall.ECONNRESET):
		return "connection reset"
	case errors.Is(err, syscall.ETIMEDOUT), errors.Is(err, context.DeadlineExceeded), errors.Is(err, os.ErrDeadlineExceeded):
		return "connection timed out"
	case errors.Is(err, syscall.ECONNABORTED):
		return "connection aborted"
	case errors.As(err, &sysErr):
		return "socket error"
	default:
		return "unknown"
	}
}
