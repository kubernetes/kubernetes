/*
 *
 * Copyright 2015 gRPC authors.
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
	"bytes"
	"fmt"
	"io"
	"net"
	"strings"
	"time"

	"golang.org/x/net/trace"
)

// EnableTracing controls whether to trace RPCs using the golang.org/x/net/trace package.
// This should only be set before any RPCs are sent or received by this program.
var EnableTracing = true

// methodFamily returns the trace family for the given method.
// It turns "/pkg.Service/GetFoo" into "pkg.Service".
func methodFamily(m string) string {
	m = strings.TrimPrefix(m, "/") // remove leading slash
	if i := strings.Index(m, "/"); i >= 0 {
		m = m[:i] // remove everything from second slash
	}
	if i := strings.LastIndex(m, "."); i >= 0 {
		m = m[i+1:] // cut down to last dotted component
	}
	return m
}

// traceInfo contains tracing information for an RPC.
type traceInfo struct {
	tr        trace.Trace
	firstLine firstLine
}

// firstLine is the first line of an RPC trace.
type firstLine struct {
	client     bool // whether this is a client (outgoing) RPC
	remoteAddr net.Addr
	deadline   time.Duration // may be zero
}

func (f *firstLine) String() string {
	var line bytes.Buffer
	io.WriteString(&line, "RPC: ")
	if f.client {
		io.WriteString(&line, "to")
	} else {
		io.WriteString(&line, "from")
	}
	fmt.Fprintf(&line, " %v deadline:", f.remoteAddr)
	if f.deadline != 0 {
		fmt.Fprint(&line, f.deadline)
	} else {
		io.WriteString(&line, "none")
	}
	return line.String()
}

// payload represents an RPC request or response payload.
type payload struct {
	sent bool        // whether this is an outgoing payload
	msg  interface{} // e.g. a proto.Message
	// TODO(dsymonds): add stringifying info to codec, and limit how much we hold here?
}

func (p payload) String() string {
	if p.sent {
		return fmt.Sprintf("sent: %v", p.msg)
	}
	return fmt.Sprintf("recv: %v", p.msg)
}

type fmtStringer struct {
	format string
	a      []interface{}
}

func (f *fmtStringer) String() string {
	return fmt.Sprintf(f.format, f.a...)
}

type stringer string

func (s stringer) String() string { return string(s) }
