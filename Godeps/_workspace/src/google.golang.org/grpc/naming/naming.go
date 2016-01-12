/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Package naming defines the naming API and related data structures for gRPC.
// The interface is EXPERIMENTAL and may be suject to change.
package naming

// OP defines the corresponding operations for a name resolution change.
type OP uint8

const (
	// Add indicates a new address is added.
	Add = iota
	// Delete indicates an exisiting address is deleted.
	Delete
)

type ServiceConfig interface{}

// Update defines a name resolution change.
type Update struct {
	// Op indicates the operation of the update.
	Op     OP
	Addr   string
	Config ServiceConfig
}

// Resolver does one-shot name resolution and creates a Watcher to
// watch the future updates.
type Resolver interface {
	// Resolve returns the name resolution results.
	Resolve(target string) ([]*Update, error)
	// NewWatcher creates a Watcher to watch the changes on target.
	NewWatcher(target string) Watcher
}

// Watcher watches the updates for a particular target.
type Watcher interface {
	// Next blocks until an update or error happens. It may return one or more
	// updates.
	Next() ([]*Update, error)
	// Stop stops the Watcher.
	Stop()
}
