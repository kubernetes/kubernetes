/*
 *
 * Copyright 2015, Google Inc.
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

/*
Package grpclog defines logging for grpc.
*/
package grpclog

import (
	"log"
	"os"
)

// Use golang's standard logger by default.
// Access is not mutex-protected: do not modify except in init()
// functions.
var logger Logger = log.New(os.Stderr, "", log.LstdFlags)

// Logger mimics golang's standard Logger as an interface.
type Logger interface {
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Fatalln(args ...interface{})
	Print(args ...interface{})
	Printf(format string, args ...interface{})
	Println(args ...interface{})
}

// SetLogger sets the logger that is used in grpc. Call only from
// init() functions.
func SetLogger(l Logger) {
	logger = l
}

// Fatal is equivalent to Print() followed by a call to os.Exit() with a non-zero exit code.
func Fatal(args ...interface{}) {
	logger.Fatal(args...)
}

// Fatalf is equivalent to Printf() followed by a call to os.Exit() with a non-zero exit code.
func Fatalf(format string, args ...interface{}) {
	logger.Fatalf(format, args...)
}

// Fatalln is equivalent to Println() followed by a call to os.Exit()) with a non-zero exit code.
func Fatalln(args ...interface{}) {
	logger.Fatalln(args...)
}

// Print prints to the logger. Arguments are handled in the manner of fmt.Print.
func Print(args ...interface{}) {
	logger.Print(args...)
}

// Printf prints to the logger. Arguments are handled in the manner of fmt.Printf.
func Printf(format string, args ...interface{}) {
	logger.Printf(format, args...)
}

// Println prints to the logger. Arguments are handled in the manner of fmt.Println.
func Println(args ...interface{}) {
	logger.Println(args...)
}
