package logrus

// The following code was sourced and modified from the
// https://github.com/tebeka/atexit package governed by the following license:
//
// Copyright (c) 2012 Miki Tebeka <miki.tebeka@gmail.com>.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import (
	"fmt"
	"os"
)

var handlers = []func(){}

func runHandler(handler func()) {
	defer func() {
		if err := recover(); err != nil {
			fmt.Fprintln(os.Stderr, "Error: Logrus exit handler error:", err)
		}
	}()

	handler()
}

func runHandlers() {
	for _, handler := range handlers {
		runHandler(handler)
	}
}

// Exit runs all the Logrus atexit handlers and then terminates the program using os.Exit(code)
func Exit(code int) {
	runHandlers()
	os.Exit(code)
}

// RegisterExitHandler adds a Logrus Exit handler, call logrus.Exit to invoke
// all handlers. The handlers will also be invoked when any Fatal log entry is
// made.
//
// This method is useful when a caller wishes to use logrus to log a fatal
// message but also needs to gracefully shutdown. An example usecase could be
// closing database connections, or sending a alert that the application is
// closing.
func RegisterExitHandler(handler func()) {
	handlers = append(handlers, handler)
}
