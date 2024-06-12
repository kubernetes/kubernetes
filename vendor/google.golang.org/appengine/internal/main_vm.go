// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

//go:build !appengine
// +build !appengine

package internal

import (
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
)

func Main() {
	MainPath = filepath.Dir(findMainPath())
	installHealthChecker(http.DefaultServeMux)

	port := "8080"
	if s := os.Getenv("PORT"); s != "" {
		port = s
	}

	host := ""
	if IsDevAppServer() {
		host = "127.0.0.1"
	}
	if err := http.ListenAndServe(host+":"+port, Middleware(http.DefaultServeMux)); err != nil {
		log.Fatalf("http.ListenAndServe: %v", err)
	}
}

// Find the path to package main by looking at the root Caller.
func findMainPath() string {
	pc := make([]uintptr, 100)
	n := runtime.Callers(2, pc)
	frames := runtime.CallersFrames(pc[:n])
	for {
		frame, more := frames.Next()
		// Tests won't have package main, instead they have testing.tRunner
		if frame.Function == "main.main" || frame.Function == "testing.tRunner" {
			return frame.File
		}
		if !more {
			break
		}
	}
	return ""
}

func installHealthChecker(mux *http.ServeMux) {
	// If no health check handler has been installed by this point, add a trivial one.
	const healthPath = "/_ah/health"
	hreq := &http.Request{
		Method: "GET",
		URL: &url.URL{
			Path: healthPath,
		},
	}
	if _, pat := mux.Handler(hreq); pat != healthPath {
		mux.HandleFunc(healthPath, func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "ok")
		})
	}
}
