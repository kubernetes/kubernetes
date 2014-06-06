// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package engine

import (
	"net/http"
	"path"
)

// ServeHTTP executes a job as specified by the http request `r`, and sends the
// result as an http response.
// This method allows an Engine instance to be passed as a standard http.Handler interface.
//
// Note that the protocol used in this methid is a convenience wrapper and is not the canonical
// implementation of remote job execution. This is because HTTP/1 does not handle stream multiplexing,
// and so cannot differentiate stdout from stderr. Additionally, headers cannot be added to a response
// once data has been written to the body, which makes it inconvenient to return metadata such
// as the exit status.
//
func (eng *Engine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	jobName := path.Base(r.URL.Path)
	jobArgs, exists := r.URL.Query()["a"]
	if !exists {
		jobArgs = []string{}
	}
	w.Header().Set("Job-Name", jobName)
	for _, arg := range jobArgs {
		w.Header().Add("Job-Args", arg)
	}
	job := eng.Job(jobName, jobArgs...)
	job.Stdout.Add(w)
	job.Stderr.Add(w)
	// FIXME: distinguish job status from engine error in Run()
	// The former should be passed as a special header, the former
	// should cause a 500 status
	w.WriteHeader(http.StatusOK)
	// The exit status cannot be sent reliably with HTTP1, because headers
	// can only be sent before the body.
	// (we could possibly use http footers via chunked encoding, but I couldn't find
	// how to use them in net/http)
	job.Run()
}
