/*
Copyright 2014 The Kubernetes Authors.

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

package routes

import (
	"io"
	"net/http"
	"path"
	"strings"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"github.com/emicklei/go-restful"
	"github.com/hpcloud/tail"
)

// Logs adds handlers for the /logs path serving log files from /var/log.
type Logs struct{}

// Install func registers the logs handler.
func (l Logs) Install(c *restful.Container) {
	// use restful: ws.Route(ws.GET("/logs/{logpath:*}").To(fileHandler))
	// See github.com/emicklei/go-restful/blob/master/examples/restful-serve-static.go
	ws := new(restful.WebService)
	ws.Path("/logs")
	ws.Doc("get log files")
	ws.Route(ws.GET("/{logpath:*}").To(logFileHandler).Param(ws.PathParameter("logpath", "path to the log").DataType("string")))
	ws.Route(ws.GET("/").To(logFileListHandler))

	c.Add(ws)
}

var logdir = "/var/log" // exposed for testing

func logFileHandler(req *restful.Request, resp *restful.Response) {
	actual := path.Join(logdir, req.PathParameter("logpath"))

	if containsDotDot(actual) {
		http.Error(resp.ResponseWriter, "invalid URL path", http.StatusBadRequest)
		return
	}

	if req.QueryParameter("follow") != "true" {
		http.ServeFile(resp.ResponseWriter, req.Request, actual)
		return
	}

	// Follow the logfile.
	file, err := tail.TailFile(actual, tail.Config{
		Follow: true,
		ReOpen: true,
		// NOTE(tallclair): Tests revealed the inotify based watcher to be quite flaky. We may want to
		// reevaluate this option in the future, but it should be thoroughly tested in multpile
		// environments before reverting.
		Poll:   true,
		Logger: tail.DiscardingLogger,
	})
	if err != nil {
		http.Error(resp.ResponseWriter, "Error reading file", http.StatusInternalServerError)
		return
	}

	// Write headers
	resp.AddHeader("Content-Type", "text/plain; charset=utf-8")
	resp.AddHeader("X-Content-Type-Options", "nosniff")
	resp.WriteHeader(http.StatusOK)

	for {
		// Drain the channel before flushing.
	drain:
		for {
			select {
			case line := <-file.Lines:
				if _, err := io.WriteString(resp, line.Text+"\n"); err != nil {
					utilruntime.HandleError(err)
					return
				}
			case <-resp.CloseNotify():
				// Client connection closed.
				return
			default:
				// Flush buffers & exit draining loop.
				resp.Flush()
				break drain
			}
		}

		select {
		case line := <-file.Lines:
			if _, err := io.WriteString(resp, line.Text+"\n"); err != nil {
				utilruntime.HandleError(err)
				return
			}
		case <-resp.CloseNotify():
			// Client connection closed.
			return
		}
	}
}

func logFileListHandler(req *restful.Request, resp *restful.Response) {
	http.ServeFile(resp.ResponseWriter, req.Request, logdir)
}

// copied from net/http/fs.go
func containsDotDot(v string) bool {
	if !strings.Contains(v, "..") {
		return false
	}
	for _, ent := range strings.FieldsFunc(v, isSlashRune) {
		if ent == ".." {
			return true
		}
	}
	return false
}
func isSlashRune(r rune) bool { return r == '/' || r == '\\' }
