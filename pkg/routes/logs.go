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
	"net/http"
	"os"
	"path"

	"github.com/emicklei/go-restful/v3"
)

// Logs adds handlers for the /logs path serving log files from /var/log.
type Logs struct{}

// Install func registers the logs handler.
func (l Logs) Install(c *restful.Container) {
	// use restful: ws.Route(ws.GET("/logs/{logpath:*}").To(fileHandler))
	// See github.com/emicklei/go-restful/blob/master/examples/static/restful-serve-static.go
	ws := new(restful.WebService)
	ws.Path("/logs")
	ws.Doc("get log files")
	ws.Route(ws.GET("/{logpath:*}").To(logFileHandler).Param(ws.PathParameter("logpath", "path to the log").DataType("string")))
	ws.Route(ws.GET("/").To(logFileListHandler))

	c.Add(ws)
}

func logFileHandler(req *restful.Request, resp *restful.Response) {
	logdir := "/var/log"
	actual := path.Join(logdir, req.PathParameter("logpath"))

	// check filename length first, return 404 if it's oversize.
	if logFileNameIsTooLong(actual) {
		http.Error(resp, "file not found", http.StatusNotFound)
		return
	}
	http.ServeFile(resp.ResponseWriter, req.Request, actual)
}

func logFileListHandler(req *restful.Request, resp *restful.Response) {
	logdir := "/var/log"
	http.ServeFile(resp.ResponseWriter, req.Request, logdir)
}

// logFileNameIsTooLong checks filename length, returns true if it's longer than 255.
// cause http.ServeFile returns default error code 500 except for NotExist and Forbidden, but we need to separate the real 500 from oversize filename here.
func logFileNameIsTooLong(filePath string) bool {
	_, err := os.Stat(filePath)
	if err != nil {
		if e, ok := err.(*os.PathError); ok && e.Err == fileNameTooLong {
			return true
		}
	}
	return false
}
