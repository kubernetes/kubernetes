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
	"path"

	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/genericapiserver/mux"
)

// Logs adds handlers for the /logs path serving log files from /var/log.
type Logs struct{}

func (l Logs) Install(c *mux.APIContainer) {
	// use restful: ws.Route(ws.GET("/logs/{logpath:*}").To(fileHandler))
	// See github.com/emicklei/go-restful/blob/master/examples/restful-serve-static.go
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
	http.ServeFile(resp.ResponseWriter, req.Request, actual)
}

func logFileListHandler(req *restful.Request, resp *restful.Response) {
	logdir := "/var/log"
	http.ServeFile(resp.ResponseWriter, req.Request, logdir)
}
