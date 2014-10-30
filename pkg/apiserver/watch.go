/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"code.google.com/p/go.net/websocket"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
)

type WatchHandler struct {
	storage map[string]RESTStorage
	codec   runtime.Codec
}

func getWatchParams(query url.Values) (label, field labels.Selector, resourceVersion string) {
	if s, err := labels.ParseSelector(query.Get("labels")); err != nil {
		label = labels.Everything()
	} else {
		label = s
	}
	if s, err := labels.ParseSelector(query.Get("fields")); err != nil {
		field = labels.Everything()
	} else {
		field = s
	}
	resourceVersion = query.Get("resourceVersion")
	return
}

var connectionUpgradeRegex = regexp.MustCompile("(^|.*,\\s*)upgrade($|\\s*,)")

func isWebsocketRequest(req *http.Request) bool {
	return connectionUpgradeRegex.MatchString(strings.ToLower(req.Header.Get("Connection"))) && strings.ToLower(req.Header.Get("Upgrade")) == "websocket"
}

// ServeHTTP processes watch requests.
func (h *WatchHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ctx := api.NewContext()
	namespace := req.URL.Query().Get("namespace")
	if len(namespace) > 0 {
		ctx = api.WithNamespace(ctx, namespace)
	}
	parts := splitPath(req.URL.Path)
	if len(parts) < 1 || req.Method != "GET" {
		notFound(w, req)
		return
	}
	storage := h.storage[parts[0]]
	if storage == nil {
		notFound(w, req)
		return
	}
	if watcher, ok := storage.(ResourceWatcher); ok {
		label, field, resourceVersion := getWatchParams(req.URL.Query())
		watching, err := watcher.Watch(ctx, label, field, resourceVersion)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		// TODO: This is one watch per connection. We want to multiplex, so that
		// multiple watches of the same thing don't create two watches downstream.
		watchServer := &WatchServer{watching, h.codec}
		if isWebsocketRequest(req) {
			websocket.Handler(watchServer.HandleWS).ServeHTTP(httplog.Unlogged(w), req)
		} else {
			watchServer.ServeHTTP(w, req)
		}
		return
	}

	notFound(w, req)
}

// WatchServer serves a watch.Interface over a websocket or vanilla HTTP.
type WatchServer struct {
	watching watch.Interface
	codec    runtime.Codec
}

// HandleWS implements a websocket handler.
func (w *WatchServer) HandleWS(ws *websocket.Conn) {
	done := make(chan struct{})
	go func() {
		var unused interface{}
		// Expect this to block until the connection is closed. Client should not
		// send anything.
		websocket.JSON.Receive(ws, &unused)
		close(done)
	}()
	for {
		select {
		case <-done:
			w.watching.Stop()
			return
		case event, ok := <-w.watching.ResultChan():
			if !ok {
				// End of results.
				return
			}
			obj, err := watchjson.Object(w.codec, &event)
			if err != nil {
				// Client disconnect.
				w.watching.Stop()
				return
			}
			if err := websocket.JSON.Send(ws, obj); err != nil {
				// Client disconnect.
				w.watching.Stop()
				return
			}
		}
	}
}

// ServeHTTP serves a series of JSON encoded events via straight HTTP with
// Transfer-Encoding: chunked.
func (self *WatchServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	loggedW := httplog.LogOf(req, w)
	w = httplog.Unlogged(w)

	cn, ok := w.(http.CloseNotifier)
	if !ok {
		loggedW.Addf("unable to get CloseNotifier")
		http.NotFound(w, req)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		loggedW.Addf("unable to get Flusher")
		http.NotFound(w, req)
		return
	}

	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	encoder := watchjson.NewEncoder(w, self.codec)
	for {
		select {
		case <-cn.CloseNotify():
			self.watching.Stop()
			return
		case event, ok := <-self.watching.ResultChan():
			if !ok {
				// End of results.
				return
			}
			if err := encoder.Encode(&event); err != nil {
				// Client disconnect.
				self.watching.Stop()
				return
			}
			flusher.Flush()
		}
	}
}
