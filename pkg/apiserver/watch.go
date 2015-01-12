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
	"path"
	"regexp"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"

	"github.com/golang/glog"
	"golang.org/x/net/websocket"
)

type WatchHandler struct {
	storage         map[string]RESTStorage
	codec           runtime.Codec
	canonicalPrefix string
	selfLinker      runtime.SelfLinker
}

// setSelfLinkAddName sets the self link, appending the object's name to the canonical path & type.
func (h *WatchHandler) setSelfLinkAddName(obj runtime.Object, req *http.Request) error {
	name, err := h.selfLinker.Name(obj)
	if err != nil {
		return err
	}
	newURL := *req.URL
	newURL.Path = path.Join(h.canonicalPrefix, req.URL.Path, name)
	newURL.RawQuery = ""
	newURL.Fragment = ""
	return h.selfLinker.SetSelfLink(obj, newURL.String())
}

func getWatchParams(query url.Values) (label, field labels.Selector, resourceVersion string, err error) {
	s, perr := labels.ParseSelector(query.Get("labels"))
	if perr != nil {
		err = perr
		return
	}
	label = s

	s, perr = labels.ParseSelector(query.Get("fields"))
	if perr != nil {
		err = perr
		return
	}
	field = s

	resourceVersion = query.Get("resourceVersion")
	return
}

var connectionUpgradeRegex = regexp.MustCompile("(^|.*,\\s*)upgrade($|\\s*,)")

func isWebsocketRequest(req *http.Request) bool {
	return connectionUpgradeRegex.MatchString(strings.ToLower(req.Header.Get("Connection"))) && strings.ToLower(req.Header.Get("Upgrade")) == "websocket"
}

// ServeHTTP processes watch requests.
func (h *WatchHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != "GET" {
		notFound(w, req)
		return
	}

	namespace, kind, _, err := KindAndNamespace(req)
	if err != nil {
		notFound(w, req)
		return
	}
	ctx := api.WithNamespace(api.NewContext(), namespace)

	storage := h.storage[kind]
	if storage == nil {
		notFound(w, req)
		return
	}
	watcher, ok := storage.(ResourceWatcher)
	if !ok {
		errorJSON(errors.NewMethodNotSupported(kind, "watch"), h.codec, w)
		return
	}

	label, field, resourceVersion, err := getWatchParams(req.URL.Query())
	if err != nil {
		errorJSON(err, h.codec, w)
		return
	}
	watching, err := watcher.Watch(ctx, label, field, resourceVersion)
	if err != nil {
		errorJSON(err, h.codec, w)
		return
	}

	// TODO: This is one watch per connection. We want to multiplex, so that
	// multiple watches of the same thing don't create two watches downstream.
	watchServer := &WatchServer{watching, h.codec, func(obj runtime.Object) {
		if err := h.setSelfLinkAddName(obj, req); err != nil {
			glog.Errorf("Failed to set self link for object %#v", obj)
		}
	}}
	if isWebsocketRequest(req) {
		websocket.Handler(watchServer.HandleWS).ServeHTTP(httplog.Unlogged(w), req)
	} else {
		watchServer.ServeHTTP(w, req)
	}
}

// WatchServer serves a watch.Interface over a websocket or vanilla HTTP.
type WatchServer struct {
	watching watch.Interface
	codec    runtime.Codec
	fixup    func(runtime.Object)
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
			w.fixup(event.Object)
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
			self.fixup(event.Object)
			if err := encoder.Encode(&event); err != nil {
				// Client disconnect.
				self.watching.Stop()
				return
			}
			flusher.Flush()
		}
	}
}
