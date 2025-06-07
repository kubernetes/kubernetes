/*
Copyright 2025 The Kubernetes Authors.

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

package imagepullprogress

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/imagepullprogress"

	"k8s.io/klog/v2"
)

func handleHTTPStreams(req *http.Request, w http.ResponseWriter, imagePullProgresser ImagePullProgresser, podName string, uid types.UID, supportedImagePullProgressProtocols []string, idleTimeout time.Duration) error {
	_, err := httpstream.Handshake(req, w, supportedImagePullProgressProtocols)
	// negotiated protocol isn't currently used server side, but could be in the future
	if err != nil {
		// Handshake writes the error to the client
		return err
	}
	streamChan := make(chan httpstream.Stream, 1)

	klog.V(5).InfoS("Upgrading image pull progress response")
	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, httpStreamReceived(streamChan))
	if conn == nil {
		return errors.New("unable to upgrade httpstream connection")
	}
	defer conn.Close()

	klog.V(5).InfoS("Connection setting image pull progress streaming connection idle timeout", "connection", conn, "idleTimeout", idleTimeout)
	conn.SetIdleTimeout(idleTimeout)

	h := &httpStreamHandler{
		conn:       conn,
		streamChan: streamChan,
		pod:        podName,
		uid:        uid,
		progresser: imagePullProgresser,
	}
	h.run(req.Context())

	return nil
}

// httpStreamReceived is the httpstream.NewStreamHandler for image pull progress streams.
func httpStreamReceived(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streams <- stream
		return nil
	}
}

type httpStreamHandler struct {
	conn       httpstream.Connection
	streamChan chan httpstream.Stream
	pod        string
	uid        types.UID
	progresser ImagePullProgresser
}

func (h *httpStreamHandler) run(ctx context.Context) {
	klog.V(5).InfoS("Connection waiting for port forward streams", "connection", h.conn)
Loop:
	for {
		select {
		case <-h.conn.CloseChan():
			klog.V(5).InfoS("Connection upgraded connection closed", "connection", h.conn)
			break Loop
		case stream := <-h.streamChan:
			go h.imagePullProgress(ctx, stream)
		}
	}
}

func (h *httpStreamHandler) imagePullProgress(ctx context.Context, stream httpstream.Stream) {
	progresses := make(chan imagepullprogress.Progress)

	encoder := json.NewEncoder(stream)
	go func() {
		for progress := range progresses {
			err := encoder.Encode(progress)
			if err != nil {
				msg := fmt.Errorf("error sending image pull progress for pod %s, uid %v: %v", h.pod, h.uid, err)
				utilruntime.HandleError(msg)
			}
		}
	}()

	err := h.progresser.ImagePullProgress(ctx, h.pod, h.uid, progresses)
	if err != nil {
		msg := fmt.Errorf("error get image pull progress for pod %s, uid %v: %v", h.pod, h.uid, err)
		utilruntime.HandleError(msg)
	}
}
