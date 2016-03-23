/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"fmt"
	"io"
	"github.com/golang/glog"
	"encoding/json"
	"strings"
	"sync"
)

// Current and total number of bytes downloaded
type DockerProgressDetail struct {
	Current int64 `json:"current"`
	Total int64 `json:"total"`
}

// Message that Docker produces indicating progress of an image pull, error is usually absent
type DockerProgressMessage struct {
	Status string `json:"status"`
	Err string `json:"error"`
	Id string `json:"id"`
	Progress string `json:"progress"`
	Detail DockerProgressDetail `json:"progressDetail"`
}

// Message that we send to clients querying the pull status
// This is just for one layer, we send an array of these
type LayerProgress struct {
	Id string `json:"id"`
	State string `json:"state"`
	Current int64 `json:"current"`
	Total int64 `json:"total"`
}

// Internal struct used to track a watching client
type watcher struct {
	w io.Writer
	finished chan bool
	err error
}

// The status of an image pull:
// Layers is the status of each layer in the image
// lock is a lock used to ensure that clients don't get garbled or partial updates
// watchers is a list of all the clients currently watching: a linked list would be more natural as we could easily remove aborted connections - for now we just skip them
type ImagePullStatus struct {
	Layers []*LayerProgress `json:"layers"`
	lock sync.Mutex
	watchers []*watcher
}

// Create an empty ImagePullStatus object
func NewImagePullStatus() *ImagePullStatus {
	return &ImagePullStatus {
		Layers: nil,
		watchers: nil,
	}
}

// Allows a client to watch the pull status
func (ips *ImagePullStatus) Watch(w io.Writer) error {
	watcher := watcher{
		w: w,
		finished: make(chan bool),
		err: ips.Get(w), // ensure they are up-to-date
	}
	// Lock before appending to the list, as another thread may be iterating over it
	ips.lock.Lock()
	ips.watchers = append(ips.watchers, &watcher)
	ips.lock.Unlock()
	// Block until the download is finished
	// The client will be sent updates by the MonitorPull function as they arrive from docker
	_ = <- watcher.finished
	return watcher.err
}

// Get a snapshot of the current state of the image pull
func (ips *ImagePullStatus) Get(w io.Writer) error {
	// Lock here so that we don't get a partial status
	ips.lock.Lock()
	js, err := json.Marshal(ips)
	ips.lock.Unlock()
	if err != nil {
		return err
	}
	// Write the status to the client
	_, err = fmt.Fprintf(w, "%s\n", js)
	return err
}

// Listen to a stream from docker and send updates to clients
func (ips *ImagePullStatus) MonitorPull(r io.ReadCloser)  {
	defer r.Close()
	// To decode status updates from docker
	decoder := json.NewDecoder(r)
	for {
		// Get the latest message
		var msg DockerProgressMessage
		err := decoder.Decode(&msg)
		if err != nil {
			glog.Errorf("%v\n", err)
			break
		}

		// This is the status seen when initiating a pull: it doesn't actually tell us anything useful
		if strings.HasPrefix(msg.Status, "Pulling from") {
			continue
		}

		updatedLayer := false

		ips.lock.Lock()

		// Attempt to find & update the layer in question
		for _, layer := range ips.Layers {
			if layer.Id == msg.Id {
				layer.State = msg.Status
				layer.Current = msg.Detail.Current
				layer.Total = msg.Detail.Total
				updatedLayer = true
				break
			}
		}

		// If we didn't find the layer then we need to add a new one
		if !updatedLayer {
			if msg.Id != "" {
				newLayer := LayerProgress {
					Id: msg.Id,
					State: msg.Status,
					Current: msg.Detail.Current,
					Total: msg.Detail.Total,
				}
				ips.Layers = append(ips.Layers, &newLayer)
			}
		}

		// Prepare to send out update
		js, err := json.Marshal(ips)
		if err != nil {
			glog.Errorf("%v\n", err)
			// Unlock here, otherwise the loop will terminate without releasing the lock
			// This condition will only happen if the stream is corrupted or the connection is broken
			ips.lock.Unlock()
			break;
		}

		// Send the update to all watchers
		for _, w := range(ips.watchers) {
			// Skip watchers that have already failed
			if w.err == nil {
				_, w.err = fmt.Fprintf(w.w, "%s\n", js)
			}
		}

		ips.lock.Unlock()

		if strings.HasPrefix(msg.Status, "Status:") {
			// Pull is complete
			break
		}
	}

	// Tell everybody that we finished
	ips.lock.Lock()
	for _, w := range(ips.watchers) {
		w.finished <- true
	}
	ips.lock.Unlock()
}
