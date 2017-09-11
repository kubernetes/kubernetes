/*
Copyright 2016 The Kubernetes Authors.

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

package eviction

import (
	"fmt"

	"github.com/golang/glog"
	"golang.org/x/sys/unix"
)

type memcgThresholdNotifier struct {
	watchfd     int
	controlfd   int
	eventfd     int
	handler     thresholdNotifierHandlerFunc
	description string
}

var _ ThresholdNotifier = &memcgThresholdNotifier{}

// NewMemCGThresholdNotifier sends notifications when a cgroup threshold
// is crossed (in either direction) for a given cgroup attribute
func NewMemCGThresholdNotifier(path, attribute, threshold, description string, handler thresholdNotifierHandlerFunc) (ThresholdNotifier, error) {
	watchfd, err := unix.Open(fmt.Sprintf("%s/%s", path, attribute), unix.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			unix.Close(watchfd)
		}
	}()
	controlfd, err := unix.Open(fmt.Sprintf("%s/cgroup.event_control", path), unix.O_WRONLY, 0)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			unix.Close(controlfd)
		}
	}()
	eventfd, err := unix.Eventfd(0, unix.EFD_CLOEXEC)
	if err != nil {
		return nil, err
	}
	if eventfd < 0 {
		err = fmt.Errorf("eventfd call failed")
		return nil, err
	}
	defer func() {
		if err != nil {
			unix.Close(eventfd)
		}
	}()
	glog.V(2).Infof("eviction: setting notification threshold to %s", threshold)
	config := fmt.Sprintf("%d %d %s", eventfd, watchfd, threshold)
	_, err = unix.Write(controlfd, []byte(config))
	if err != nil {
		return nil, err
	}
	return &memcgThresholdNotifier{
		watchfd:     watchfd,
		controlfd:   controlfd,
		eventfd:     eventfd,
		handler:     handler,
		description: description,
	}, nil
}

func getThresholdEvents(eventfd int, eventCh chan<- struct{}, stopCh <-chan struct{}) {
	for {
		buf := make([]byte, 8)
		_, err := unix.Read(eventfd, buf)
		if err != nil {
			return
		}

		select {
		case eventCh <- struct{}{}:
		case <-stopCh:
			return
		}
	}
}

func (n *memcgThresholdNotifier) Start(stopCh <-chan struct{}) {
	eventCh := make(chan struct{})
	go getThresholdEvents(n.eventfd, eventCh, stopCh)
	for {
		select {
		case <-stopCh:
			glog.V(2).Infof("eviction: stopping threshold notifier")
			unix.Close(n.watchfd)
			unix.Close(n.controlfd)
			unix.Close(n.eventfd)
			return
		case <-eventCh:
			glog.V(2).Infof("eviction: threshold crossed")
			n.handler(n.description)
		}
	}
}
