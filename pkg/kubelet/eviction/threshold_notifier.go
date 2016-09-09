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

/*
#include <sys/eventfd.h>
*/
import "C"

import (
	"fmt"
	"syscall"
)

type ThresholdNotifier interface {
	Start()
	WaitForNotification()
	Close()
}

type memcgThresholdNotifier struct {
	watchfd   int
	controlfd int
	eventfd   int
	trigger   chan int
}

var _ ThresholdNotifier = &memcgThresholdNotifier{}

// NewMemCGThresholdNotifier sends notifications when a memory threshold
// is crossed (in either direction) for a given memory cgroup
func NewMemCGThresholdNotifier(cgPath, watchedAttr, threshold string) (ThresholdNotifier, error) {
	watchfd, err := syscall.Open(fmt.Sprintf("%s/memory.%s", cgPath, watchedAttr), syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	controlfd, err := syscall.Open(fmt.Sprintf("%s/cgroup.event_control", cgPath), syscall.O_WRONLY, 0)
	if err != nil {
		syscall.Close(watchfd)
		return nil, err
	}
	efd, err := C.eventfd(0, C.EFD_CLOEXEC)
	if err != nil {
		syscall.Close(watchfd)
		syscall.Close(controlfd)
		return nil, err
	}
	eventfd := int(efd)
	if eventfd < 0 {
		return nil, fmt.Errorf("eventfd call failed")
	}
	config := fmt.Sprintf("%d %d %s", eventfd, watchfd, threshold)
	_, err = syscall.Write(controlfd, []byte(config))
	if err != nil {
		syscall.Close(watchfd)
		syscall.Close(controlfd)
		syscall.Close(eventfd)
		return nil, err
	}
	return &memcgThresholdNotifier{
		watchfd:   watchfd,
		controlfd: controlfd,
		eventfd:   eventfd,
		trigger:   make(chan int, 1),
	}, nil
}

func (n *memcgThresholdNotifier) Start() {
	for {
		buf := make([]byte, 8)
		_, err := syscall.Read(n.eventfd, buf)
		if err != nil {
			return
		}
		n.trigger <- 0
	}
}

func (n *memcgThresholdNotifier) WaitForNotification() {
	<-n.trigger
}

func (n *memcgThresholdNotifier) Close() {
	syscall.Close(n.eventfd)
	syscall.Close(n.watchfd)
	syscall.Close(n.controlfd)
	close(n.trigger)
}

type noOpThresholdNotifier struct {
	trigger chan int
}

var _ ThresholdNotifier = &noOpThresholdNotifier{}

// NewNoOpThresholdNotifier never returns any notificaitons
func NewNoOpThresholdNotifier() ThresholdNotifier {
	return &noOpThresholdNotifier{
		trigger: make(chan int, 1),
	}
}

func (n *noOpThresholdNotifier) Start() {}

func (n *noOpThresholdNotifier) WaitForNotification() {
	// channel is never written to, blocks forever
	<-n.trigger
}

func (n *noOpThresholdNotifier) Close() {
	close(n.trigger)
}
