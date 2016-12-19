// +build cgo

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

package main

/*
#include <sys/eventfd.h>
*/
import "C"

import (
	"flag"
	"fmt"
	"runtime/debug"
	"syscall"
	"time"
)

// thresholdNotifierHandlerFunc is a function that takes action in response to a crossed threshold
type thresholdNotifierHandlerFunc func(thresholdDescription string)

// ThresholdNotifier notifies the user when an attribute crosses a threshold value
type ThresholdNotifier interface {
	Start(stopCh <-chan struct{})
}

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
	watchfd, err := syscall.Open(fmt.Sprintf("%s/%s", path, attribute), syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			syscall.Close(watchfd)
		}
	}()
	controlfd, err := syscall.Open(fmt.Sprintf("%s/cgroup.event_control", path), syscall.O_WRONLY, 0)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			syscall.Close(controlfd)
		}
	}()
	efd, err := C.eventfd(0, C.EFD_CLOEXEC)
	if err != nil {
		return nil, err
	}
	eventfd := int(efd)
	if eventfd < 0 {
		err = fmt.Errorf("eventfd call failed")
		return nil, err
	}
	defer func() {
		if err != nil {
			syscall.Close(eventfd)
		}
	}()
	fmt.Println("eviction: setting notification threshold to %s", threshold)
	config := fmt.Sprintf("%d %d %s", eventfd, watchfd, threshold)
	_, err = syscall.Write(controlfd, []byte(config))
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

func getThresholdEvents(eventfd int, eventCh chan<- int) {
	for {
		buf := make([]byte, 8)
		_, err := syscall.Read(eventfd, buf)
		if err != nil {
			return
		}
		eventCh <- 0
	}
}

func (n *memcgThresholdNotifier) Start(stopCh <-chan struct{}) {
	eventCh := make(chan int, 1)
	go getThresholdEvents(n.eventfd, eventCh)
	for {
		select {
		case <-stopCh:
			fmt.Println("eviction: stopping threshold notifier")
			syscall.Close(n.watchfd)
			syscall.Close(n.controlfd)
			syscall.Close(n.eventfd)
			close(eventCh)
			return
		case <-eventCh:
			n.handler(n.description)
		}
	}
}

var (
	argUsageThreshold = flag.Int("usageThreshold", 1000000000, "threshold to be triggered")
)

func main() {
	flag.Parse()
	cgpath := "/sys/fs/cgroup/memory"
	attribute := "memory.usage_in_bytes"
	description := fmt.Sprintf("%s<%d", attribute, *argUsageThreshold)
	memSleepDuration := 100 * time.Millisecond
	stepSize := 500000000

	memcgThresholdNotifier, err := NewMemCGThresholdNotifier(cgpath, attribute, fmt.Sprintf("%d", *argUsageThreshold), description, func(thresholdDescription string) {
		fmt.Printf("Threshold met: %v\n", thresholdDescription)
	})
	if err != nil {
		fmt.Printf("Error during NewMemGCThresholdNotifier: %v\n", err)
	}
	go memcgThresholdNotifier.Start(make(chan struct{}))
	for {
		newBuffer := make([]byte, stepSize)
		for i := range newBuffer {
			newBuffer[i] = 0
		}
		fmt.Printf("Allocated memory... \n")
		newBuffer = nil
		time.Sleep(memSleepDuration)
		debug.FreeOSMemory()
	}
}
