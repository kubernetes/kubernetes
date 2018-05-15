// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package service

import (
	"os"

	"github.com/golang/glog"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
)

var (
	service *handler
)

type handler struct {
	tosvc   chan bool
	fromsvc chan error
}

// InitService is the entry point for running the daemon as a Windows
// service. It returns an indication of whether it is running as a service;
// and an error.
func InitService(serviceName string) error {
	h := &handler{
		tosvc:   make(chan bool),
		fromsvc: make(chan error),
	}

	service = h
	var err error
	go func() {
		err = svc.Run(serviceName, h)
		h.fromsvc <- err
	}()

	// Wait for the first signal from the service handler.
	err = <-h.fromsvc
	if err != nil {
		return err
	}
	glog.Infof("Running %s as a Windows service!", serviceName)
	return nil
}

func (h *handler) Execute(_ []string, r <-chan svc.ChangeRequest, s chan<- svc.Status) (bool, uint32) {
	s <- svc.Status{State: svc.StartPending, Accepts: 0}
	// Unblock initService()
	h.fromsvc <- nil

	s <- svc.Status{State: svc.Running, Accepts: svc.AcceptStop | svc.AcceptShutdown | svc.Accepted(windows.SERVICE_ACCEPT_PARAMCHANGE)}
	glog.Infof("Service running")
Loop:
	for {
		select {
		case <-h.tosvc:
			break Loop
		case c := <-r:
			switch c.Cmd {
			case svc.Cmd(windows.SERVICE_CONTROL_PARAMCHANGE):
				s <- c.CurrentStatus
			case svc.Interrogate:
				s <- c.CurrentStatus
			case svc.Stop, svc.Shutdown:
				s <- svc.Status{State: svc.Stopped}
				// TODO: Stop the kubelet gracefully instead of killing the process
				os.Exit(0)
			}
		}
	}

	return false, 0
}
