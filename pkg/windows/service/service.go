//go:build windows

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
	"syscall"
	"time"
	"unsafe"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/klog/v2"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
)

type handler struct {
	tosvc              chan bool
	fromsvc            chan error
	acceptPreshutdown  bool
	preshutdownHandler PreshutdownHandler
}

type PreshutdownHandler interface {
	ProcessShutdownEvent() error
}

// SERVICE_PRESHUTDOWN_INFO structure
type SERVICE_PRESHUTDOWN_INFO struct {
	PreshutdownTimeout uint32 // The time-out value, in milliseconds.
}

func QueryPreShutdownInfo(h windows.Handle) (*SERVICE_PRESHUTDOWN_INFO, error) {
	// Query the SERVICE_CONFIG_PRESHUTDOWN_INFO
	n := uint32(1024)
	b := make([]byte, n)
	for {
		err := windows.QueryServiceConfig2(h, windows.SERVICE_CONFIG_PRESHUTDOWN_INFO, &b[0], n, &n)
		if err == nil {
			break
		}
		if err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
			return nil, err
		}
		if n <= uint32(len(b)) {
			return nil, err
		}

		b = make([]byte, n)
	}

	// Convert the buffer to SERVICE_PRESHUTDOWN_INFO
	info := (*SERVICE_PRESHUTDOWN_INFO)(unsafe.Pointer(&b[0]))

	return info, nil
}

func UpdatePreShutdownInfo(h windows.Handle, timeoutMilliSeconds uint32) error {
	// Set preshutdown info
	preshutdownInfo := SERVICE_PRESHUTDOWN_INFO{
		PreshutdownTimeout: timeoutMilliSeconds,
	}

	err := windows.ChangeServiceConfig2(h, windows.SERVICE_CONFIG_PRESHUTDOWN_INFO, (*byte)(unsafe.Pointer(&preshutdownInfo)))
	if err != nil {
		return err
	}

	return nil
}

var thehandler *handler // This is, unfortunately, a global along with the service, which means only one service per process.

// InitService is the entry point for running the daemon as a Windows
// service. It returns an indication of whether it is running as a service;
// and an error.
func InitService(serviceName string) error {
	return initService(serviceName, false)
}

// InitService is the entry point for running the daemon as a Windows
// service which will accept preshutdown event. It returns an indication
// of whether it is running as a service; and an error.
func InitServiceWithShutdown(serviceName string) error {
	return initService(serviceName, true)
}

// initService will try to run the daemon as a Windows
// service, with an option to indicate if the service will accept the preshutdown event.
func initService(serviceName string, acceptPreshutdown bool) error {
	var err error
	h := &handler{
		tosvc:              make(chan bool),
		fromsvc:            make(chan error),
		acceptPreshutdown:  acceptPreshutdown,
		preshutdownHandler: nil,
	}

	thehandler = h

	go func() {
		err = svc.Run(serviceName, h)
		h.fromsvc <- err
	}()

	// Wait for the first signal from the service handler.
	err = <-h.fromsvc
	if err != nil {
		klog.Errorf("Running %s as a Windows has error %v!", serviceName, err)
		return err
	}
	klog.Infof("Running %s as a Windows service!", serviceName)
	return nil
}

func SetPreShutdownHandler(preshutdownhandler PreshutdownHandler) {
	thehandler.preshutdownHandler = preshutdownhandler
}

func IsServiceInitialized() bool {
	return thehandler != nil
}

func (h *handler) Execute(_ []string, r <-chan svc.ChangeRequest, s chan<- svc.Status) (bool, uint32) {
	s <- svc.Status{State: svc.StartPending, Accepts: 0}
	// Unblock initService()
	h.fromsvc <- nil

	if h.acceptPreshutdown {
		s <- svc.Status{State: svc.Running, Accepts: svc.AcceptStop | svc.AcceptPreShutdown | svc.Accepted(windows.SERVICE_ACCEPT_PARAMCHANGE)}
		klog.Infof("Accept preshutdown")
	} else {
		s <- svc.Status{State: svc.Running, Accepts: svc.AcceptStop | svc.AcceptShutdown | svc.Accepted(windows.SERVICE_ACCEPT_PARAMCHANGE)}
	}

	klog.Infof("Service running")
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
				klog.Infof("Service stopping")

				s <- svc.Status{State: svc.StopPending}
				// We need to translate this request into a signal that can be handled by the signal handler
				// handling shutdowns normally (currently apiserver/pkg/server/signal.go).
				// If we do not do this, our main threads won't be notified of the upcoming shutdown.
				// Since Windows services do not use any console, we cannot simply generate a CTRL_BREAK_EVENT
				// but need a dedicated notification mechanism.
				graceful := server.RequestShutdown()

				// Free up the control handler and let us terminate as gracefully as possible.
				// If that takes too long, the service controller will kill the remaining threads.
				// As per https://docs.microsoft.com/en-us/windows/desktop/services/service-control-handler-function
				s <- svc.Status{State: svc.StopPending}

				// If we cannot exit gracefully, we really only can exit our process, so at least the
				// service manager will think that we gracefully exited. At the time of writing this comment this is
				// needed for applications that do not use signals (e.g. kube-proxy)
				if !graceful {
					go func() {
						// Ensure the SCM was notified (The operation above (send to s) was received and communicated to the
						// service control manager - so it doesn't look like the service crashes)
						time.Sleep(1 * time.Second)
						os.Exit(0)
					}()
				}
				break Loop
			case svc.PreShutdown:
				klog.Infof("Node pre-shutdown")
				s <- svc.Status{State: svc.StopPending}

				if h.preshutdownHandler != nil {
					h.preshutdownHandler.ProcessShutdownEvent()
				}

				break Loop
			}
		}
	}

	return false, 0
}
