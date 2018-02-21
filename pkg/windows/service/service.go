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
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/golang/glog"

	"github.com/spf13/pflag"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

var (
	flServiceName        string
	flServiceDisplayName string
	flRegisterService    bool
	flUnregisterService  bool
	flRunService         bool

	service *handler
)

type handler struct {
	tosvc   chan bool
	fromsvc chan error
}

func getServicePath() (string, error) {
	p, err := exec.LookPath(os.Args[0])
	if err != nil {
		return "", err
	}
	return filepath.Abs(p)
}

func registerService() error {
	p, err := getServicePath()
	if err != nil {
		return err
	}
	m, err := mgr.ConnectRemote("")
	if err != nil {
		return err
	}
	defer m.Disconnect()

	c := mgr.Config{
		ServiceType:  windows.SERVICE_WIN32_OWN_PROCESS,
		StartType:    mgr.StartAutomatic,
		ErrorControl: mgr.ErrorNormal,
		DisplayName:  flServiceDisplayName,
	}

	// Configure the service to launch with the arguments that were just passed.
	args := []string{"--run-service"}
	skipNextArgument := false
	for _, a := range os.Args[1:] {
		if skipNextArgument {
			skipNextArgument = false
			continue
		}
		if a != "--register-service" && a != "--unregister-service" {
			if a == "--service-name" || a == "--service-displayname" {
				skipNextArgument = true
				continue
			} else {
				args = append(args, a)
			}
		}
	}

	s, err := m.CreateService(flServiceName, p, c, args...)
	if err != nil {
		return err
	}
	defer s.Close()

	// See http://stackoverflow.com/questions/35151052/how-do-i-configure-failure-actions-of-a-windows-service-written-in-go
	const (
		scActionNone       = 0
		scActionRestart    = 1
		scActionReboot     = 2
		scActionRunCommand = 3

		serviceConfigFailureActions = 2
	)

	type serviceFailureActions struct {
		ResetPeriod  uint32
		RebootMsg    *uint16
		Command      *uint16
		ActionsCount uint32
		Actions      uintptr
	}

	type scAction struct {
		Type  uint32
		Delay uint32
	}
	t := []scAction{
		{Type: scActionRestart, Delay: uint32(60 * time.Second / time.Millisecond)},
		{Type: scActionRestart, Delay: uint32(600 * time.Second / time.Millisecond)},
		{Type: scActionNone},
	}
	lpInfo := serviceFailureActions{ResetPeriod: uint32(24 * time.Hour / time.Second), ActionsCount: uint32(3), Actions: uintptr(unsafe.Pointer(&t[0]))}
	err = windows.ChangeServiceConfig2(s.Handle, serviceConfigFailureActions, (*byte)(unsafe.Pointer(&lpInfo)))
	if err == nil {
		glog.Infof("Service successfully registered!")
	}
	return err
}

func unregisterService() error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()

	s, err := m.OpenService(flServiceName)
	if err != nil {
		return err
	}
	defer s.Close()

	err = s.Delete()
	if err != nil {
		return err
	}
	glog.Infof("Service successfully unregistered!")
	return nil
}

// InitService is the entry point for running the daemon as a Windows
// service. It returns an indication to stop (if registering/un-registering);
// an indication of whether it is running as a service; and an error.
func InitService() (bool, bool, error) {
	if flUnregisterService {
		if flRegisterService {
			return true, false, fmt.Errorf("--register-service and --unregister-service cannot be used together")
		}
		return true, false, unregisterService()
	}

	if flRegisterService {
		return true, false, registerService()
	}

	if !flRunService {
		return false, false, nil
	}

	h := &handler{
		tosvc:   make(chan bool),
		fromsvc: make(chan error),
	}

	service = h
	var err error
	go func() {
		err = svc.Run(flServiceName, h)
		h.fromsvc <- err
	}()

	// Wait for the first signal from the service handler.
	err = <-h.fromsvc
	if err != nil {
		return false, false, err
	}
	return false, true, nil
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

// InstallServiceFlags adds additional flags for the Windows binary to
// enable the registration as a service
func InstallServiceFlags(flags *pflag.FlagSet, serviceName string, serviceDisplayName string) {
	flags.StringVar(&flServiceName, "service-name", serviceName, "Set the Windows service name")
	flags.StringVar(&flServiceDisplayName, "service-displayname", serviceDisplayName, "Set the Windows service display name")
	flags.BoolVar(&flRegisterService, "register-service", false, "Register the service and exit")
	flags.BoolVar(&flUnregisterService, "unregister-service", false, "Unregister the service and exit")
	flags.BoolVar(&flRunService, "run-service", false, "Start the service before exiting")
	flags.MarkHidden("run-service")
}
