/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"bytes"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"github.com/vmware/govmomi/toolbox/hgfs"
)

const (
	// TOOLS_VERSION_UNMANAGED as defined in open-vm-tools/lib/include/vm_tools_version.h
	toolsVersionUnmanaged = 0x7fffffff

	// RPCIN_MAX_DELAY as defined in rpcChannelInt.h:
	maxDelay = 10
)

var (
	capabilities = []string{
		// Without tools.set.version, the UI reports Tools are "running", but "not installed"
		fmt.Sprintf("tools.set.version %d", toolsVersionUnmanaged),

		// Required to invoke guest power operations (shutdown, reboot)
		"tools.capability.statechange",

		"tools.capability.hgfs_server toolbox 1",
	}

	netInterfaceAddrs = net.InterfaceAddrs

	// If we have an RPCI send error, the channels will be reset.
	// open-vm-tools/lib/rpcChannel/rpcChannel.c:RpcChannelCheckReset also backs off in this case
	resetDelay = time.Duration(500) // 500 * 10ms == 5s
)

// Service receives and dispatches incoming RPC requests from the vmx
type Service struct {
	name     string
	in       Channel
	out      *ChannelOut
	handlers map[string]Handler
	stop     chan struct{}
	wg       *sync.WaitGroup
	delay    time.Duration
	rpcError bool

	Command *CommandServer
	Power   *PowerCommandHandler

	PrimaryIP func() string
}

// NewService initializes a Service instance
func NewService(rpcIn Channel, rpcOut Channel) *Service {
	s := &Service{
		name:     "toolbox", // Same name used by vmtoolsd
		in:       NewTraceChannel(rpcIn),
		out:      &ChannelOut{NewTraceChannel(rpcOut)},
		handlers: make(map[string]Handler),
		wg:       new(sync.WaitGroup),
		stop:     make(chan struct{}),

		PrimaryIP: DefaultIP,
	}

	s.RegisterHandler("reset", s.Reset)
	s.RegisterHandler("ping", s.Ping)
	s.RegisterHandler("Set_Option", s.SetOption)
	s.RegisterHandler("Capabilities_Register", s.CapabilitiesRegister)

	s.Command = registerCommandServer(s)
	s.Command.FileServer = hgfs.NewServer()
	s.Command.FileServer.RegisterFileHandler("proc", s.Command.ProcessManager)
	s.Command.FileServer.RegisterFileHandler(hgfs.ArchiveScheme, hgfs.NewArchiveHandler())

	s.Power = registerPowerCommandHandler(s)

	return s
}

// backoff exponentially increases the RPC poll delay up to maxDelay
func (s *Service) backoff() {
	if s.delay < maxDelay {
		if s.delay > 0 {
			d := s.delay * 2
			if d > s.delay && d < maxDelay {
				s.delay = d
			} else {
				s.delay = maxDelay
			}
		} else {
			s.delay = 1
		}
	}
}

func (s *Service) stopChannel() {
	_ = s.in.Stop()
	_ = s.out.Stop()
}

func (s *Service) startChannel() error {
	err := s.in.Start()
	if err != nil {
		return err
	}

	return s.out.Start()
}

func (s *Service) checkReset() error {
	if s.rpcError {
		s.stopChannel()
		err := s.startChannel()
		if err != nil {
			s.delay = resetDelay
			return err
		}
		s.rpcError = false
	}

	return nil
}

// Start initializes the RPC channels and starts a goroutine to listen for incoming RPC requests
func (s *Service) Start() error {
	err := s.startChannel()
	if err != nil {
		return err
	}

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()

		// Same polling interval and backoff logic as vmtoolsd.
		// Required in our case at startup at least, otherwise it is possible
		// we miss the 1 Capabilities_Register call for example.

		// Note we Send(response) even when nil, to let the VMX know we are here
		var response []byte

		for {
			select {
			case <-s.stop:
				s.stopChannel()
				return
			case <-time.After(time.Millisecond * 10 * s.delay):
				if err = s.checkReset(); err != nil {
					continue
				}

				err = s.in.Send(response)
				response = nil
				if err != nil {
					s.delay = resetDelay
					s.rpcError = true
					continue
				}

				request, _ := s.in.Receive()

				if len(request) > 0 {
					response = s.Dispatch(request)

					s.delay = 0
				} else {
					s.backoff()
				}
			}
		}
	}()

	return nil
}

// Stop cancels the RPC listener routine created via Start
func (s *Service) Stop() {
	close(s.stop)
}

// Wait blocks until Start returns, allowing any current RPC in progress to complete.
func (s *Service) Wait() {
	s.wg.Wait()
}

// Handler is given the raw argument portion of an RPC request and returns a response
type Handler func([]byte) ([]byte, error)

// RegisterHandler for the given RPC name
func (s *Service) RegisterHandler(name string, handler Handler) {
	s.handlers[name] = handler
}

// Dispatch an incoming RPC request to a Handler
func (s *Service) Dispatch(request []byte) []byte {
	msg := bytes.SplitN(request, []byte{' '}, 2)
	name := msg[0]

	// Trim NULL byte terminator
	name = bytes.TrimRight(name, "\x00")

	handler, ok := s.handlers[string(name)]

	if !ok {
		log.Printf("unknown command: %q\n", name)
		return []byte("Unknown Command")
	}

	var args []byte
	if len(msg) == 2 {
		args = msg[1]
	}

	response, err := handler(args)
	if err == nil {
		response = append([]byte("OK "), response...)
	} else {
		log.Printf("error calling %s: %s\n", name, err)
		response = append([]byte("ERR "), response...)
	}

	return response
}

// Reset is the default Handler for reset requests
func (s *Service) Reset([]byte) ([]byte, error) {
	s.SendGuestInfo() // Send the IP info ASAP

	return []byte("ATR " + s.name), nil
}

// Ping is the default Handler for ping requests
func (s *Service) Ping([]byte) ([]byte, error) {
	return nil, nil
}

// SetOption is the default Handler for Set_Option requests
func (s *Service) SetOption(args []byte) ([]byte, error) {
	opts := bytes.SplitN(args, []byte{' '}, 2)
	key := string(opts[0])
	val := string(opts[1])

	if Trace {
		fmt.Fprintf(os.Stderr, "set option %q=%q\n", key, val)
	}

	switch key {
	case "broadcastIP": // TODO: const-ify
		if val == "1" {
			ip := s.PrimaryIP()
			if ip == "" {
				log.Printf("failed to find primary IP")
				return nil, nil
			}
			msg := fmt.Sprintf("info-set guestinfo.ip %s", ip)
			_, err := s.out.Request([]byte(msg))
			if err != nil {
				return nil, err
			}

			s.SendGuestInfo()
		}
	default:
		// TODO: handle other options...
	}

	return nil, nil
}

// DefaultIP is used by default when responding to a Set_Option broadcastIP request
// It can be overridden with the Service.PrimaryIP field
func DefaultIP() string {
	addrs, err := netInterfaceAddrs()
	if err == nil {
		for _, addr := range addrs {
			if ip, ok := addr.(*net.IPNet); ok && !ip.IP.IsLoopback() {
				if ip.IP.To4() != nil {
					return ip.IP.String()
				}
			}
		}
	}

	return ""
}

func (s *Service) CapabilitiesRegister([]byte) ([]byte, error) {
	for _, cap := range capabilities {
		_, err := s.out.Request([]byte(cap))
		if err != nil {
			log.Printf("send %q: %s", cap, err)
		}
	}

	return nil, nil
}

func (s *Service) SendGuestInfo() {
	info := []func() ([]byte, error){
		GuestInfoNicInfoRequest,
	}

	for i, r := range info {
		b, err := r()

		if err == nil {
			_, err = s.out.Request(b)
		}

		if err != nil {
			log.Printf("SendGuestInfo %d: %s", i, err)
		}
	}
}
