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
	"context"
	"errors"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/vmware/govmomi/toolbox/hgfs"
	"github.com/vmware/govmomi/toolbox/vix"
	"github.com/vmware/govmomi/vim25/types"
)

func TestDefaultIP(t *testing.T) {
	ip := DefaultIP()
	if ip == "" {
		t.Error("failed to get a default IP address")
	}
}

type testRPC struct {
	cmd    string
	expect string
}

type mockChannelIn struct {
	t       *testing.T
	service *Service
	rpc     []*testRPC
	wg      sync.WaitGroup
	start   error
	sendErr int
	count   struct {
		send  int
		stop  int
		start int
	}
}

func (c *mockChannelIn) Start() error {
	c.count.start++
	return c.start
}

func (c *mockChannelIn) Stop() error {
	c.count.stop++
	return nil
}

func (c *mockChannelIn) Receive() ([]byte, error) {
	if len(c.rpc) == 0 {
		if c.rpc != nil {
			// All test RPC requests have been consumed
			c.wg.Done()
			c.rpc = nil
		}
		return nil, io.EOF
	}

	return []byte(c.rpc[0].cmd), nil
}

func (c *mockChannelIn) Send(buf []byte) error {
	if c.sendErr > 0 {
		c.count.send++
		if c.count.send%c.sendErr == 0 {
			c.wg.Done()
			return errors.New("rpci send error")
		}
	}

	if buf == nil {
		return nil
	}

	expect := c.rpc[0].expect
	if string(buf) != expect {
		c.t.Errorf("expected %q reply for request %q, got: %q", expect, c.rpc[0].cmd, buf)
	}

	c.rpc = c.rpc[1:]

	return nil
}

// discard rpc out for now
type mockChannelOut struct {
	reply [][]byte
	start error
}

func (c *mockChannelOut) Start() error {
	return c.start
}

func (c *mockChannelOut) Stop() error {
	return nil
}

func (c *mockChannelOut) Receive() ([]byte, error) {
	if len(c.reply) == 0 {
		return nil, io.EOF
	}
	reply := c.reply[0]
	c.reply = c.reply[1:]
	return reply, nil
}

func (c *mockChannelOut) Send(buf []byte) error {
	if len(buf) == 0 {
		return io.ErrShortBuffer
	}
	return nil
}

func TestServiceRun(t *testing.T) {
	in := new(mockChannelIn)
	out := new(mockChannelOut)

	service := NewService(in, out)

	in.rpc = []*testRPC{
		{"reset", "OK ATR toolbox"},
		{"ping", "OK "},
		{"Set_Option synctime 0", "OK "},
		{"Capabilities_Register", "OK "},
		{"Set_Option broadcastIP 1", "OK "},
	}

	in.wg.Add(1)

	// replies to register capabilities
	for i := 0; i < len(capabilities); i++ {
		out.reply = append(out.reply, rpciOK)
	}

	out.reply = append(out.reply,
		rpciOK, // reply to SendGuestInfo call in Reset()
		rpciOK, // reply to IP broadcast
	)

	in.service = service

	in.t = t

	err := service.Start()
	if err != nil {
		t.Fatal(err)
	}

	in.wg.Wait()

	service.Stop()
	service.Wait()

	// verify we don't set delay > maxDelay
	for i := 0; i <= maxDelay+1; i++ {
		service.backoff()
	}

	if service.delay != maxDelay {
		t.Errorf("delay=%d", service.delay)
	}
}

func TestServiceErrors(t *testing.T) {
	Trace = true
	if !testing.Verbose() {
		// cover TraceChannel but discard output
		traceLog = ioutil.Discard
	}

	netInterfaceAddrs = func() ([]net.Addr, error) {
		return nil, io.EOF
	}

	in := new(mockChannelIn)
	out := new(mockChannelOut)

	service := NewService(in, out)

	service.RegisterHandler("Sorry", func([]byte) ([]byte, error) {
		return nil, errors.New("i am so sorry")
	})

	ip := ""
	service.PrimaryIP = func() string {
		if ip == "" {
			ip = "127"
		} else if ip == "127" {
			ip = "127.0.0.1"
		} else if ip == "127.0.0.1" {
			ip = ""
		}
		return ip
	}

	in.rpc = []*testRPC{
		{"Capabilities_Register", "OK "},
		{"Set_Option broadcastIP 1", "ERR "},
		{"Set_Option broadcastIP 1", "OK "},
		{"Set_Option broadcastIP 1", "OK "},
		{"NOPE", "Unknown Command"},
		{"Sorry", "ERR "},
	}

	in.wg.Add(1)

	// replies to register capabilities
	for i := 0; i < len(capabilities); i++ {
		out.reply = append(out.reply, rpciERR)
	}

	foo := []byte("foo")
	out.reply = append(
		out.reply,
		rpciERR,
		rpciOK,
		rpciOK,
		append(rpciOK, foo...),
		rpciERR,
	)

	in.service = service

	in.t = t

	err := service.Start()
	if err != nil {
		t.Fatal(err)
	}

	in.wg.Wait()

	// Done serving RPCs, test ChannelOut errors
	reply, err := service.out.Request(rpciOK)
	if err != nil {
		t.Error(err)
	}

	if !bytes.Equal(reply, foo) {
		t.Errorf("reply=%q", foo)
	}

	_, err = service.out.Request(rpciOK)
	if err == nil {
		t.Error("expected error")
	}

	_, err = service.out.Request(nil)
	if err == nil {
		t.Error("expected error")
	}

	service.Stop()
	service.Wait()

	// cover service start error paths
	start := errors.New("fail")

	in.start = start
	err = service.Start()
	if err != start {
		t.Error("expected error")
	}

	in.start = nil
	out.start = start
	err = service.Start()
	if err != start {
		t.Error("expected error")
	}
}

func TestServiceResetChannel(t *testing.T) {
	in := new(mockChannelIn)
	out := new(mockChannelOut)

	service := NewService(in, out)

	resetDelay = maxDelay

	fails := 2
	in.wg.Add(fails)
	in.sendErr = 10

	err := service.Start()
	if err != nil {
		t.Fatal(err)
	}

	in.wg.Wait()

	service.Stop()
	service.Wait()

	expect := fails
	if in.count.start != expect || in.count.stop != expect {
		t.Errorf("count=%#v", in.count)
	}
}

var (
	testESX = flag.Bool("toolbox.testesx", false, "Test toolbox service against ESX (vmtoolsd must not be running)")
	testPID = flag.Int64("toolbox.testpid", 0, "PID to return from toolbox start command")
	testOn  = flag.String("toolbox.powerState", "", "Power state of VM prior to starting the test")
)

// echoHandler for testing hgfs.FileHandler
type echoHandler struct{}

func (e *echoHandler) Stat(u *url.URL) (os.FileInfo, error) {
	if u.RawQuery == "" {
		return nil, errors.New("no query")
	}

	if u.Query().Get("foo") != "bar" {
		return nil, errors.New("invalid query")
	}

	return os.Stat(u.Path)
}

func (e *echoHandler) Open(u *url.URL, mode int32) (hgfs.File, error) {
	_, err := e.Stat(u)
	if err != nil {
		return nil, err
	}

	return os.Open(u.Path)
}

func TestServiceRunESX(t *testing.T) {
	if *testESX == false {
		t.SkipNow()
	}

	Trace = testing.Verbose()

	// A server that echos HTTP requests, for testing toolbox's http.RoundTripper
	echo := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Write(w)
	}))
	// Client side can use 'govc guest.getenv' to get the URL w/ random port
	_ = os.Setenv("TOOLBOX_ECHO_SERVER", echo.URL)

	var wg sync.WaitGroup

	in := NewBackdoorChannelIn()
	out := NewBackdoorChannelOut()

	service := NewService(in, out)
	service.Command.FileServer.RegisterFileHandler("echo", new(echoHandler))

	ping := sync.NewCond(new(sync.Mutex))

	service.RegisterHandler("ping", func(b []byte) ([]byte, error) {
		ping.Broadcast()
		return service.Ping(b)
	})

	// assert that reset, ping, Set_Option and Capabilities_Register are called at least once
	for _, name := range []string{"reset", "ping", "Set_Option", "Capabilities_Register"} {
		n := name
		h := service.handlers[name]
		wg.Add(1)

		service.handlers[name] = func(b []byte) ([]byte, error) {
			defer wg.Done()

			service.handlers[n] = h // reset

			return h(b)
		}
	}

	if *testOn == string(types.VirtualMachinePowerStatePoweredOff) {
		wg.Add(1)
		service.Power.PowerOn.Handler = func() error {
			defer wg.Done()
			log.Print("power on event")
			return nil
		}
	} else {
		log.Print("skipping power on test")
	}

	if *testPID != 0 {
		service.Command.ProcessStartCommand = func(m *ProcessManager, r *vix.StartProgramRequest) (int64, error) {
			wg.Add(1)
			defer wg.Done()

			switch r.ProgramPath {
			case "/bin/date":
				return *testPID, nil
			case "sleep":
				p := NewProcessFunc(func(ctx context.Context, arg string) error {
					d, err := time.ParseDuration(arg)
					if err != nil {
						return err
					}

					select {
					case <-ctx.Done():
						return &ProcessError{Err: ctx.Err(), ExitCode: 42}
					case <-time.After(d):
					}

					return nil
				})
				return m.Start(r, p)
			default:
				return DefaultStartCommand(m, r)
			}
		}
	}

	service.PrimaryIP = func() string {
		log.Print("broadcasting IP")
		return DefaultIP()
	}

	log.Print("starting toolbox service")
	err := service.Start()
	if err != nil {
		log.Fatal(err)
	}

	wg.Wait()

	// wait for 1 last ping to make sure the final response has reached the client before stopping
	ping.L.Lock()
	ping.Wait()
	ping.L.Unlock()

	service.Stop()
	service.Wait()
}
