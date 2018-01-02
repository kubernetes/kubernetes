// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/daemon"
	rktlog "github.com/coreos/rkt/pkg/log"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"
	"github.com/kr/pty"
)

var (
	log     *rktlog.Logger
	diag    *rktlog.Logger
	action  string
	appName string
	debug   bool
)

const (
	// iottymux store several bits of information for a
	// specific instance under /rkt/iottymux/<app>/
	pathPrefix = "/rkt/iottymux"

	// curren version of JSON API (for `list`)
	apiVersion = 1
)

func init() {
	// debug flag is not here, as it comes from env instead of CLI
	flag.StringVar(&action, "action", "list", "Sub-action to perform")
	flag.StringVar(&appName, "app", "", "Target application name")
}

// Endpoint represents a single attachable endpoint for an application
type Endpoint struct {
	// Name, freeform (eg. stdin, tty, etc.)
	Name string `json:"name"`
	// Domain, golang compatible (eg. tcp4, unix, etc.)
	Domain string `json:"domain"`
	// Address, golang compatible (eg. 127.0.0.1:3333, /tmp/file.sock, etc.)
	Address string `json:"address"`
}

// Targets references all attachable endpoints, for status persistence
type Targets struct {
	Version int        `json:"version"`
	Targets []Endpoint `json:"targets"`
}

// iottymux is a multi-action binary which can be used for:
//  * creating and muxing a TTY for an application
//  * proxying streams for an application (stdin/stdout/stderr) over TCP
//  * listing available attachable endpoints for an application
func main() {
	var err error
	// Parse flag and initialize logging
	flag.Parse()
	if os.Getenv("STAGE1_DEBUG") == "true" {
		debug = true
	}
	stage1initcommon.InitDebug(debug)
	log, diag, _ = rktlog.NewLogSet("iottymux", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// validate app name
	_, err = types.NewACName(appName)
	if err != nil {
		log.Printf("invalid app name (%s): %v", appName, err)
		os.Exit(254)
	}

	var r error
	statusFile := filepath.Join(pathPrefix, appName, "endpoints")

	// TODO(lucab): split this some more. Mux is part of pod service,
	// while other actions are called from stage0. Those should be split.
	switch action {
	// attaching
	case "auto-attach":
		r = actionAttach(statusFile, true)
	case "custom-attach":
		r = actionAttach(statusFile, false)

	// muxing and proxying
	case "iomux":
		r = actionIOMux(statusFile)
	case "ttymux":
		r = actionTTYMux(statusFile)

	// listing
	case "list":
		r = actionPrint(statusFile, os.Stdout)

	default:
		r = fmt.Errorf("unknown action %q", action)
	}

	if r != nil && r != io.EOF {
		log.FatalE("runtime failure", r)
	}
	os.Exit(0)
}

// actionAttach handles the attach action, either in "automatic" or "custom endpoints" mode.
func actionAttach(statusPath string, autoMode bool) error {
	var endpoints Targets
	dialTimeout := 15 * time.Second

	// retrieve available endpoints
	statusFile, err := os.OpenFile(statusPath, os.O_RDONLY, os.ModePerm)
	if err != nil {
		return err
	}
	err = json.NewDecoder(statusFile).Decode(&endpoints)
	_ = statusFile.Close()
	if err != nil {
		return err
	}

	// retrieve custom attaching modes
	customTargets := struct {
		ttyIn  bool
		ttyOut bool
		stdin  bool
		stdout bool
		stderr bool
	}{}
	if !autoMode {
		customTargets.ttyIn, _ = strconv.ParseBool(os.Getenv("STAGE2_ATTACH_TTYIN"))
		customTargets.ttyOut, _ = strconv.ParseBool(os.Getenv("STAGE2_ATTACH_TTYOUT"))
		customTargets.stdin, _ = strconv.ParseBool(os.Getenv("STAGE2_ATTACH_STDIN"))
		customTargets.stdout, _ = strconv.ParseBool(os.Getenv("STAGE2_ATTACH_STDOUT"))
		customTargets.stderr, _ = strconv.ParseBool(os.Getenv("STAGE2_ATTACH_STDERR"))
	}

	// Proxy I/O between this process and the iottymux service:
	//  - input (stdin, tty-in) copying routines can only be canceled by process killing (ie. user detaching)
	//  - output (stdout, stderr, tty-out) copying routines are canceled by errors when reading from remote service
	c := make(chan error)
	copyOut := func(w io.Writer, conn net.Conn) {
		_, err := io.Copy(w, conn)
		c <- err
	}

	for _, ep := range endpoints.Targets {
		d := net.Dialer{Timeout: dialTimeout}
		conn, err := d.Dial(ep.Domain, ep.Address)
		if err != nil {
			return err
		}
		defer conn.Close()
		switch ep.Name {
		case "stdin":
			if autoMode || customTargets.stdin {
				go io.Copy(conn, os.Stdin)
			}
		case "stdout":
			if autoMode || customTargets.stdout {
				go copyOut(os.Stdout, conn)
			}
		case "stderr":
			if autoMode || customTargets.stderr {
				go copyOut(os.Stderr, conn)
			}
		case "tty":
			if autoMode || customTargets.ttyIn {
				go io.Copy(conn, os.Stdin)
			}

			if autoMode || customTargets.ttyOut {
				go copyOut(os.Stdout, conn)
			} else {
				go copyOut(ioutil.Discard, conn)
			}
		}
	}

	// as soon as one output copying routine fails, this unblocks and the whole process exits
	return <-c
}

// actionPrint prints out available endpoints by unmarshalling the Targets struct
// from JSON at the given path. This is used by external tools to see which attaching
// modes are available for an application (eg. `rkt attach --mode=list`)
func actionPrint(path string, out io.Writer) error {
	var endpoints Targets
	statusFile, err := os.OpenFile(path, os.O_RDONLY, os.ModePerm)
	if err != nil {
		return err
	}

	err = json.NewDecoder(statusFile).Decode(&endpoints)
	_ = statusFile.Close()
	if err != nil {
		return err
	}

	// TODO(lucab): move to encoder.SetIndent (golang >= 1.7)
	status, err := json.MarshalIndent(endpoints, "", "    ")
	if err != nil {
		return nil
	}
	_, err = out.Write(status)
	return err
}

// actionTTYMux handles TTY muxing and proxying.
// It creates a PTY pair and bind-mounts the slave to `/rkt/iottymux/<app>/stage2-pts`.
// Once ready, it sd-notifies as READY so that the main application can be started.
func actionTTYMux(statusFile string) error {
	// Open a new TTY pair (master/slave)
	ptm, pts, err := pty.Open()
	if err != nil {
		return err
	}
	ttySlavePath := pts.Name()
	_ = pts.Close()
	defer ptm.Close()
	diag.Printf("TTY created, slave pty at %q\n", ttySlavePath)

	// TODO(lucab): set a sane TTY mode here (echo, controls and such).

	// Slave TTY has a dynamic name (eg. /dev/pts/<n>) but a predictable name
	// is needed, in order to be used as `TTYPath=` value in application unit.
	// A bind mount is put in place for that, here.
	ttypath := filepath.Join(pathPrefix, appName, "stage2-pts")
	f, err := os.Create(ttypath)
	if err != nil {
		return err
	}
	err = syscall.Mount(pts.Name(), ttypath, "", syscall.MS_BIND, "")
	if err != nil {
		return err
	}
	// TODO(lucab): double-check this is fine here (alternatives: app-stop or app-rm)
	defer syscall.Unmount(ttypath, 0)
	defer f.Close()

	// TODO(lucab): investigate sending fd to systemd-manager to ensure we never close
	// the PTY master fd. Open questions: dupfd and ownership.

	// signal to systemd that the PTY is ready and application can start.
	// sd-notify is required here, so a non-delivered status is an hard failure.
	ok, err := daemon.SdNotify(true, "READY=1")
	if !ok {
		return fmt.Errorf("failure during startup notification: %v", err)
	}
	diag.Print("TTY handler ready\n")

	// Open sockets
	ep := Endpoint{
		Name:    "tty",
		Domain:  "unix",
		Address: filepath.Join(pathPrefix, appName, "sock-tty"),
	}
	endpoints := Targets{apiVersion, []Endpoint{ep}}
	listener, err := net.Listen(ep.Domain, ep.Address)
	if err != nil {
		return fmt.Errorf("unable to open tty listener: %s", err)
	}
	defer listener.Close()
	diag.Printf("Listening for TTY on %s\n", ep.Address)

	// write available endpoints to status file
	sf, err := os.OpenFile(statusFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.ModePerm)
	if err != nil {
		return err
	}
	err = json.NewEncoder(sf).Encode(endpoints)
	_ = sf.Close()
	if err != nil {
		return err
	}

	// Proxy between TTY and remote clients.
	c := make(chan error)
	clients := make(chan net.Conn)
	go acceptConn(listener, clients, "tty")
	go proxyIO(clients, ptm, c)

	dispatchSig(c)

	// If nothing else fails, ttymux service will be waiting here forever
	// and be terminated by systemd only when the main application exits.
	return <-c
}

// actionIOMux handles I/O streams muxing and proxying (stdin/stdout/stderr)
func actionIOMux(statusFile string) error {
	// Slice containing mapping for "fdnum -> stream -> fifo -> socket":
	//  0 -> stdin  -> /rkt/iottymux/<app>/stage2-stdin  -> /rkt/iottymux/<app>/sock-stdin
	//  1 -> stdout -> /rkt/iottymux/<app>/stage2-stdout -> /rkt/iottymux/<app>/sock-stdout
	//  2 -> stderr -> /rkt/iottymux/<app>/stage2-stderr -> /rkt/iottymux/<app>/sock-stderr
	streams := [3]struct {
		listener net.Listener
		fifo     *os.File
	}{}

	// open FIFOs and create sockets
	streamsSetup := [3]struct {
		streamName    string
		isEnabled     bool
		fifoPath      string
		fifoOpenFlags int
		socketDomain  string
		socketAddress string
	}{
		{
			"stdin",
			false,
			filepath.Join(pathPrefix, appName, "stage2-stdin"),
			os.O_WRONLY,
			"unix",
			filepath.Join(pathPrefix, appName, "sock-stdin"),
		},
		{
			"stdout",
			false,
			filepath.Join(pathPrefix, appName, "stage2-stdout"),
			os.O_RDONLY,
			"unix",
			filepath.Join(pathPrefix, appName, "sock-stdout"),
		},
		{
			"stderr",
			false,
			filepath.Join(pathPrefix, appName, "stage2-stderr"),
			os.O_RDONLY,
			"unix",
			filepath.Join(pathPrefix, appName, "sock-stderr"),
		},
	}
	for i, f := range [3]string{"STAGE2_STDIN", "STAGE2_STDOUT", "STAGE2_STDERR"} {
		streamsSetup[i].isEnabled, _ = strconv.ParseBool(os.Getenv(f))
	}

	var endpoints Targets
	endpoints.Version = 1
	for i, entry := range streamsSetup {
		if streamsSetup[i].isEnabled {
			var err error
			ep := Endpoint{
				Name:    entry.streamName,
				Domain:  entry.socketDomain,
				Address: entry.socketAddress,
			}
			streams[i].fifo, err = os.OpenFile(entry.fifoPath, entry.fifoOpenFlags, os.ModeNamedPipe)
			if err != nil {
				return fmt.Errorf("invalid %s FIFO: %s", entry.streamName, err)
			}
			defer streams[i].fifo.Close()
			streams[i].listener, err = net.Listen(ep.Domain, ep.Address)
			if err != nil {
				return fmt.Errorf("unable to open %s listener: %s", entry.streamName, err)
			}
			defer streams[i].listener.Close()
			endpoints.Targets = append(endpoints.Targets, ep)
			diag.Printf("Listening for %s on %s\n", entry.streamName, ep.Address)
		}
	}

	// write available endpoints to status file
	sf, err := os.OpenFile(statusFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.ModePerm)
	if err != nil {
		return err
	}
	err = json.NewEncoder(sf).Encode(endpoints)
	_ = sf.Close()
	if err != nil {
		return err
	}

	c := make(chan error)

	// TODO(lucab): finalize custom logging modes
	logMode := os.Getenv("STAGE1_LOGMODE")
	var logFile *os.File
	switch logMode {
	case "k8s-plain":
		var err error
		// TODO(lucab): check what should be the target path with Euan
		logTarget := filepath.Join(pathPrefix, appName, "logfile")
		logFile, err = os.OpenFile(logTarget, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.ModePerm)
		if err != nil {
			return err
		}
		defer logFile.Close()
	}

	// proxy stdin
	if streams[0].fifo != nil && streams[0].listener != nil {
		clients := make(chan net.Conn)
		go acceptConn(streams[0].listener, clients, "stdin")
		go muxInput(clients, streams[0].fifo)
	}

	// proxy stdout
	if streams[1].fifo != nil && streams[1].listener != nil {
		localTargets := make(chan io.WriteCloser)
		clients := make(chan net.Conn)
		lines := make(chan []byte)
		go bufferLine(streams[1].fifo, lines, c)
		go acceptConn(streams[1].listener, clients, "stdout")
		go muxOutput("stdout", lines, clients, localTargets)
		if logFile != nil {
			localTargets <- logFile
		}
	}

	// proxy stderr
	if streams[2].fifo != nil && streams[2].listener != nil {
		localTargets := make(chan io.WriteCloser)
		clients := make(chan net.Conn)
		lines := make(chan []byte)
		go bufferLine(streams[2].fifo, lines, c)
		go acceptConn(streams[2].listener, clients, "stderr")
		go muxOutput("stderr", lines, clients, localTargets)
		if logFile != nil {
			localTargets <- logFile
		}
	}

	dispatchSig(c)

	// If nothing else fails, iomux service will be waiting here forever
	// and be terminated by systemd only when the main application exits.
	return <-c
}

// dispatchSig launches a goroutine and closes the given stop channel
// when SIGTERM, SIGHUP, or SIGINT is received.
func dispatchSig(stop chan<- error) {
	sigChan := make(chan os.Signal)
	signal.Notify(
		sigChan,
		syscall.SIGTERM,
		syscall.SIGHUP,
		syscall.SIGINT,
	)

	go func() {
		diag.Println("Waiting for signal")
		sig := <-sigChan
		diag.Printf("Received signal %v\n", sig)
		close(stop)
	}()
}

// bufferLine buffers and queues a single line from a Reader to a multiplexer
// If reading from src fails, it hard-fails and propagates the error back.
func bufferLine(src io.Reader, c chan<- []byte, ec chan<- error) {
	rd := bufio.NewReader(src)
	for {
		lineOut, err := rd.ReadBytes('\n')
		if len(lineOut) > 0 {
			c <- lineOut
		}
		if err != nil {
			ec <- err
		}
	}
}

// acceptConn accepts a single client and queues it for further proxying
// It is never canceled explicitly, as it is bound to the lifetime of the main process.
func acceptConn(socket net.Listener, c chan<- net.Conn, stream string) {
	for {
		conn, err := socket.Accept()
		if err == nil {
			diag.Printf("Accepted new connection for %s\n", stream)
			c <- conn
		}
	}
}

// proxyIO performs bi-directional byte-by-byte forwarding
// TODO(lucab): this may become line-buffered and muxed to logs
// TODO(lucab): reset terminal state on new attach
func proxyIO(clients <-chan net.Conn, tty *os.File, ttyFailure chan<- error) {
	ec := make(chan error)

	// ttyToRemote copies output from application TTY to remote client.
	// If copier reaches TTY EOF, it hard-fails and propagates the error up.
	ttyToRemote := func(dst net.Conn, src *os.File) {
		_, err := io.Copy(dst, src)
		if err == nil {
			_ = dst.Close()
			close(ec)
		}
	}

	// remoteToTTY copies input from remote client to application TTY.
	// When copying stops/fails, it recovers and just closes this connection.
	remoteToTTY := func(dst *os.File, src net.Conn) {
		io.Copy(dst, src)
		src.Close()
	}

	for {
		select {
		// a new remote client
		case cl := <-clients:
			go ttyToRemote(cl, tty)
			go remoteToTTY(tty, cl)

		// a TTY failure from one of the copier
		case tf := <-ec:
			ttyFailure <- tf
			return
		}
	}
}

// muxInput accepts remote clients and multiplex input line from them
func muxInput(clients <-chan net.Conn, stdin *os.File) {
	for {
		select {
		case c := <-clients:
			go bufferInput(c, stdin)
		}
	}
}

// bufferInput buffers and write a single line from a remote client to the local app
func bufferInput(conn net.Conn, stdin *os.File) {
	rd := bufio.NewReader(conn)
	defer conn.Close()
	for {
		lineIn, err := rd.ReadBytes('\n')
		if len(lineIn) == 0 && err != nil {
			return
		}
		_, err = stdin.Write(lineIn)
		if err != nil {
			return
		}
	}
}

// muxOutput receives remote clients and local log targets,
// multiplexing output lines to them
func muxOutput(streamLabel string, lines chan []byte, clients <-chan net.Conn, targets <-chan io.WriteCloser) {
	var logs []io.WriteCloser
	var conns []io.WriteCloser

	writeAndFilter := func(wc io.WriteCloser, line []byte) bool {
		_, err := wc.Write(line)
		if err != nil {
			wc.Close()
		}
		return err != nil
	}

	logsWriteAndFilter := func(wc io.WriteCloser, line []byte) bool {
		out := []byte(fmt.Sprintf("%s %s %s", time.Now().Format(time.RFC3339Nano), streamLabel, line))
		return writeAndFilter(wc, out)
	}

	for {
		select {
		// an incoming output line to multiplex
		// TODO(lucab): ordered non-blocking writes
		case l := <-lines:
			conns = filterTargets(conns, l, writeAndFilter)
			logs = filterTargets(logs, l, logsWriteAndFilter)

		// a new remote client
		case c := <-clients:
			conns = append(conns, c)

		// a new local log target
		case t := <-targets:
			logs = append(logs, t)
		}
	}
}

// filterTargets passes line to each writer in wcs,
// filtering out single writers if filter returns true.
func filterTargets(
	wcs []io.WriteCloser,
	line []byte,
	filter func(io.WriteCloser, []byte) bool,
) []io.WriteCloser {
	var filteredTargets []io.WriteCloser

	for _, c := range wcs {
		if !filter(c, line) {
			filteredTargets = append(filteredTargets, c)
		}
	}
	return filteredTargets
}
