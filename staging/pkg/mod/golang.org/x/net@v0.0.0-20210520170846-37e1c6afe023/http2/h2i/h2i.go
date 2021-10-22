// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows

/*
The h2i command is an interactive HTTP/2 console.

Usage:
  $ h2i [flags] <hostname>

Interactive commands in the console: (all parts case-insensitive)

  ping [data]
  settings ack
  settings FOO=n BAR=z
  headers      (open a new stream by typing HTTP/1.1)
*/
package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"golang.org/x/term"
)

// Flags
var (
	flagNextProto = flag.String("nextproto", "h2,h2-14", "Comma-separated list of NPN/ALPN protocol names to negotiate.")
	flagInsecure  = flag.Bool("insecure", false, "Whether to skip TLS cert validation")
	flagSettings  = flag.String("settings", "empty", "comma-separated list of KEY=value settings for the initial SETTINGS frame. The magic value 'empty' sends an empty initial settings frame, and the magic value 'omit' causes no initial settings frame to be sent.")
	flagDial      = flag.String("dial", "", "optional ip:port to dial, to connect to a host:port but use a different SNI name (including a SNI name without DNS)")
)

type command struct {
	run func(*h2i, []string) error // required

	// complete optionally specifies tokens (case-insensitive) which are
	// valid for this subcommand.
	complete func() []string
}

var commands = map[string]command{
	"ping": {run: (*h2i).cmdPing},
	"settings": {
		run: (*h2i).cmdSettings,
		complete: func() []string {
			return []string{
				"ACK",
				http2.SettingHeaderTableSize.String(),
				http2.SettingEnablePush.String(),
				http2.SettingMaxConcurrentStreams.String(),
				http2.SettingInitialWindowSize.String(),
				http2.SettingMaxFrameSize.String(),
				http2.SettingMaxHeaderListSize.String(),
			}
		},
	},
	"quit":    {run: (*h2i).cmdQuit},
	"headers": {run: (*h2i).cmdHeaders},
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: h2i <hostname>\n\n")
	flag.PrintDefaults()
}

// withPort adds ":443" if another port isn't already present.
func withPort(host string) string {
	if _, _, err := net.SplitHostPort(host); err != nil {
		return net.JoinHostPort(host, "443")
	}
	return host
}

// withoutPort strips the port from addr if present.
func withoutPort(addr string) string {
	if h, _, err := net.SplitHostPort(addr); err == nil {
		return h
	}
	return addr
}

// h2i is the app's state.
type h2i struct {
	host   string
	tc     *tls.Conn
	framer *http2.Framer
	term   *term.Terminal

	// owned by the command loop:
	streamID uint32
	hbuf     bytes.Buffer
	henc     *hpack.Encoder

	// owned by the readFrames loop:
	peerSetting map[http2.SettingID]uint32
	hdec        *hpack.Decoder
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
		os.Exit(2)
	}
	log.SetFlags(0)

	host := flag.Arg(0)
	app := &h2i{
		host:        host,
		peerSetting: make(map[http2.SettingID]uint32),
	}
	app.henc = hpack.NewEncoder(&app.hbuf)

	if err := app.Main(); err != nil {
		if app.term != nil {
			app.logf("%v\n", err)
		} else {
			fmt.Fprintf(os.Stderr, "%v\n", err)
		}
		os.Exit(1)
	}
	fmt.Fprintf(os.Stdout, "\n")
}

func (app *h2i) Main() error {
	cfg := &tls.Config{
		ServerName:         withoutPort(app.host),
		NextProtos:         strings.Split(*flagNextProto, ","),
		InsecureSkipVerify: *flagInsecure,
	}

	hostAndPort := *flagDial
	if hostAndPort == "" {
		hostAndPort = withPort(app.host)
	}
	log.Printf("Connecting to %s ...", hostAndPort)
	tc, err := tls.Dial("tcp", hostAndPort, cfg)
	if err != nil {
		return fmt.Errorf("Error dialing %s: %v", hostAndPort, err)
	}
	log.Printf("Connected to %v", tc.RemoteAddr())
	defer tc.Close()

	if err := tc.Handshake(); err != nil {
		return fmt.Errorf("TLS handshake: %v", err)
	}
	if !*flagInsecure {
		if err := tc.VerifyHostname(app.host); err != nil {
			return fmt.Errorf("VerifyHostname: %v", err)
		}
	}
	state := tc.ConnectionState()
	log.Printf("Negotiated protocol %q", state.NegotiatedProtocol)
	if !state.NegotiatedProtocolIsMutual || state.NegotiatedProtocol == "" {
		return fmt.Errorf("Could not negotiate protocol mutually")
	}

	if _, err := io.WriteString(tc, http2.ClientPreface); err != nil {
		return err
	}

	app.framer = http2.NewFramer(tc, tc)

	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return err
	}
	defer term.Restore(0, oldState)

	var screen = struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout}

	app.term = term.NewTerminal(screen, "h2i> ")
	lastWord := regexp.MustCompile(`.+\W(\w+)$`)
	app.term.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		if key != '\t' {
			return
		}
		if pos != len(line) {
			// TODO: we're being lazy for now, only supporting tab completion at the end.
			return
		}
		// Auto-complete for the command itself.
		if !strings.Contains(line, " ") {
			var name string
			name, _, ok = lookupCommand(line)
			if !ok {
				return
			}
			return name, len(name), true
		}
		_, c, ok := lookupCommand(line[:strings.IndexByte(line, ' ')])
		if !ok || c.complete == nil {
			return
		}
		if strings.HasSuffix(line, " ") {
			app.logf("%s", strings.Join(c.complete(), " "))
			return line, pos, true
		}
		m := lastWord.FindStringSubmatch(line)
		if m == nil {
			return line, len(line), true
		}
		soFar := m[1]
		var match []string
		for _, cand := range c.complete() {
			if len(soFar) > len(cand) || !strings.EqualFold(cand[:len(soFar)], soFar) {
				continue
			}
			match = append(match, cand)
		}
		if len(match) == 0 {
			return
		}
		if len(match) > 1 {
			// TODO: auto-complete any common prefix
			app.logf("%s", strings.Join(match, " "))
			return line, pos, true
		}
		newLine = line[:len(line)-len(soFar)] + match[0]
		return newLine, len(newLine), true

	}

	errc := make(chan error, 2)
	go func() { errc <- app.readFrames() }()
	go func() { errc <- app.readConsole() }()
	return <-errc
}

func (app *h2i) logf(format string, args ...interface{}) {
	fmt.Fprintf(app.term, format+"\r\n", args...)
}

func (app *h2i) readConsole() error {
	if s := *flagSettings; s != "omit" {
		var args []string
		if s != "empty" {
			args = strings.Split(s, ",")
		}
		_, c, ok := lookupCommand("settings")
		if !ok {
			panic("settings command not found")
		}
		c.run(app, args)
	}

	for {
		line, err := app.term.ReadLine()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("term.ReadLine: %v", err)
		}
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		cmd, args := f[0], f[1:]
		if _, c, ok := lookupCommand(cmd); ok {
			err = c.run(app, args)
		} else {
			app.logf("Unknown command %q", line)
		}
		if err == errExitApp {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func lookupCommand(prefix string) (name string, c command, ok bool) {
	prefix = strings.ToLower(prefix)
	if c, ok = commands[prefix]; ok {
		return prefix, c, ok
	}

	for full, candidate := range commands {
		if strings.HasPrefix(full, prefix) {
			if c.run != nil {
				return "", command{}, false // ambiguous
			}
			c = candidate
			name = full
		}
	}
	return name, c, c.run != nil
}

var errExitApp = errors.New("internal sentinel error value to quit the console reading loop")

func (a *h2i) cmdQuit(args []string) error {
	if len(args) > 0 {
		a.logf("the QUIT command takes no argument")
		return nil
	}
	return errExitApp
}

func (a *h2i) cmdSettings(args []string) error {
	if len(args) == 1 && strings.EqualFold(args[0], "ACK") {
		return a.framer.WriteSettingsAck()
	}
	var settings []http2.Setting
	for _, arg := range args {
		if strings.EqualFold(arg, "ACK") {
			a.logf("Error: ACK must be only argument with the SETTINGS command")
			return nil
		}
		eq := strings.Index(arg, "=")
		if eq == -1 {
			a.logf("Error: invalid argument %q (expected SETTING_NAME=nnnn)", arg)
			return nil
		}
		sid, ok := settingByName(arg[:eq])
		if !ok {
			a.logf("Error: unknown setting name %q", arg[:eq])
			return nil
		}
		val, err := strconv.ParseUint(arg[eq+1:], 10, 32)
		if err != nil {
			a.logf("Error: invalid argument %q (expected SETTING_NAME=nnnn)", arg)
			return nil
		}
		settings = append(settings, http2.Setting{
			ID:  sid,
			Val: uint32(val),
		})
	}
	a.logf("Sending: %v", settings)
	return a.framer.WriteSettings(settings...)
}

func settingByName(name string) (http2.SettingID, bool) {
	for _, sid := range [...]http2.SettingID{
		http2.SettingHeaderTableSize,
		http2.SettingEnablePush,
		http2.SettingMaxConcurrentStreams,
		http2.SettingInitialWindowSize,
		http2.SettingMaxFrameSize,
		http2.SettingMaxHeaderListSize,
	} {
		if strings.EqualFold(sid.String(), name) {
			return sid, true
		}
	}
	return 0, false
}

func (app *h2i) cmdPing(args []string) error {
	if len(args) > 1 {
		app.logf("invalid PING usage: only accepts 0 or 1 args")
		return nil // nil means don't end the program
	}
	var data [8]byte
	if len(args) == 1 {
		copy(data[:], args[0])
	} else {
		copy(data[:], "h2i_ping")
	}
	return app.framer.WritePing(false, data)
}

func (app *h2i) cmdHeaders(args []string) error {
	if len(args) > 0 {
		app.logf("Error: HEADERS doesn't yet take arguments.")
		// TODO: flags for restricting window size, to force CONTINUATION
		// frames.
		return nil
	}
	var h1req bytes.Buffer
	app.term.SetPrompt("(as HTTP/1.1)> ")
	defer app.term.SetPrompt("h2i> ")
	for {
		line, err := app.term.ReadLine()
		if err != nil {
			return err
		}
		h1req.WriteString(line)
		h1req.WriteString("\r\n")
		if line == "" {
			break
		}
	}
	req, err := http.ReadRequest(bufio.NewReader(&h1req))
	if err != nil {
		app.logf("Invalid HTTP/1.1 request: %v", err)
		return nil
	}
	if app.streamID == 0 {
		app.streamID = 1
	} else {
		app.streamID += 2
	}
	app.logf("Opening Stream-ID %d:", app.streamID)
	hbf := app.encodeHeaders(req)
	if len(hbf) > 16<<10 {
		app.logf("TODO: h2i doesn't yet write CONTINUATION frames. Copy it from transport.go")
		return nil
	}
	return app.framer.WriteHeaders(http2.HeadersFrameParam{
		StreamID:      app.streamID,
		BlockFragment: hbf,
		EndStream:     req.Method == "GET" || req.Method == "HEAD", // good enough for now
		EndHeaders:    true,                                        // for now
	})
}

func (app *h2i) readFrames() error {
	for {
		f, err := app.framer.ReadFrame()
		if err != nil {
			return fmt.Errorf("ReadFrame: %v", err)
		}
		app.logf("%v", f)
		switch f := f.(type) {
		case *http2.PingFrame:
			app.logf("  Data = %q", f.Data)
		case *http2.SettingsFrame:
			f.ForeachSetting(func(s http2.Setting) error {
				app.logf("  %v", s)
				app.peerSetting[s.ID] = s.Val
				return nil
			})
		case *http2.WindowUpdateFrame:
			app.logf("  Window-Increment = %v", f.Increment)
		case *http2.GoAwayFrame:
			app.logf("  Last-Stream-ID = %d; Error-Code = %v (%d)", f.LastStreamID, f.ErrCode, f.ErrCode)
		case *http2.DataFrame:
			app.logf("  %q", f.Data())
		case *http2.HeadersFrame:
			if f.HasPriority() {
				app.logf("  PRIORITY = %v", f.Priority)
			}
			if app.hdec == nil {
				// TODO: if the user uses h2i to send a SETTINGS frame advertising
				// something larger, we'll need to respect SETTINGS_HEADER_TABLE_SIZE
				// and stuff here instead of using the 4k default. But for now:
				tableSize := uint32(4 << 10)
				app.hdec = hpack.NewDecoder(tableSize, app.onNewHeaderField)
			}
			app.hdec.Write(f.HeaderBlockFragment())
		case *http2.PushPromiseFrame:
			if app.hdec == nil {
				// TODO: if the user uses h2i to send a SETTINGS frame advertising
				// something larger, we'll need to respect SETTINGS_HEADER_TABLE_SIZE
				// and stuff here instead of using the 4k default. But for now:
				tableSize := uint32(4 << 10)
				app.hdec = hpack.NewDecoder(tableSize, app.onNewHeaderField)
			}
			app.hdec.Write(f.HeaderBlockFragment())
		}
	}
}

// called from readLoop
func (app *h2i) onNewHeaderField(f hpack.HeaderField) {
	if f.Sensitive {
		app.logf("  %s = %q (SENSITIVE)", f.Name, f.Value)
	}
	app.logf("  %s = %q", f.Name, f.Value)
}

func (app *h2i) encodeHeaders(req *http.Request) []byte {
	app.hbuf.Reset()

	// TODO(bradfitz): figure out :authority-vs-Host stuff between http2 and Go
	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	path := req.RequestURI
	if path == "" {
		path = "/"
	}

	app.writeHeader(":authority", host) // probably not right for all sites
	app.writeHeader(":method", req.Method)
	app.writeHeader(":path", path)
	app.writeHeader(":scheme", "https")

	for k, vv := range req.Header {
		lowKey := strings.ToLower(k)
		if lowKey == "host" {
			continue
		}
		for _, v := range vv {
			app.writeHeader(lowKey, v)
		}
	}
	return app.hbuf.Bytes()
}

func (app *h2i) writeHeader(name, value string) {
	app.henc.WriteField(hpack.HeaderField{Name: name, Value: value})
	app.logf(" %s = %s", name, value)
}
