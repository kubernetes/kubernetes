// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Session tests.

import (
	"bytes"
	crypto_rand "crypto/rand"
	"errors"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"testing"

	"golang.org/x/crypto/ssh/terminal"
)

type serverType func(Channel, <-chan *Request, *testing.T)

// dial constructs a new test server and returns a *ClientConn.
func dial(handler serverType, t *testing.T) *Client {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}

	go func() {
		defer c1.Close()
		conf := ServerConfig{
			NoClientAuth: true,
		}
		conf.AddHostKey(testSigners["rsa"])

		_, chans, reqs, err := NewServerConn(c1, &conf)
		if err != nil {
			t.Fatalf("Unable to handshake: %v", err)
		}
		go DiscardRequests(reqs)

		for newCh := range chans {
			if newCh.ChannelType() != "session" {
				newCh.Reject(UnknownChannelType, "unknown channel type")
				continue
			}

			ch, inReqs, err := newCh.Accept()
			if err != nil {
				t.Errorf("Accept: %v", err)
				continue
			}
			go func() {
				handler(ch, inReqs, t)
			}()
		}
	}()

	config := &ClientConfig{
		User: "testuser",
	}

	conn, chans, reqs, err := NewClientConn(c2, "", config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %v", err)
	}

	return NewClient(conn, chans, reqs)
}

// Test a simple string is returned to session.Stdout.
func TestSessionShell(t *testing.T) {
	conn := dial(shellHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	stdout := new(bytes.Buffer)
	session.Stdout = stdout
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	if err := session.Wait(); err != nil {
		t.Fatalf("Remote command did not exit cleanly: %v", err)
	}
	actual := stdout.String()
	if actual != "golang" {
		t.Fatalf("Remote shell did not return expected string: expected=golang, actual=%s", actual)
	}
}

// TODO(dfc) add support for Std{in,err}Pipe when the Server supports it.

// Test a simple string is returned via StdoutPipe.
func TestSessionStdoutPipe(t *testing.T) {
	conn := dial(shellHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	stdout, err := session.StdoutPipe()
	if err != nil {
		t.Fatalf("Unable to request StdoutPipe(): %v", err)
	}
	var buf bytes.Buffer
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	done := make(chan bool, 1)
	go func() {
		if _, err := io.Copy(&buf, stdout); err != nil {
			t.Errorf("Copy of stdout failed: %v", err)
		}
		done <- true
	}()
	if err := session.Wait(); err != nil {
		t.Fatalf("Remote command did not exit cleanly: %v", err)
	}
	<-done
	actual := buf.String()
	if actual != "golang" {
		t.Fatalf("Remote shell did not return expected string: expected=golang, actual=%s", actual)
	}
}

// Test that a simple string is returned via the Output helper,
// and that stderr is discarded.
func TestSessionOutput(t *testing.T) {
	conn := dial(fixedOutputHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()

	buf, err := session.Output("") // cmd is ignored by fixedOutputHandler
	if err != nil {
		t.Error("Remote command did not exit cleanly:", err)
	}
	w := "this-is-stdout."
	g := string(buf)
	if g != w {
		t.Error("Remote command did not return expected string:")
		t.Logf("want %q", w)
		t.Logf("got  %q", g)
	}
}

// Test that both stdout and stderr are returned
// via the CombinedOutput helper.
func TestSessionCombinedOutput(t *testing.T) {
	conn := dial(fixedOutputHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()

	buf, err := session.CombinedOutput("") // cmd is ignored by fixedOutputHandler
	if err != nil {
		t.Error("Remote command did not exit cleanly:", err)
	}
	const stdout = "this-is-stdout."
	const stderr = "this-is-stderr."
	g := string(buf)
	if g != stdout+stderr && g != stderr+stdout {
		t.Error("Remote command did not return expected string:")
		t.Logf("want %q, or %q", stdout+stderr, stderr+stdout)
		t.Logf("got  %q", g)
	}
}

// Test non-0 exit status is returned correctly.
func TestExitStatusNonZero(t *testing.T) {
	conn := dial(exitStatusNonZeroHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.ExitStatus() != 15 {
		t.Fatalf("expected command to exit with 15 but got %v", e.ExitStatus())
	}
}

// Test 0 exit status is returned correctly.
func TestExitStatusZero(t *testing.T) {
	conn := dial(exitStatusZeroHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()

	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err != nil {
		t.Fatalf("expected nil but got %v", err)
	}
}

// Test exit signal and status are both returned correctly.
func TestExitSignalAndStatus(t *testing.T) {
	conn := dial(exitSignalAndStatusHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "TERM" || e.ExitStatus() != 15 {
		t.Fatalf("expected command to exit with signal TERM and status 15 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

// Test exit signal and status are both returned correctly.
func TestKnownExitSignalOnly(t *testing.T) {
	conn := dial(exitSignalHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "TERM" || e.ExitStatus() != 143 {
		t.Fatalf("expected command to exit with signal TERM and status 143 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

// Test exit signal and status are both returned correctly.
func TestUnknownExitSignal(t *testing.T) {
	conn := dial(exitSignalUnknownHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "SYS" || e.ExitStatus() != 128 {
		t.Fatalf("expected command to exit with signal SYS and status 128 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

func TestExitWithoutStatusOrSignal(t *testing.T) {
	conn := dial(exitWithoutSignalOrStatus, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %v", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	if _, ok := err.(*ExitMissingError); !ok {
		t.Fatalf("got %T want *ExitMissingError", err)
	}
}

// windowTestBytes is the number of bytes that we'll send to the SSH server.
const windowTestBytes = 16000 * 200

// TestServerWindow writes random data to the server. The server is expected to echo
// the same data back, which is compared against the original.
func TestServerWindow(t *testing.T) {
	origBuf := bytes.NewBuffer(make([]byte, 0, windowTestBytes))
	io.CopyN(origBuf, crypto_rand.Reader, windowTestBytes)
	origBytes := origBuf.Bytes()

	conn := dial(echoHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	result := make(chan []byte)

	go func() {
		defer close(result)
		echoedBuf := bytes.NewBuffer(make([]byte, 0, windowTestBytes))
		serverStdout, err := session.StdoutPipe()
		if err != nil {
			t.Errorf("StdoutPipe failed: %v", err)
			return
		}
		n, err := copyNRandomly("stdout", echoedBuf, serverStdout, windowTestBytes)
		if err != nil && err != io.EOF {
			t.Errorf("Read only %d bytes from server, expected %d: %v", n, windowTestBytes, err)
		}
		result <- echoedBuf.Bytes()
	}()

	serverStdin, err := session.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe failed: %v", err)
	}
	written, err := copyNRandomly("stdin", serverStdin, origBuf, windowTestBytes)
	if err != nil {
		t.Fatalf("failed to copy origBuf to serverStdin: %v", err)
	}
	if written != windowTestBytes {
		t.Fatalf("Wrote only %d of %d bytes to server", written, windowTestBytes)
	}

	echoedBytes := <-result

	if !bytes.Equal(origBytes, echoedBytes) {
		t.Fatalf("Echoed buffer differed from original, orig %d, echoed %d", len(origBytes), len(echoedBytes))
	}
}

// Verify the client can handle a keepalive packet from the server.
func TestClientHandlesKeepalives(t *testing.T) {
	conn := dial(channelKeepaliveSender, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %v", err)
	}
	err = session.Wait()
	if err != nil {
		t.Fatalf("expected nil but got: %v", err)
	}
}

type exitStatusMsg struct {
	Status uint32
}

type exitSignalMsg struct {
	Signal     string
	CoreDumped bool
	Errmsg     string
	Lang       string
}

func handleTerminalRequests(in <-chan *Request) {
	for req := range in {
		ok := false
		switch req.Type {
		case "shell":
			ok = true
			if len(req.Payload) > 0 {
				// We don't accept any commands, only the default shell.
				ok = false
			}
		case "env":
			ok = true
		}
		req.Reply(ok, nil)
	}
}

func newServerShell(ch Channel, in <-chan *Request, prompt string) *terminal.Terminal {
	term := terminal.NewTerminal(ch, prompt)
	go handleTerminalRequests(in)
	return term
}

func exitStatusZeroHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	// this string is returned to stdout
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	sendStatus(0, ch, t)
}

func exitStatusNonZeroHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	sendStatus(15, ch, t)
}

func exitSignalAndStatusHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	sendStatus(15, ch, t)
	sendSignal("TERM", ch, t)
}

func exitSignalHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	sendSignal("TERM", ch, t)
}

func exitSignalUnknownHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	sendSignal("SYS", ch, t)
}

func exitWithoutSignalOrStatus(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
}

func shellHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	// this string is returned to stdout
	shell := newServerShell(ch, in, "golang")
	readLine(shell, t)
	sendStatus(0, ch, t)
}

// Ignores the command, writes fixed strings to stderr and stdout.
// Strings are "this-is-stdout." and "this-is-stderr.".
func fixedOutputHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	_, err := ch.Read(nil)

	req, ok := <-in
	if !ok {
		t.Fatalf("error: expected channel request, got: %#v", err)
		return
	}

	// ignore request, always send some text
	req.Reply(true, nil)

	_, err = io.WriteString(ch, "this-is-stdout.")
	if err != nil {
		t.Fatalf("error writing on server: %v", err)
	}
	_, err = io.WriteString(ch.Stderr(), "this-is-stderr.")
	if err != nil {
		t.Fatalf("error writing on server: %v", err)
	}
	sendStatus(0, ch, t)
}

func readLine(shell *terminal.Terminal, t *testing.T) {
	if _, err := shell.ReadLine(); err != nil && err != io.EOF {
		t.Errorf("unable to read line: %v", err)
	}
}

func sendStatus(status uint32, ch Channel, t *testing.T) {
	msg := exitStatusMsg{
		Status: status,
	}
	if _, err := ch.SendRequest("exit-status", false, Marshal(&msg)); err != nil {
		t.Errorf("unable to send status: %v", err)
	}
}

func sendSignal(signal string, ch Channel, t *testing.T) {
	sig := exitSignalMsg{
		Signal:     signal,
		CoreDumped: false,
		Errmsg:     "Process terminated",
		Lang:       "en-GB-oed",
	}
	if _, err := ch.SendRequest("exit-signal", false, Marshal(&sig)); err != nil {
		t.Errorf("unable to send signal: %v", err)
	}
}

func discardHandler(ch Channel, t *testing.T) {
	defer ch.Close()
	io.Copy(ioutil.Discard, ch)
}

func echoHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	if n, err := copyNRandomly("echohandler", ch, ch, windowTestBytes); err != nil {
		t.Errorf("short write, wrote %d, expected %d: %v ", n, windowTestBytes, err)
	}
}

// copyNRandomly copies n bytes from src to dst. It uses a variable, and random,
// buffer size to exercise more code paths.
func copyNRandomly(title string, dst io.Writer, src io.Reader, n int) (int, error) {
	var (
		buf       = make([]byte, 32*1024)
		written   int
		remaining = n
	)
	for remaining > 0 {
		l := rand.Intn(1 << 15)
		if remaining < l {
			l = remaining
		}
		nr, er := src.Read(buf[:l])
		nw, ew := dst.Write(buf[:nr])
		remaining -= nw
		written += nw
		if ew != nil {
			return written, ew
		}
		if nr != nw {
			return written, io.ErrShortWrite
		}
		if er != nil && er != io.EOF {
			return written, er
		}
	}
	return written, nil
}

func channelKeepaliveSender(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	shell := newServerShell(ch, in, "> ")
	readLine(shell, t)
	if _, err := ch.SendRequest("keepalive@openssh.com", true, nil); err != nil {
		t.Errorf("unable to send channel keepalive request: %v", err)
	}
	sendStatus(0, ch, t)
}

func TestClientWriteEOF(t *testing.T) {
	conn := dial(simpleEchoHandler, t)
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	stdin, err := session.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe failed: %v", err)
	}
	stdout, err := session.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe failed: %v", err)
	}

	data := []byte(`0000`)
	_, err = stdin.Write(data)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	stdin.Close()

	res, err := ioutil.ReadAll(stdout)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if !bytes.Equal(data, res) {
		t.Fatalf("Read differed from write, wrote: %v, read: %v", data, res)
	}
}

func simpleEchoHandler(ch Channel, in <-chan *Request, t *testing.T) {
	defer ch.Close()
	data, err := ioutil.ReadAll(ch)
	if err != nil {
		t.Errorf("handler read error: %v", err)
	}
	_, err = ch.Write(data)
	if err != nil {
		t.Errorf("handler write error: %v", err)
	}
}

func TestSessionID(t *testing.T) {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	serverID := make(chan []byte, 1)
	clientID := make(chan []byte, 1)

	serverConf := &ServerConfig{
		NoClientAuth: true,
	}
	serverConf.AddHostKey(testSigners["ecdsa"])
	clientConf := &ClientConfig{
		User: "user",
	}

	go func() {
		conn, chans, reqs, err := NewServerConn(c1, serverConf)
		if err != nil {
			t.Fatalf("server handshake: %v", err)
		}
		serverID <- conn.SessionID()
		go DiscardRequests(reqs)
		for ch := range chans {
			ch.Reject(Prohibited, "")
		}
	}()

	go func() {
		conn, chans, reqs, err := NewClientConn(c2, "", clientConf)
		if err != nil {
			t.Fatalf("client handshake: %v", err)
		}
		clientID <- conn.SessionID()
		go DiscardRequests(reqs)
		for ch := range chans {
			ch.Reject(Prohibited, "")
		}
	}()

	s := <-serverID
	c := <-clientID
	if bytes.Compare(s, c) != 0 {
		t.Errorf("server session ID (%x) != client session ID (%x)", s, c)
	} else if len(s) == 0 {
		t.Errorf("client and server SessionID were empty.")
	}
}

type noReadConn struct {
	readSeen bool
	net.Conn
}

func (c *noReadConn) Close() error {
	return nil
}

func (c *noReadConn) Read(b []byte) (int, error) {
	c.readSeen = true
	return 0, errors.New("noReadConn error")
}

func TestInvalidServerConfiguration(t *testing.T) {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	serveConn := noReadConn{Conn: c1}
	serverConf := &ServerConfig{}

	NewServerConn(&serveConn, serverConf)
	if serveConn.readSeen {
		t.Fatalf("NewServerConn attempted to Read() from Conn while configuration is missing host key")
	}

	serverConf.AddHostKey(testSigners["ecdsa"])

	NewServerConn(&serveConn, serverConf)
	if serveConn.readSeen {
		t.Fatalf("NewServerConn attempted to Read() from Conn while configuration is missing authentication method")
	}
}

func TestHostKeyAlgorithms(t *testing.T) {
	serverConf := &ServerConfig{
		NoClientAuth: true,
	}
	serverConf.AddHostKey(testSigners["rsa"])
	serverConf.AddHostKey(testSigners["ecdsa"])

	connect := func(clientConf *ClientConfig, want string) {
		var alg string
		clientConf.HostKeyCallback = func(h string, a net.Addr, key PublicKey) error {
			alg = key.Type()
			return nil
		}
		c1, c2, err := netPipe()
		if err != nil {
			t.Fatalf("netPipe: %v", err)
		}
		defer c1.Close()
		defer c2.Close()

		go NewServerConn(c1, serverConf)
		_, _, _, err = NewClientConn(c2, "", clientConf)
		if err != nil {
			t.Fatalf("NewClientConn: %v", err)
		}
		if alg != want {
			t.Errorf("selected key algorithm %s, want %s", alg, want)
		}
	}

	// By default, we get the preferred algorithm, which is ECDSA 256.

	clientConf := &ClientConfig{}
	connect(clientConf, KeyAlgoECDSA256)

	// Client asks for RSA explicitly.
	clientConf.HostKeyAlgorithms = []string{KeyAlgoRSA}
	connect(clientConf, KeyAlgoRSA)

	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	go NewServerConn(c1, serverConf)
	clientConf.HostKeyAlgorithms = []string{"nonexistent-hostkey-algo"}
	_, _, _, err = NewClientConn(c2, "", clientConf)
	if err == nil {
		t.Fatal("succeeded connecting with unknown hostkey algorithm")
	}
}
