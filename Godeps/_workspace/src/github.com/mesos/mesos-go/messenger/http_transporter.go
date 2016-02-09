/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package messenger

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

var (
	discardOnStopError = fmt.Errorf("discarding message because transport is shutting down")
	errNotStarted      = errors.New("HTTP transport has not been started")
	errTerminal        = errors.New("HTTP transport is terminated")
	errAlreadyRunning  = errors.New("HTTP transport is already running")

	httpTransport, httpClient = &http.Transport{
		Dial: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
	},
		&http.Client{
			Transport: httpTransport,
			Timeout:   DefaultReadTimeout,
		}
)

// httpTransporter is a subset of the Transporter interface
type httpTransporter interface {
	Send(ctx context.Context, msg *Message) error
	Recv() (*Message, error)
	Inject(ctx context.Context, msg *Message) error
	Install(messageName string)
	Start() (upid.UPID, <-chan error)
	Stop(graceful bool) error
}

type notStartedState struct {
	h *HTTPTransporter
}

type stoppedState struct{}

type runningState struct {
	*notStartedState
}

/* -- not-started state */

func (s *notStartedState) Send(ctx context.Context, msg *Message) error   { return errNotStarted }
func (s *notStartedState) Recv() (*Message, error)                        { return nil, errNotStarted }
func (s *notStartedState) Inject(ctx context.Context, msg *Message) error { return errNotStarted }
func (s *notStartedState) Stop(graceful bool) error                       { return errNotStarted }
func (s *notStartedState) Install(messageName string)                     { s.h.install(messageName) }
func (s *notStartedState) Start() (upid.UPID, <-chan error) {
	s.h.state = &runningState{s}
	return s.h.start()
}

/* -- stopped state */

func (s *stoppedState) Send(ctx context.Context, msg *Message) error   { return errTerminal }
func (s *stoppedState) Recv() (*Message, error)                        { return nil, errTerminal }
func (s *stoppedState) Inject(ctx context.Context, msg *Message) error { return errTerminal }
func (s *stoppedState) Stop(graceful bool) error                       { return errTerminal }
func (s *stoppedState) Install(messageName string)                     {}
func (s *stoppedState) Start() (upid.UPID, <-chan error) {
	ch := make(chan error, 1)
	ch <- errTerminal
	return upid.UPID{}, ch
}

/* -- running state */

func (s *runningState) Send(ctx context.Context, msg *Message) error   { return s.h.send(ctx, msg) }
func (s *runningState) Recv() (*Message, error)                        { return s.h.recv() }
func (s *runningState) Inject(ctx context.Context, msg *Message) error { return s.h.inject(ctx, msg) }
func (s *runningState) Stop(graceful bool) error {
	s.h.state = &stoppedState{}
	return s.h.stop(graceful)
}
func (s *runningState) Start() (upid.UPID, <-chan error) {
	ch := make(chan error, 1)
	ch <- errAlreadyRunning
	return upid.UPID{}, ch
}

// HTTPTransporter implements the interfaces of the Transporter.
type HTTPTransporter struct {
	// If the host is empty("") then it will listen on localhost.
	// If the port is empty("") then it will listen on random port.
	upid         upid.UPID
	listener     net.Listener // TODO(yifan): Change to TCPListener.
	mux          *http.ServeMux
	tr           *http.Transport
	client       *http.Client
	messageQueue chan *Message
	address      net.IP // optional binding address
	shouldQuit   chan struct{}
	stateLock    sync.RWMutex // protect lifecycle (start/stop) funcs
	state        httpTransporter
}

// NewHTTPTransporter creates a new http transporter with an optional binding address.
func NewHTTPTransporter(upid upid.UPID, address net.IP) *HTTPTransporter {
	result := &HTTPTransporter{
		upid:         upid,
		messageQueue: make(chan *Message, defaultQueueSize),
		mux:          http.NewServeMux(),
		client:       httpClient,
		tr:           httpTransport,
		address:      address,
		shouldQuit:   make(chan struct{}),
	}
	result.state = &notStartedState{result}
	return result
}

func (t *HTTPTransporter) getState() httpTransporter {
	t.stateLock.RLock()
	defer t.stateLock.RUnlock()
	return t.state
}

// some network errors are probably recoverable, attempt to determine that here.
func isRecoverableError(err error) bool {
	if urlErr, ok := err.(*url.Error); ok {
		log.V(2).Infof("checking url.Error for recoverability")
		return urlErr.Op == "Post" && isRecoverableError(urlErr.Err)
	} else if netErr, ok := err.(*net.OpError); ok && netErr.Err != nil {
		log.V(2).Infof("checking net.OpError for recoverability: %#v", err)
		if netErr.Temporary() {
			return true
		}
		//TODO(jdef) this is pretty hackish, there's probably a better way
		//TODO(jdef) should also check for EHOSTDOWN and EHOSTUNREACH
		return (netErr.Op == "dial" && netErr.Net == "tcp" && netErr.Err == syscall.ECONNREFUSED)
	}
	log.V(2).Infof("unrecoverable error: %#v", err)
	return false
}

type recoverableError struct {
	Err error
}

func (e *recoverableError) Error() string {
	if e == nil {
		return ""
	}
	return e.Err.Error()
}

// Send sends the message to its specified upid.
func (t *HTTPTransporter) Send(ctx context.Context, msg *Message) (sendError error) {
	return t.getState().Send(ctx, msg)
}

func (t *HTTPTransporter) send(ctx context.Context, msg *Message) (sendError error) {
	log.V(2).Infof("Sending message to %v via http\n", msg.UPID)
	req, err := t.makeLibprocessRequest(msg)
	if err != nil {
		log.Errorf("Failed to make libprocess request: %v\n", err)
		return err
	}
	duration := 1 * time.Second
	for attempt := 0; attempt < 5; attempt++ { //TODO(jdef) extract/parameterize constant
		if sendError != nil {
			duration *= 2
			log.Warningf("attempting to recover from error '%v', waiting before retry: %v", sendError, duration)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(duration):
				// ..retry request, continue
			case <-t.shouldQuit:
				return discardOnStopError
			}
		}
		sendError = t.httpDo(ctx, req, func(resp *http.Response, err error) error {
			if err != nil {
				if isRecoverableError(err) {
					return &recoverableError{Err: err}
				}
				log.Infof("Failed to POST: %v\n", err)
				return err
			}
			defer resp.Body.Close()

			// ensure master acknowledgement.
			if (resp.StatusCode != http.StatusOK) &&
				(resp.StatusCode != http.StatusAccepted) {
				msg := fmt.Sprintf("Master %s rejected %s.  Returned status %s.",
					msg.UPID, msg.RequestURI(), resp.Status)
				log.Warning(msg)
				return fmt.Errorf(msg)
			}
			return nil
		})
		if sendError == nil {
			// success
			return
		} else if _, ok := sendError.(*recoverableError); ok {
			// recoverable, attempt backoff?
			continue
		}
		// unrecoverable
		break
	}
	if recoverable, ok := sendError.(*recoverableError); ok {
		sendError = recoverable.Err
	}
	return
}

func (t *HTTPTransporter) httpDo(ctx context.Context, req *http.Request, f func(*http.Response, error) error) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.shouldQuit:
		return discardOnStopError
	default: // continue
	}

	c := make(chan error, 1)
	go func() { c <- f(t.client.Do(req)) }()
	select {
	case <-ctx.Done():
		t.tr.CancelRequest(req)
		<-c // Wait for f to return.
		return ctx.Err()
	case err := <-c:
		return err
	case <-t.shouldQuit:
		t.tr.CancelRequest(req)
		<-c // Wait for f to return.
		return discardOnStopError
	}
}

// Recv returns the message, one at a time.
func (t *HTTPTransporter) Recv() (*Message, error) {
	return t.getState().Recv()
}

func (t *HTTPTransporter) recv() (*Message, error) {
	select {
	default:
		select {
		case msg := <-t.messageQueue:
			return msg, nil
		case <-t.shouldQuit:
		}
	case <-t.shouldQuit:
	}
	return nil, discardOnStopError
}

//Inject places a message into the incoming message queue.
func (t *HTTPTransporter) Inject(ctx context.Context, msg *Message) error {
	return t.getState().Inject(ctx, msg)
}

func (t *HTTPTransporter) inject(ctx context.Context, msg *Message) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.shouldQuit:
		return discardOnStopError
	default: // continue
	}

	select {
	case t.messageQueue <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-t.shouldQuit:
		return discardOnStopError
	}
}

// Install the request URI according to the message's name.
func (t *HTTPTransporter) Install(msgName string) {
	t.getState().Install(msgName)
}

func (t *HTTPTransporter) install(msgName string) {
	requestURI := fmt.Sprintf("/%s/%s", t.upid.ID, msgName)
	t.mux.HandleFunc(requestURI, t.messageDecoder)
}

type loggedListener struct {
	delegate net.Listener
	done     <-chan struct{}
}

func (l *loggedListener) Accept() (c net.Conn, err error) {
	c, err = l.delegate.Accept()
	if c != nil {
		log.Infoln("accepted connection from", c.RemoteAddr())
		c = logConnection(c)
	} else if err != nil {
		select {
		case <-l.done:
		default:
			log.Errorln("failed to accept connection:", err.Error())
		}
	}
	return
}

func (l *loggedListener) Close() (err error) {
	err = l.delegate.Close()
	if err != nil {
		select {
		case <-l.done:
		default:
			log.Errorln("error closing listener:", err.Error())
		}
	} else {
		log.Infoln("closed listener")
	}
	return
}

func (l *loggedListener) Addr() net.Addr { return l.delegate.Addr() }

func logConnection(c net.Conn) net.Conn {
	w := hex.Dumper(os.Stdout)
	r := io.TeeReader(c, w)
	return &loggedConnection{
		Conn:   c,
		reader: r,
	}
}

type loggedConnection struct {
	net.Conn
	reader io.Reader
}

func (c *loggedConnection) Read(b []byte) (int, error) {
	return c.reader.Read(b)
}

// Listen starts listen on UPID. If UPID is empty, the transporter
// will listen on a random port, and then fill the UPID with the
// host:port it is listening.
func (t *HTTPTransporter) listen() error {
	var host string
	if t.address != nil {
		host = t.address.String()
	} else {
		host = t.upid.Host
	}

	var port string
	if t.upid.Port != "" {
		port = t.upid.Port
	} else {
		port = "0"
	}

	// NOTE: Explicitly specifies IPv4 because Libprocess
	// only supports IPv4 for now.
	ln, err := net.Listen("tcp4", net.JoinHostPort(host, port))
	if err != nil {
		log.Errorf("HTTPTransporter failed to listen: %v\n", err)
		return err
	}
	// Save the host:port in case they are not specified in upid.
	host, port, _ = net.SplitHostPort(ln.Addr().String())
	log.Infoln("listening on", host, "port", port)

	if len(t.upid.Host) == 0 {
		t.upid.Host = host
	}

	if len(t.upid.Port) == 0 || t.upid.Port == "0" {
		t.upid.Port = port
	}

	if log.V(3) {
		t.listener = &loggedListener{delegate: ln, done: t.shouldQuit}
	} else {
		t.listener = ln
	}
	return nil
}

// Start starts the http transporter
func (t *HTTPTransporter) Start() (upid.UPID, <-chan error) {
	t.stateLock.Lock()
	defer t.stateLock.Unlock()
	return t.state.Start()
}

// start expects to be guarded by stateLock
func (t *HTTPTransporter) start() (upid.UPID, <-chan error) {
	ch := make(chan error, 1)
	if err := t.listen(); err != nil {
		ch <- err
		return upid.UPID{}, ch
	}

	// TODO(yifan): Set read/write deadline.
	go func() {
		s := &http.Server{
			ReadTimeout:  DefaultReadTimeout,
			WriteTimeout: DefaultWriteTimeout,
			Handler:      t.mux,
		}
		err := s.Serve(t.listener)
		select {
		case <-t.shouldQuit:
			log.V(1).Infof("HTTP server stopped because of shutdown")
			ch <- nil
		default:
			if err != nil && log.V(1) {
				log.Errorln("HTTP server stopped with error", err.Error())
			} else {
				log.V(1).Infof("HTTP server stopped")
			}
			ch <- err
			t.Stop(false)
		}
	}()
	return t.upid, ch
}

// Stop stops the http transporter by closing the listener.
func (t *HTTPTransporter) Stop(graceful bool) error {
	t.stateLock.Lock()
	defer t.stateLock.Unlock()
	return t.state.Stop(graceful)
}

// stop expects to be guarded by stateLock
func (t *HTTPTransporter) stop(graceful bool) error {
	close(t.shouldQuit)

	log.Info("stopping HTTP transport")

	//TODO(jdef) if graceful, wait for pending requests to terminate

	err := t.listener.Close()
	return err
}

// UPID returns the upid of the transporter.
func (t *HTTPTransporter) UPID() upid.UPID {
	t.stateLock.Lock()
	defer t.stateLock.Unlock()
	return t.upid
}

func (t *HTTPTransporter) messageDecoder(w http.ResponseWriter, r *http.Request) {
	// Verify it's a libprocess request.
	from, err := getLibprocessFrom(r)
	if err != nil {
		log.Errorf("Ignoring the request, because it's not a libprocess request: %v\n", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	decoder := DecodeHTTP(w, r)
	defer decoder.Cancel(true)

	t.processRequests(from, decoder.Requests())

	// log an error if there's one waiting, otherwise move on
	select {
	case err, ok := <-decoder.Err():
		if ok {
			log.Errorf("failed to decode HTTP message: %v", err)
		}
	default:
	}
}

func (t *HTTPTransporter) processRequests(from *upid.UPID, incoming <-chan *Request) {
	for {
		select {
		case r, ok := <-incoming:
			if !ok || !t.processOneRequest(from, r) {
				return
			}
		case <-t.shouldQuit:
			return
		}
	}
}

func (t *HTTPTransporter) processOneRequest(from *upid.UPID, request *Request) (keepGoing bool) {
	// regardless of whether we write a Response we must close this chan
	defer close(request.response)
	keepGoing = true

	//TODO(jdef) this is probably inefficient given the current implementation of the
	// decoder: no need to make another copy of data that's already competely buffered
	data, err := ioutil.ReadAll(request.Body)
	if err != nil {
		// this is unlikely given the current implementation of the decoder:
		// the body has been completely buffered in memory already
		log.Errorf("failed to read HTTP body: %v", err)
		return
	}
	log.V(2).Infof("Receiving %q %v from %v, length %v", request.Method, request.URL, from, len(data))
	m := &Message{
		UPID:  from,
		Name:  extractNameFromRequestURI(request.RequestURI),
		Bytes: data,
	}

	// deterministic behavior and output..
	select {
	case <-t.shouldQuit:
		keepGoing = false
		select {
		case t.messageQueue <- m:
		default:
		}
	case t.messageQueue <- m:
		select {
		case <-t.shouldQuit:
			keepGoing = false
		default:
		}
	}

	// Only send back an HTTP response if this isn't from libprocess
	// (which we determine by looking at the User-Agent). This is
	// necessary because older versions of libprocess would try and
	// recv the data and parse it as an HTTP request which would
	// fail thus causing the socket to get closed (but now
	// libprocess will ignore responses, see ignore_data).
	// see https://github.com/apache/mesos/blob/adecbfa6a216815bd7dc7d26e721c4c87e465c30/3rdparty/libprocess/src/process.cpp#L2192
	if _, ok := parseLibprocessAgent(request.Header); !ok {
		log.V(2).Infof("not libprocess agent, sending a 202")
		request.response <- Response{
			code:   202,
			reason: "Accepted",
		} // should never block
	}
	return
}

func (t *HTTPTransporter) makeLibprocessRequest(msg *Message) (*http.Request, error) {
	if msg.UPID == nil {
		panic(fmt.Sprintf("message is missing UPID: %+v", msg))
	}
	hostport := net.JoinHostPort(msg.UPID.Host, msg.UPID.Port)
	targetURL := fmt.Sprintf("http://%s%s", hostport, msg.RequestURI())
	log.V(2).Infof("libproc target URL %s", targetURL)
	req, err := http.NewRequest("POST", targetURL, bytes.NewReader(msg.Bytes))
	if err != nil {
		log.Errorf("Failed to create request: %v\n", err)
		return nil, err
	}
	if !msg.isV1API() {
		req.Header.Add("Libprocess-From", t.upid.String())
		req.Header.Add("Connection", "Keep-Alive")
	}
	req.Header.Add("Content-Type", "application/x-protobuf")

	return req, nil
}

func getLibprocessFrom(r *http.Request) (*upid.UPID, error) {
	if r.Method != "POST" {
		return nil, fmt.Errorf("Not a POST request")
	}
	if agent, ok := parseLibprocessAgent(r.Header); ok {
		return upid.Parse(agent)
	}
	lf, ok := r.Header["Libprocess-From"]
	if ok {
		// TODO(yifan): Just take the first field for now.
		return upid.Parse(lf[0])
	}
	return nil, fmt.Errorf("Cannot find 'User-Agent' or 'Libprocess-From'")
}

func parseLibprocessAgent(h http.Header) (string, bool) {
	const prefix = "libprocess/"
	if ua, ok := h["User-Agent"]; ok {
		for _, agent := range ua {
			if strings.HasPrefix(agent, prefix) {
				return agent[len(prefix):], true
			}
		}
	}
	return "", false
}
