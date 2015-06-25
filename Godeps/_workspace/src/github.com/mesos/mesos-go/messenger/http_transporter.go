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
	"fmt"
	"github.com/mesos/mesos-go/upid"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	log "github.com/golang/glog"
	"golang.org/x/net/context"
)

var (
	discardOnStopError = fmt.Errorf("discarding message because transport is shutting down")
)

// HTTPTransporter implements the interfaces of the Transporter.
type HTTPTransporter struct {
	// If the host is empty("") then it will listen on localhost.
	// If the port is empty("") then it will listen on random port.
	upid         *upid.UPID
	listener     net.Listener // TODO(yifan): Change to TCPListener.
	mux          *http.ServeMux
	tr           *http.Transport
	client       *http.Client // TODO(yifan): Set read/write deadline.
	messageQueue chan *Message
	address      net.IP // optional binding address
	started      chan struct{}
	stopped      chan struct{}
	stopping     int32
	lifeLock     sync.Mutex // protect lifecycle (start/stop) funcs
}

// NewHTTPTransporter creates a new http transporter with an optional binding address.
func NewHTTPTransporter(upid *upid.UPID, address net.IP) *HTTPTransporter {
	tr := &http.Transport{}
	result := &HTTPTransporter{
		upid:         upid,
		messageQueue: make(chan *Message, defaultQueueSize),
		mux:          http.NewServeMux(),
		client:       &http.Client{Transport: tr},
		tr:           tr,
		address:      address,
		started:      make(chan struct{}),
		stopped:      make(chan struct{}),
	}
	close(result.stopped)
	return result
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
			case <-t.stopped:
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
	case <-t.stopped:
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
	case <-t.stopped:
		t.tr.CancelRequest(req)
		<-c // Wait for f to return.
		return discardOnStopError
	}
}

// Recv returns the message, one at a time.
func (t *HTTPTransporter) Recv() (*Message, error) {
	select {
	default:
		select {
		case msg := <-t.messageQueue:
			return msg, nil
		case <-t.stopped:
		}
	case <-t.stopped:
	}
	return nil, discardOnStopError
}

//Inject places a message into the incoming message queue.
func (t *HTTPTransporter) Inject(ctx context.Context, msg *Message) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.stopped:
		return discardOnStopError
	default: // continue
	}

	select {
	case t.messageQueue <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-t.stopped:
		return discardOnStopError
	}
}

// Install the request URI according to the message's name.
func (t *HTTPTransporter) Install(msgName string) {
	requestURI := fmt.Sprintf("/%s/%s", t.upid.ID, msgName)
	t.mux.HandleFunc(requestURI, t.messageHandler)
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
	port := t.upid.Port
	// NOTE: Explicitly specifies IPv4 because Libprocess
	// only supports IPv4 for now.
	ln, err := net.Listen("tcp4", net.JoinHostPort(host, port))
	if err != nil {
		log.Errorf("HTTPTransporter failed to listen: %v\n", err)
		return err
	}
	// Save the host:port in case they are not specified in upid.
	host, port, _ = net.SplitHostPort(ln.Addr().String())
	t.upid.Host, t.upid.Port = host, port
	t.listener = ln
	return nil
}

// Start starts the http transporter
func (t *HTTPTransporter) Start() <-chan error {
	t.lifeLock.Lock()
	defer t.lifeLock.Unlock()

	select {
	case <-t.started:
		// already started
		return nil
	case <-t.stopped:
		defer close(t.started)
		t.stopped = make(chan struct{})
		atomic.StoreInt32(&t.stopping, 0)
	default:
		panic("not started, not stopped, what am i? how can i start?")
	}

	ch := make(chan error, 1)
	if err := t.listen(); err != nil {
		ch <- err
	} else {
		// TODO(yifan): Set read/write deadline.
		log.Infof("http transport listening on %v", t.listener.Addr())
		go func() {
			err := http.Serve(t.listener, t.mux)
			if atomic.CompareAndSwapInt32(&t.stopping, 1, 0) {
				ch <- nil
			} else {
				ch <- err
			}
		}()
	}
	return ch
}

// Stop stops the http transporter by closing the listener.
func (t *HTTPTransporter) Stop(graceful bool) error {
	t.lifeLock.Lock()
	defer t.lifeLock.Unlock()

	select {
	case <-t.stopped:
		// already stopped
		return nil
	case <-t.started:
		defer close(t.stopped)
		t.started = make(chan struct{})
	default:
		panic("not started, not stopped, what am i? how can i stop?")
	}
	//TODO(jdef) if graceful, wait for pending requests to terminate
	atomic.StoreInt32(&t.stopping, 1)
	err := t.listener.Close()
	return err
}

// UPID returns the upid of the transporter.
func (t *HTTPTransporter) UPID() *upid.UPID {
	return t.upid
}

func (t *HTTPTransporter) messageHandler(w http.ResponseWriter, r *http.Request) {
	// Verify it's a libprocess request.
	from, err := getLibprocessFrom(r)
	if err != nil {
		log.Errorf("Ignoring the request, because it's not a libprocess request: %v\n", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	data, err := ioutil.ReadAll(r.Body)
	if err != nil {
		log.Errorf("Failed to read HTTP body: %v\n", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	log.V(2).Infof("Receiving message from %v, length %v\n", from, len(data))
	w.WriteHeader(http.StatusAccepted)
	t.messageQueue <- &Message{
		UPID:  from,
		Name:  extractNameFromRequestURI(r.RequestURI),
		Bytes: data,
	}
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
	req.Header.Add("Libprocess-From", t.upid.String())
	req.Header.Add("Content-Type", "application/x-protobuf")
	req.Header.Add("Connection", "Keep-Alive")

	return req, nil
}

func getLibprocessFrom(r *http.Request) (*upid.UPID, error) {
	if r.Method != "POST" {
		return nil, fmt.Errorf("Not a POST request")
	}
	ua, ok := r.Header["User-Agent"]
	if ok && strings.HasPrefix(ua[0], "libprocess/") {
		// TODO(yifan): Just take the first field for now.
		return upid.Parse(ua[0][len("libprocess/"):])
	}
	lf, ok := r.Header["Libprocess-From"]
	if ok {
		// TODO(yifan): Just take the first field for now.
		return upid.Parse(lf[0])
	}
	return nil, fmt.Errorf("Cannot find 'User-Agent' or 'Libprocess-From'")
}
