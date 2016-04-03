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
	"fmt"
	"net"
	"reflect"
	"strconv"
	"sync"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosproto/scheduler"
	"github.com/mesos/mesos-go/mesosutil/process"
	"github.com/mesos/mesos-go/messenger/sessionid"
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

const (
	defaultQueueSize = 1024
)

// MessageHandler is the callback of the message. When the callback
// is invoked, the sender's upid and the message is passed to the callback.
type MessageHandler func(from *upid.UPID, pbMsg proto.Message)

// Messenger defines the interfaces that should be implemented.
type Messenger interface {
	Install(handler MessageHandler, msg proto.Message) error
	Send(ctx context.Context, upid *upid.UPID, msg proto.Message) error
	Route(ctx context.Context, from *upid.UPID, msg proto.Message) error
	Start() error
	Stop() error
	UPID() upid.UPID
}

type errorHandlerFunc func(context.Context, *Message, error) error
type dispatchFunc func(errorHandlerFunc)

// MesosMessenger is an implementation of the Messenger interface.
type MesosMessenger struct {
	upid              upid.UPID
	sendingQueue      chan dispatchFunc
	installedMessages map[string]reflect.Type
	installedHandlers map[string]MessageHandler
	stop              chan struct{}
	stopOnce          sync.Once
	tr                Transporter
	guardHandlers     sync.RWMutex // protect simultaneous changes to messages/handlers maps
}

// ForHostname creates a new default messenger (HTTP), using UPIDBindingAddress to
// determine the binding-address used for both the UPID.Host and Transport binding address.
func ForHostname(proc *process.Process, hostname string, bindingAddress net.IP, port uint16, publishedAddress net.IP) (Messenger, error) {
	upid := upid.UPID{
		ID:   proc.Label(),
		Port: strconv.Itoa(int(port)),
	}
	host, err := UPIDBindingAddress(hostname, bindingAddress)
	if err != nil {
		return nil, err
	}

	var publishedHost string
	if publishedAddress != nil {
		publishedHost, err = UPIDBindingAddress(hostname, publishedAddress)
		if err != nil {
			return nil, err
		}
	}

	if publishedHost != "" {
		upid.Host = publishedHost
	} else {
		upid.Host = host
	}

	return NewHttpWithBindingAddress(upid, bindingAddress), nil
}

// UPIDBindingAddress determines the value of UPID.Host that will be used to build
// a Transport. If a non-nil, non-wildcard bindingAddress is specified then it will be used
// for both the UPID and Transport binding address. Otherwise hostname is resolved to an IP
// address and the UPID.Host is set to that address and the bindingAddress is passed through
// to the Transport.
func UPIDBindingAddress(hostname string, bindingAddress net.IP) (string, error) {
	upidHost := ""
	if bindingAddress != nil && "0.0.0.0" != bindingAddress.String() {
		upidHost = bindingAddress.String()
	} else {
		if hostname == "" || hostname == "0.0.0.0" {
			return "", fmt.Errorf("invalid hostname (%q) specified with binding address %v", hostname, bindingAddress)
		}
		ip := net.ParseIP(hostname)
		if ip != nil {
			ip = ip.To4()
		}
		if ip == nil {
			ips, err := net.LookupIP(hostname)
			if err != nil {
				return "", err
			}
			// try to find an ipv4 and use that
			for _, addr := range ips {
				if ip = addr.To4(); ip != nil {
					break
				}
			}
			if ip == nil {
				// no ipv4? best guess, just take the first addr
				if len(ips) > 0 {
					ip = ips[0]
					log.Warningf("failed to find an IPv4 address for '%v', best guess is '%v'", hostname, ip)
				} else {
					return "", fmt.Errorf("failed to determine IP address for host '%v'", hostname)
				}
			}
		}
		upidHost = ip.String()
	}
	return upidHost, nil
}

// NewMesosMessenger creates a new mesos messenger.
func NewHttp(upid upid.UPID, opts ...httpOpt) *MesosMessenger {
	return NewHttpWithBindingAddress(upid, nil, opts...)
}

func NewHttpWithBindingAddress(upid upid.UPID, address net.IP, opts ...httpOpt) *MesosMessenger {
	return New(NewHTTPTransporter(upid, address, opts...))
}

func New(t Transporter) *MesosMessenger {
	return &MesosMessenger{
		sendingQueue:      make(chan dispatchFunc, defaultQueueSize),
		installedMessages: make(map[string]reflect.Type),
		installedHandlers: make(map[string]MessageHandler),
		tr:                t,
	}
}

/// Install installs the handler with the given message.
func (m *MesosMessenger) Install(handler MessageHandler, msg proto.Message) error {
	// Check if the message is a pointer.
	mtype := reflect.TypeOf(msg)
	if mtype.Kind() != reflect.Ptr {
		return fmt.Errorf("Message %v is not a Ptr type", msg)
	}

	// Check if the message is already installed.
	name := getMessageName(msg)
	if _, ok := m.installedMessages[name]; ok {
		return fmt.Errorf("Message %v is already installed", name)
	}

	m.guardHandlers.Lock()
	defer m.guardHandlers.Unlock()

	m.installedMessages[name] = mtype.Elem()
	m.installedHandlers[name] = handler
	m.tr.Install(name)
	return nil
}

// Send puts a message into the outgoing queue, waiting to be sent.
// With buffered channels, this will not block under moderate throughput.
// When an error is generated, the error can be communicated by placing
// a message on the incoming queue to be handled upstream.
func (m *MesosMessenger) Send(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	if upid == nil {
		panic("cannot sent a message to a nil pid")
	} else if *upid == m.upid {
		return fmt.Errorf("Send the message to self")
	}

	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}

	name := getMessageName(msg)
	log.V(2).Infof("Sending message %v to %v\n", name, upid)

	wrapped := &Message{upid, name, msg, b}
	d := dispatchFunc(func(rf errorHandlerFunc) {
		err := m.tr.Send(ctx, wrapped)
		err = rf(ctx, wrapped, err)
		if err != nil {
			m.reportError("send", wrapped, err)
		}
	})
	select {
	case <-ctx.Done():
		return ctx.Err()
	case m.sendingQueue <- d:
		return nil
	}
}

// Route puts a message either in the incoming or outgoing queue.
// This method is useful for:
// 1) routing internal error to callback handlers
// 2) testing components without starting remote servers.
func (m *MesosMessenger) Route(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	if upid == nil {
		panic("cannot route a message to a nil pid")
	} else if *upid != m.upid {
		// if destination is not self, send to outbound.
		return m.Send(ctx, upid, msg)
	}

	name := getMessageName(msg)
	log.V(2).Infof("routing message %q to self", name)

	_, handler, ok := m.messageBinding(name)
	if !ok {
		return fmt.Errorf("failed to route message, no message binding for %q", name)
	}

	// the implication of this is that messages can be delivered to self even if the
	// messenger has been stopped. is that OK?
	go handler(upid, msg)
	return nil
}

// Start starts the messenger; expects to be called once and only once.
func (m *MesosMessenger) Start() error {

	m.stop = make(chan struct{})

	pid, errChan := m.tr.Start()
	if pid == (upid.UPID{}) {
		err := <-errChan
		return fmt.Errorf("failed to start messenger: %v", err)
	}

	// the pid that we're actually bound as
	m.upid = pid

	go m.sendLoop()
	go m.decodeLoop()

	// wait for a listener error or a stop signal; either way stop the messenger

	// TODO(jdef) a better implementation would attempt to re-listen; need to coordinate
	// access to m.upid in that case. probably better off with a state machine instead of
	// what we have now.
	go func() {
		select {
		case err := <-errChan:
			if err != nil {
				//TODO(jdef) should the driver abort in this case? probably
				//since this messenger will never attempt to re-establish the
				//transport
				log.Errorln("transport stopped unexpectedly:", err.Error())
			}
			err = m.Stop()
			if err != nil && err != errTerminal {
				log.Errorln("failed to stop messenger cleanly: ", err.Error())
			}
		case <-m.stop:
		}
	}()
	return nil
}

// Stop stops the messenger and clean up all the goroutines.
func (m *MesosMessenger) Stop() (err error) {
	m.stopOnce.Do(func() {
		select {
		case <-m.stop:
		default:
			defer close(m.stop)
		}

		log.Infof("stopping messenger %v..", m.upid)

		//TODO(jdef) don't hardcode the graceful flag here
		if err2 := m.tr.Stop(true); err2 != nil && err2 != errTerminal {
			log.Warningf("failed to stop the transporter: %v\n", err2)
			err = err2
		}
	})
	return
}

// UPID returns the upid of the messenger.
func (m *MesosMessenger) UPID() upid.UPID {
	return m.upid
}

func (m *MesosMessenger) reportError(action string, msg *Message, err error) {
	// log message transmission errors but don't shoot the messenger.
	// this approach essentially drops all undelivered messages on the floor.
	name := ""
	if msg != nil {
		name = msg.Name
	}
	log.Errorf("failed to %s message %q: %+v", action, name, err)
}

func (m *MesosMessenger) sendLoop() {
	for {
		select {
		case <-m.stop:
			return
		case f := <-m.sendingQueue:
			f(errorHandlerFunc(func(ctx context.Context, msg *Message, err error) error {
				if _, ok := err.(*networkError); ok {
					// if transport reports a network error, then
					// we're probably disconnected from the remote process?
					pid := msg.UPID.String()
					neterr := &mesos.InternalNetworkError{Pid: &pid}
					sessionID, ok := sessionid.FromContext(ctx)
					if ok {
						neterr.Session = &sessionID
					}
					log.V(1).Infof("routing network error for pid %q session %q", pid, sessionID)
					err2 := m.Route(ctx, &m.upid, neterr)
					if err2 != nil {
						log.Error(err2)
					} else {
						log.V(1).Infof("swallowing raw error because we're reporting a networkError: %v", err)
						return nil
					}
				}
				return err
			}))
		}
	}
}

// Since HTTPTransporter.Recv() is already buffered, so we don't need a 'recvLoop' here.
func (m *MesosMessenger) decodeLoop() {
	for {
		select {
		case <-m.stop:
			return
		default:
		}
		msg, err := m.tr.Recv()
		if err != nil {
			if err == discardOnStopError {
				log.V(1).Info("exiting decodeLoop, transport shutting down")
				return
			} else {
				panic(fmt.Sprintf("unexpected transport error: %v", err))
			}
		}

		log.V(2).Infof("Receiving message %v from %v\n", msg.Name, msg.UPID)
		protoMessage, handler, found := m.messageBinding(msg.Name)
		if !found {
			log.Warningf("no message binding for message %q", msg.Name)
			continue
		}

		msg.ProtoMessage = protoMessage
		if err := proto.Unmarshal(msg.Bytes, msg.ProtoMessage); err != nil {
			log.Errorf("Failed to unmarshal message %v: %v\n", msg, err)
			continue
		}

		handler(msg.UPID, msg.ProtoMessage)
	}
}

func (m *MesosMessenger) messageBinding(name string) (proto.Message, MessageHandler, bool) {
	m.guardHandlers.RLock()
	defer m.guardHandlers.RUnlock()

	gotype, ok := m.installedMessages[name]
	if !ok {
		return nil, nil, false
	}

	handler, ok := m.installedHandlers[name]
	if !ok {
		return nil, nil, false
	}

	protoMessage := reflect.New(gotype).Interface().(proto.Message)
	return protoMessage, handler, true
}

// getMessageName returns the name of the message in the mesos manner.
func getMessageName(msg proto.Message) string {
	var msgName string

	switch msg := msg.(type) {
	case *scheduler.Call:
		msgName = "scheduler"
	default:
		msgName = fmt.Sprintf("%v.%v", "mesos.internal", reflect.TypeOf(msg).Elem().Name())
	}

	return msgName
}
