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
	"github.com/mesos/mesos-go/mesosutil/process"
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

// MesosMessenger is an implementation of the Messenger interface.
type MesosMessenger struct {
	upid              upid.UPID
	encodingQueue     chan *Message
	sendingQueue      chan *Message
	installedMessages map[string]reflect.Type
	installedHandlers map[string]MessageHandler
	stop              chan struct{}
	stopOnce          sync.Once
	tr                Transporter
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
func NewHttp(upid upid.UPID) *MesosMessenger {
	return NewHttpWithBindingAddress(upid, nil)
}

func NewHttpWithBindingAddress(upid upid.UPID, address net.IP) *MesosMessenger {
	return New(upid, NewHTTPTransporter(upid, address))
}

func New(upid upid.UPID, t Transporter) *MesosMessenger {
	return &MesosMessenger{
		encodingQueue:     make(chan *Message, defaultQueueSize),
		sendingQueue:      make(chan *Message, defaultQueueSize),
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
	name := getMessageName(msg)
	log.V(2).Infof("Sending message %v to %v\n", name, upid)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case m.encodingQueue <- &Message{upid, name, msg, nil}:
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

	// TODO(jdef) this has an unfortunate performance impact for self-messaging. implement
	// something more reasonable here.
	data, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	name := getMessageName(msg)
	return m.tr.Inject(ctx, &Message{upid, name, msg, data})
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
	go m.encodeLoop()
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

		log.Info("stopping messenger..")

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

func (m *MesosMessenger) encodeLoop() {
	for {
		select {
		case <-m.stop:
			return
		case msg := <-m.encodingQueue:
			e := func() error {
				//TODO(jdef) implement timeout for context
				ctx, cancel := context.WithCancel(context.TODO())
				defer cancel()

				b, err := proto.Marshal(msg.ProtoMessage)
				if err != nil {
					return err
				}
				msg.Bytes = b
				select {
				case <-ctx.Done():
					return ctx.Err()
				case m.sendingQueue <- msg:
					return nil
				}
			}()
			if e != nil {
				m.reportError(fmt.Errorf("Failed to enqueue message %v: %v", msg, e))
			}
		}
	}
}

func (m *MesosMessenger) reportError(err error) {
	log.V(2).Info(err)
	//TODO(jdef) implement timeout for context
	ctx, cancel := context.WithCancel(context.TODO())
	defer cancel()

	c := make(chan error, 1)
	pid := m.upid
	go func() { c <- m.Route(ctx, &pid, &mesos.FrameworkErrorMessage{Message: proto.String(err.Error())}) }()
	select {
	case <-ctx.Done():
		<-c // wait for Route to return
	case e := <-c:
		if e != nil {
			log.Errorf("failed to report error %v due to: %v", err, e)
		}
	}
}

func (m *MesosMessenger) sendLoop() {
	for {
		select {
		case <-m.stop:
			return
		case msg := <-m.sendingQueue:
			e := func() error {
				//TODO(jdef) implement timeout for context
				ctx, cancel := context.WithCancel(context.TODO())
				defer cancel()

				c := make(chan error, 1)
				go func() { c <- m.tr.Send(ctx, msg) }()

				select {
				case <-ctx.Done():
					// Transport layer must use the context to detect cancelled requests.
					<-c // wait for Send to return
					return ctx.Err()
				case err := <-c:
					return err
				}
			}()
			if e != nil {
				m.reportError(fmt.Errorf("Failed to send message %v: %v", msg.Name, e))
			}
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
		msg.ProtoMessage = reflect.New(m.installedMessages[msg.Name]).Interface().(proto.Message)
		if err := proto.Unmarshal(msg.Bytes, msg.ProtoMessage); err != nil {
			log.Errorf("Failed to unmarshal message %v: %v\n", msg, err)
			continue
		}
		// TODO(yifan): Catch panic.
		m.installedHandlers[msg.Name](msg.UPID, msg.ProtoMessage)
	}
}

// getMessageName returns the name of the message in the mesos manner.
func getMessageName(msg proto.Message) string {
	return fmt.Sprintf("%v.%v", "mesos.internal", reflect.TypeOf(msg).Elem().Name())
}
