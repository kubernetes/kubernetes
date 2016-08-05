// Copyright 2015 Google Inc. All Rights Reserved.
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

package netlink

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"unsafe"
)

/*
#cgo CFLAGS: -I/usr/include/libnl3
#cgo LDFLAGS: -lnl-3 -lnl-genl-3

#include <stdint.h>

#include <netlink/netlink.h>
#include <netlink/genl/genl.h>

int callbackGateway(struct nl_msg *msg, void *arg);
*/
import "C"

// Netlink message flags.
const (
	MFRequest  = C.NLM_F_REQUEST
	MFMulti    = C.NLM_F_MULTI
	MFACK      = C.NLM_F_ACK
	MFEcho     = C.NLM_F_ECHO
	MFDumpIntr = C.NLM_F_DUMP_INTR
	MFRoot     = C.NLM_F_ROOT
	MFMatch    = C.NLM_F_MATCH
	MFAtomic   = C.NLM_F_ATOMIC
	MFDump     = C.NLM_F_DUMP
	MFReplace  = C.NLM_F_REPLACE
	MFExcl     = C.NLM_F_EXCL
	MFCreate   = C.NLM_F_CREATE
	MFAppend   = C.NLM_F_APPEND
)

// CallbackFunc is a netlink message callback function.
type CallbackFunc func(*Message, interface{}) error

// callbackArg contains information passed for libnl callbacks.
type callbackArg struct {
	id  uintptr
	fn  CallbackFunc
	arg interface{}
	err error
}

var (
	nextCallbackID uintptr
	callbacks      = make(map[uintptr]*callbackArg)
	callbacksLock  sync.RWMutex
)

// registerCallback registers a callback and returns the allocated callback ID.
func registerCallback(cbArg *callbackArg) uintptr {
	callbacksLock.Lock()
	defer callbacksLock.Unlock()
	cbArg.id = nextCallbackID
	nextCallbackID++
	if _, ok := callbacks[cbArg.id]; ok {
		panic(fmt.Sprintf("Callback ID %d already in use", cbArg.id))
	}
	callbacks[cbArg.id] = cbArg
	return cbArg.id
}

// unregisterCallback unregisters a callback.
func unregisterCallback(cbArg *callbackArg) {
	callbacksLock.Lock()
	defer callbacksLock.Unlock()
	if _, ok := callbacks[cbArg.id]; !ok {
		panic(fmt.Sprintf("Callback ID %d not registered", cbArg.id))
	}
	delete(callbacks, cbArg.id)
}

// callback is the Go callback trampoline that is called from the
// callbackGateway C function, which in turn gets called from libnl.
//
//export callback
func callback(nlm *C.struct_nl_msg, nla unsafe.Pointer) C.int {
	cbID := uintptr(nla)
	callbacksLock.RLock()
	cbArg := callbacks[cbID]
	callbacksLock.RUnlock()

	if cbArg == nil {
		panic(fmt.Sprintf("No netlink callback with ID %d", cbID))
	}

	cbMsg := &Message{nlm: nlm}
	if err := cbArg.fn(cbMsg, cbArg.arg); err != nil {
		cbArg.err = err
		return C.NL_STOP
	}
	return C.NL_OK
}

func callbackDefault(msg *Message, arg interface{}) error {
	return nil
}

func callbackUnmarshal(msg *Message, arg interface{}) error {
	return msg.Unmarshal(arg)
}

// Message represents a netlink message.
type Message struct {
	nlm *C.struct_nl_msg
}

// NewMessage returns an initialised netlink message.
func NewMessage(command, family, flags int) (*Message, error) {
	nlm := C.nlmsg_alloc()
	if nlm == nil {
		return nil, errors.New("failed to create netlink message")
	}
	C.genlmsg_put(nlm, C.NL_AUTO_PID, C.NL_AUTO_SEQ, C.int(family), 0, C.int(flags), C.uint8_t(command), genlVersion)
	return &Message{nlm: nlm}, nil
}

// NewMessageFromBytes returns a netlink message that is initialised from the
// given byte slice.
func NewMessageFromBytes(nlb []byte) (*Message, error) {
	nlm := C.nlmsg_alloc_size(C.size_t(len(nlb)))
	if nlm == nil {
		return nil, errors.New("failed to create netlink message")
	}
	nlh := C.nlmsg_hdr(nlm)
	copy((*[1 << 20]byte)(unsafe.Pointer(nlh))[:len(nlb)], nlb)
	return &Message{nlm: nlm}, nil
}

// Free frees resources associated with a netlink message.
func (m *Message) Free() {
	C.nlmsg_free(m.nlm)
	m.nlm = nil
}

// Bytes returns the byte slice representation of a netlink message.
func (m *Message) Bytes() ([]byte, error) {
	if m.nlm == nil {
		return nil, errors.New("no netlink message")
	}
	nlh := C.nlmsg_hdr(m.nlm)
	nlb := make([]byte, nlh.nlmsg_len)
	copy(nlb, (*[1 << 20]byte)(unsafe.Pointer(nlh))[:nlh.nlmsg_len])
	return nlb, nil
}

// Marshal converts the given struct into a netlink message. Each field within
// the struct is added as netlink data, with structs and pointers to structs
// being recursively added as nested data.
//
// Supported data types and their netlink mappings are as follows:
//
//	uint8:		NLA_U8
//	uint16:		NLA_U16
//	uint32:		NLA_U32
//	uint64:		NLA_U64
//	string:		NLA_STRING
//	byte array:	NLA_UNSPEC
//	struct:		NLA_NESTED
//	net.IP:		NLA_UNSPEC
//
// Each field must have a `netlink' tag, which can contain the following
// comma separated options:
//
//	attr:x		specify the netlink attribute for this field (required)
//	network		value will be converted to its network byte order
//	omitempty	if the field has a zero value it will be omitted
//	optional	mark the field as being an optional (for unmarshalling)
//
// Other Go data types are unsupported and an error will be returned if one
// is encountered.
func (m *Message) Marshal(v interface{}) error {
	val := reflect.Indirect(reflect.ValueOf(v))
	if val.Kind() != reflect.Struct {
		return fmt.Errorf("%v is not a struct or a pointer to a struct", reflect.TypeOf(v))
	}
	return marshal(val, "", nil, m.nlm)
}

// Unmarshal parses the netlink message and fills the struct referenced by the
// given pointer. The supported data types and netlink encodings are the same
// as for Marshal.
func (m *Message) Unmarshal(v interface{}) error {
	val := reflect.Indirect(reflect.ValueOf(v))
	if val.Kind() != reflect.Struct || !val.CanSet() {
		return fmt.Errorf("%v is not a pointer to a struct", reflect.TypeOf(v))
	}
	maxAttrID, err := structMaxAttrID(val)
	if err != nil {
		return err
	}
	attrs, err := parseMessage(m.nlm, maxAttrID)
	if err != nil {
		return err
	}
	return unmarshal(val, "", nil, attrs)
}

// Send sends the netlink message.
func (m *Message) Send() error {
	return m.SendCallback(callbackDefault, nil)
}

// SendCallback sends the netlink message. The specified callback function
// will be called for each message that is received in response.
func (m *Message) SendCallback(fn CallbackFunc, arg interface{}) error {
	s, err := newSocket()
	if err != nil {
		return err
	}
	defer s.free()

	if errno := C.genl_connect(s.nls); errno != 0 {
		return &Error{errno, "failed to connect to netlink"}
	}
	defer C.nl_close(s.nls)

	cbArg := &callbackArg{fn: fn, arg: arg}
	cbID := registerCallback(cbArg)
	defer unregisterCallback(cbArg)

	if errno := C.nl_socket_modify_cb(s.nls, C.NL_CB_VALID, C.NL_CB_CUSTOM, (C.nl_recvmsg_msg_cb_t)(unsafe.Pointer(C.callbackGateway)), unsafe.Pointer(cbID)); errno != 0 {
		return &Error{errno, "failed to modify callback"}
	}
	// nl_send_auto_complete returns number of bytes sent or a negative
	// errno on failure.
	if errno := C.nl_send_auto_complete(s.nls, m.nlm); errno < 0 {
		return &Error{errno, "failed to send netlink message"}
	}
	if errno := C.nl_recvmsgs_default(s.nls); errno != 0 {
		return &Error{errno, "failed to receive messages"}
	}
	return nil
}

// SendMessage creates and sends a netlink message.
func SendMessage(command, family, flags int) error {
	return SendMessageCallback(command, family, flags, callbackDefault, nil)
}

// SendMessageCallback creates and sends a netlink message. The specified
// callback function will be called for each message that is received in
// response.
func SendMessageCallback(command, family, flags int, cb CallbackFunc, arg interface{}) error {
	msg, err := NewMessage(command, family, flags)
	if err != nil {
		return err
	}
	defer msg.Free()

	return msg.SendCallback(cb, arg)
}

// SendMessageMarshalled creates a netlink message and marshals the given
// struct into the message, before sending it.
func SendMessageMarshalled(command, family, flags int, v interface{}) error {
	msg, err := NewMessage(command, family, flags)
	if err != nil {
		return err
	}
	defer msg.Free()

	if err := msg.Marshal(v); err != nil {
		return err
	}
	return msg.Send()
}

// SendMessageUnmarshal creates and sends a netlink message. All messages
// received in response will be unmarshalled into the given struct.
func SendMessageUnmarshal(command, family, flags int, v interface{}) error {
	return SendMessageCallback(command, family, flags, callbackUnmarshal, v)
}
