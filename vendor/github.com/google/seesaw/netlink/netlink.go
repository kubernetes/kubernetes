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

// Package netlink provides a Go interface to netlink via the libnl library.
package netlink

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"reflect"
	"strconv"
	"strings"
	"unsafe"
)

/*
#cgo CFLAGS: -I/usr/include/libnl3
#cgo LDFLAGS: -lnl-3 -lnl-genl-3

#include <stdint.h>

#include <netlink/netlink.h>
#include <netlink/genl/genl.h>
#include <netlink/genl/ctrl.h>
*/
import "C"

const (
	genlVersion = 1
	nlMaxBytes  = 512
)

// uint16FromNetwork converts the given value from its network byte order.
func uint16FromNetwork(u uint16) uint16 {
	b := *(*[2]byte)(unsafe.Pointer(&u))
	return binary.BigEndian.Uint16(b[:])
}

// uint16ToNetwork converts the given value to its network byte order.
func uint16ToNetwork(u uint16) uint16 {
	var b [2]byte
	binary.BigEndian.PutUint16(b[:], u)
	return *(*uint16)(unsafe.Pointer(&b))
}

// uint32FromNetwork converts the given value from its network byte order.
func uint32FromNetwork(u uint32) uint32 {
	b := *(*[4]byte)(unsafe.Pointer(&u))
	return binary.BigEndian.Uint32(b[:])
}

// uint32ToNetwork converts the given value to its network byte order.
func uint32ToNetwork(u uint32) uint32 {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], u)
	return *(*uint32)(unsafe.Pointer(&b))
}

// uint64FromNetwork converts the given value from its network byte order.
func uint64FromNetwork(u uint64) uint64 {
	b := *(*[8]byte)(unsafe.Pointer(&u))
	return binary.BigEndian.Uint64(b[:])
}

// uint64ToNetwork converts the given value to its network byte order.
func uint64ToNetwork(u uint64) uint64 {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], u)
	return *(*uint64)(unsafe.Pointer(&b))
}

type fieldParams struct {
	attr      uint16
	network   bool
	omitempty bool
	optional  bool
}

func parseFieldParams(params string) (*fieldParams, error) {
	if params == "" {
		return nil, nil
	}
	fp := &fieldParams{}
	for _, param := range strings.Split(params, ",") {
		parts := strings.Split(param, ":")
		switch name := parts[0]; name {
		case "attr":
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid field parameter %q", param)
			}
			val := parts[1]
			n, err := strconv.ParseUint(val, 10, 16)
			if err != nil {
				return nil, fmt.Errorf("invalid value %q for field parameter %s", val, name)
			}
			fp.attr = uint16(n)

		case "network":
			fp.network = true

		case "omitempty":
			fp.omitempty = true

		case "optional":
			fp.optional = true

		default:
			return nil, fmt.Errorf("unknown field parameter %q", name)
		}
	}
	return fp, nil
}

// structMaxAttrID returns the maximum attribute ID found on netlink tagged
// fields within the given struct and any untagged structs it contains.
func structMaxAttrID(v reflect.Value) (uint16, error) {
	if v.Kind() != reflect.Struct {
		return 0, fmt.Errorf("%v is not a struct", v.Type())
	}
	st := v.Type()

	var maxAttrID uint16
	for i := 0; i < st.NumField(); i++ {
		ft, fv := st.Field(i), v.Field(i)
		fp, err := parseFieldParams(ft.Tag.Get("netlink"))
		if err != nil {
			return 0, err
		}
		if fp != nil && fp.attr > maxAttrID {
			maxAttrID = fp.attr
		}
		if fp == nil && fv.Kind() == reflect.Struct {
			attrID, err := structMaxAttrID(fv)
			if err != nil {
				return 0, err
			}
			if attrID > maxAttrID {
				maxAttrID = attrID
			}
		}
	}
	return maxAttrID, nil
}

func xcalloc(n, size uint) (unsafe.Pointer, error) {
	if n < 1 || size < 1 {
		return nil, errors.New("invalid allocation size")
	}
	// TODO(jsing): We should really use C.SIZE_MAX here, however that
	// currently translates to -1 via cgo...
	if ^C.size_t(0)/C.size_t(n) < C.size_t(size) {
		return nil, errors.New("integer overflow")
	}
	m := C.calloc(C.size_t(n), C.size_t(size))
	if m == nil {
		return nil, errors.New("out of memory")
	}
	return m, nil
}

type attribute struct {
	nla *C.struct_nlattr
}

type nlAttrs struct {
	attrs **C.struct_nlattr
	n     uint
}

func newNLAttrs(n uint) (*nlAttrs, error) {
	size := unsafe.Sizeof(&C.struct_nlattr{})
	m, err := xcalloc(n, uint(size))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate netlink attributes: %v", err)
	}
	return &nlAttrs{attrs: (**C.struct_nlattr)(m), n: n}, nil
}

func (a nlAttrs) free() {
	if a.attrs == nil {
		return
	}
	C.free(unsafe.Pointer(a.attrs))
	a.attrs = nil
}

func (a nlAttrs) attributes() map[uint16]*attribute {
	if a.attrs == nil {
		return nil
	}
	attrs := make(map[uint16]*attribute)
	for _, nla := range (*[1 << 20]*C.struct_nlattr)(unsafe.Pointer(a.attrs))[:a.n] {
		if nla == nil {
			continue
		}
		attrs[uint16(nla.nla_type)] = &attribute{nla: nla}
	}
	return attrs
}

func parseMessage(nlm *C.struct_nl_msg, maxAttrID uint16) (map[uint16]*attribute, error) {
	if nlm == nil {
		return nil, errors.New("no netlink message")
	}

	nlas, err := newNLAttrs(uint(maxAttrID) + 1)
	if err != nil {
		return nil, err
	}
	defer nlas.free()

	var nlaPolicy *C.struct_nla_policy
	if errno := C.genlmsg_parse(C.nlmsg_hdr(nlm), 0, nlas.attrs, C.int(maxAttrID), nlaPolicy); errno != 0 {
		return nil, &Error{errno, "failed to parse netlink message"}
	}

	return nlas.attributes(), nil
}

func parseNested(attr *attribute, maxAttrID uint16) (map[uint16]*attribute, error) {
	nlas, err := newNLAttrs(uint(maxAttrID) + 1)
	if err != nil {
		return nil, err
	}
	defer nlas.free()

	var nlaPolicy *C.struct_nla_policy
	if errno := C.nla_parse_nested(nlas.attrs, C.int(maxAttrID), attr.nla, nlaPolicy); errno != 0 {
		return nil, errors.New("failed to parse netlink nested attribute")
	}

	return nlas.attributes(), nil
}

var (
	netIPType = reflect.TypeOf(net.IP{})
)

func marshal(v reflect.Value, field string, params *fieldParams, nlm *C.struct_nl_msg) error {
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		v = v.Elem()
	}

	if !v.CanInterface() {
		return fmt.Errorf("field %s is unexported", field)
	}

	if params != nil && params.omitempty {
		switch v.Kind() {
		case reflect.Array, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.String:
			if v.Interface() == reflect.Zero(v.Type()).Interface() {
				return nil
			}
		}
	}

	if nl, ok := v.Interface().(Marshaler); ok {
		b := nl.Bytes()
		if len(b) > nlMaxBytes {
			return fmt.Errorf("field %s is %d bytes - exceeds maximum of %d", field, len(b), nlMaxBytes)
		}
		var d [nlMaxBytes]byte
		copy(d[:], b)
		if errno := C.nla_put(nlm, C.int(params.attr), C.int(len(b)), unsafe.Pointer(&d)); errno != 0 {
			return &Error{errno, "failed to put data"}
		}
		return nil
	}

	switch v.Type() {
	case netIPType:
		var d [net.IPv6len]byte
		ip := net.IP(v.Bytes())
		if ip4 := ip.To4(); ip4 != nil {
			copy(d[:], ip4)
		} else {
			copy(d[:], ip)
		}
		if errno := C.nla_put(nlm, C.int(params.attr), C.int(len(d)), unsafe.Pointer(&d)); errno != 0 {
			return &Error{errno, "failed to put IP address"}
		}
		return nil
	}

	switch v.Kind() {
	case reflect.Array:
		if k := v.Type().Elem().Kind(); k != reflect.Uint8 {
			return fmt.Errorf("field %s is an array of unsupported type %v", field, k)
		}
		// Passing down the array pointer directly can trip
		// the cgo pointer checks if the array is part of a
		// larger struct that contains pointers, since cgo
		// can't tell that we are only passing the array.
		// Avoid the checks by making a copy.
		vl := v.Len()
		b := make([]byte, vl)
		for i := 0; i < vl; i++ {
			b[i] = byte(v.Index(i).Uint())
		}
		if errno := C.nla_put(nlm, C.int(params.attr), C.int(vl), unsafe.Pointer(&b[0])); errno != 0 {
			return &Error{errno, "failed to put data"}
		}

	case reflect.Uint8:
		if errno := C.nla_put_u8(nlm, C.int(params.attr), C.uint8_t(v.Uint())); errno != 0 {
			return &Error{errno, "failed to put u8"}
		}

	case reflect.Uint16:
		uval := uint16(v.Uint())
		if params.network {
			uval = uint16ToNetwork(uval)
		}
		if errno := C.nla_put_u16(nlm, C.int(params.attr), C.uint16_t(uval)); errno != 0 {
			return &Error{errno, "failed to put u16"}
		}

	case reflect.Uint32:
		uval := uint32(v.Uint())
		if params.network {
			uval = uint32ToNetwork(uval)
		}
		if errno := C.nla_put_u32(nlm, C.int(params.attr), C.uint32_t(uval)); errno != 0 {
			return &Error{errno, "failed to put u32"}
		}

	case reflect.Uint64:
		uval := uint64(v.Uint())
		if params.network {
			uval = uint64ToNetwork(uval)
		}
		if errno := C.nla_put_u64(nlm, C.int(params.attr), C.uint64_t(uval)); errno != 0 {
			return &Error{errno, "failed to put u64"}
		}

	case reflect.String:
		cs := C.CString(v.String())
		if cs == nil {
			return errors.New("failed to allocate string")
		}
		defer C.free(unsafe.Pointer(cs))
		if errno := C.nla_put_string(nlm, C.int(params.attr), cs); errno != 0 {
			return &Error{errno, "failed to put string"}
		}

	case reflect.Struct:
		var nla *C.struct_nlattr
		if params != nil {
			if nla = C.nla_nest_start(nlm, C.int(params.attr)); nla == nil {
				return errors.New("failed to start nested attribute")
			}
		}
		st := v.Type()
		for i := 0; i < st.NumField(); i++ {
			ft, fv := st.Field(i), v.Field(i)
			fp, err := parseFieldParams(ft.Tag.Get("netlink"))
			if err != nil {
				return err
			}
			if fp == nil && fv.Kind() != reflect.Struct {
				continue
			}
			if err := marshal(fv, ft.Name, fp, nlm); err != nil {
				return err
			}
		}
		if nla != nil {
			C.nla_nest_end(nlm, nla)
		}

	default:
		return fmt.Errorf("field %s has unsupported type %v (%v)", field, v.Type(), v.Kind())
	}
	return nil
}

func unmarshal(v reflect.Value, field string, params *fieldParams, attrs map[uint16]*attribute) error {
	var attr *attribute
	if params != nil {
		attr = attrs[params.attr]
		if attr == nil {
			if !params.optional {
				return fmt.Errorf("missing attribute for required field %s", field)
			}
			return nil
		}
	}

	if v.Kind() != reflect.Struct && !v.CanSet() {
		return fmt.Errorf("field %s is unsettable", field)
	}

	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return nil
		}
		v = v.Elem()
	}

	va := v.Addr()
	var unmarshaler Unmarshaler
	if va.Type().Implements(reflect.TypeOf(&unmarshaler).Elem()) {
		if !v.CanSet() {
			return fmt.Errorf("field %s implements Unmarshaler but is unsettable", field)
		}
		nl := va.Interface().(Unmarshaler)
		b := C.GoBytes(unsafe.Pointer(C.nla_data(attr.nla)), C.int(attr.nla.nla_len))
		nl.SetBytes(b)
		return nil
	}

	switch v.Type() {
	case netIPType:
		ip := net.IP(C.GoBytes(unsafe.Pointer(C.nla_data(attr.nla)), C.int(attr.nla.nla_len)))
		if len(ip) < net.IPv6len {
			return fmt.Errorf("IP address attribute for field %s is %d bytes (want at least %d)", field, len(ip), net.IPv6len)
		}
		ip = ip[:net.IPv6len]
		if bytes.Equal(ip[net.IPv4len:], make([]byte, net.IPv6len-net.IPv4len)) {
			ip = net.IPv4(ip[0], ip[1], ip[2], ip[3])
		}
		v.SetBytes(ip)
		return nil
	}

	switch v.Kind() {
	case reflect.Array:
		if k := v.Type().Elem().Kind(); k != reflect.Uint8 {
			return fmt.Errorf("field %s is an array of unsupported type %v", field, k)
		}
		b := (*[1 << 20]byte)(unsafe.Pointer(v.UnsafeAddr()))[:v.Len()]
		d := C.GoBytes(unsafe.Pointer(C.nla_data(attr.nla)), C.int(attr.nla.nla_len))
		copy(b, d)

	case reflect.Uint8:
		v.SetUint(uint64(C.nla_get_u8(attr.nla)))

	case reflect.Uint16:
		uval := uint16(C.nla_get_u16(attr.nla))
		if params.network {
			uval = uint16FromNetwork(uval)
		}
		v.SetUint(uint64(uval))

	case reflect.Uint32:
		uval := uint32(C.nla_get_u32(attr.nla))
		if params.network {
			uval = uint32FromNetwork(uval)
		}
		v.SetUint(uint64(uval))

	case reflect.Uint64:
		uval := uint64(C.nla_get_u64(attr.nla))
		if params.network {
			uval = uint64FromNetwork(uval)
		}
		v.SetUint(uint64(uval))

	case reflect.String:
		v.SetString(C.GoString(C.nla_get_string(attr.nla)))

	case reflect.Struct:
		if attr != nil {
			maxAttrID, err := structMaxAttrID(v)
			if err != nil {
				return err
			}
			attrs, err = parseNested(attr, maxAttrID)
			if err != nil {
				return err
			}
		}

		st := v.Type()
		for i := 0; i < st.NumField(); i++ {
			ft, fv := st.Field(i), v.Field(i)
			fp, err := parseFieldParams(ft.Tag.Get("netlink"))
			if err != nil {
				return err
			}
			if fp == nil && fv.Kind() != reflect.Struct {
				continue
			}
			if fv.Kind() == reflect.Ptr && fv.IsNil() && attrs[fp.attr] != nil {
				fv.Set(reflect.New(ft.Type.Elem()))
			}
			if err := unmarshal(fv, ft.Name, fp, attrs); err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("field %s has unsupported type %v (%v)", field, v.Type(), v.Kind())
	}
	return nil
}

type socket struct {
	nls *C.struct_nl_sock
}

func newSocket() (*socket, error) {
	nls := C.nl_socket_alloc()
	if nls == nil {
		return nil, errors.New("failed to allocate netlink socket")
	}
	return &socket{nls: nls}, nil
}

func (s *socket) free() {
	C.nl_socket_free(s.nls)
	s.nls = nil
}

// Error represents a netlink error.
type Error struct {
	errno C.int
	msg   string
}

// Error returns the string representation of a netlink error.
func (e *Error) Error() string {
	nle := C.GoString(C.nl_geterror(e.errno))
	return fmt.Sprintf("%s: %s", e.msg, strings.ToLower(nle))
}

// Family returns the family identifier for the specified family name.
func Family(name string) (int, error) {
	s, err := newSocket()
	if err != nil {
		return -1, err
	}
	defer s.free()

	if errno := C.genl_connect(s.nls); errno != 0 {
		return -1, &Error{errno, "failed to connect to netlink"}
	}
	defer C.nl_close((*C.struct_nl_sock)(s.nls))

	cn := C.CString(name)
	defer C.free(unsafe.Pointer(cn))
	family := C.genl_ctrl_resolve(s.nls, cn)
	if family < 0 {
		return -1, errors.New("failed to resolve family name")
	}
	return int(family), nil
}

// Marshaler represents a type that is capable of marshaling itself into its
// netlink representation.
type Marshaler interface {
	// Bytes returns a byte slice containing the netlink representation of
	// the given type.
	Bytes() []byte
}

// Unmarshaler represents a type that is capable of unmarshaling itself from
// its netlink representation.
type Unmarshaler interface {
	// SetBytes sets the value of the given type from a byte slice that
	// contains its netlink representation.
	SetBytes([]byte)
}
