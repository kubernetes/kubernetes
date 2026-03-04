/*
   Copyright The containerd Authors.

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

package typeurl

import (
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"reflect"
	"sync"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/known/anypb"
)

var (
	mu       sync.RWMutex
	registry = make(map[reflect.Type]string)
	handlers []handler
)

type handler interface {
	Marshaller(interface{}) func() ([]byte, error)
	Unmarshaller(interface{}) func([]byte) error
	TypeURL(interface{}) string
	GetType(url string) (reflect.Type, bool)
}

// Definitions of common error types used throughout typeurl.
//
// These error types are used with errors.Wrap and errors.Wrapf to add context
// to an error.
//
// To detect an error class, use errors.Is() functions to tell whether an
// error is of this type.

var (
	ErrNotFound = errors.New("not found")
)

// Any contains an arbitrary protcol buffer message along with its type.
//
// While there is google.golang.org/protobuf/types/known/anypb.Any,
// we'd like to have our own to hide the underlying protocol buffer
// implementations from containerd clients.
//
// https://developers.google.com/protocol-buffers/docs/proto3#any
type Any interface {
	// GetTypeUrl returns a URL/resource name that uniquely identifies
	// the type of the serialized protocol buffer message.
	GetTypeUrl() string

	// GetValue returns a valid serialized protocol buffer of the type that
	// GetTypeUrl() indicates.
	GetValue() []byte
}

type anyType struct {
	typeURL string
	value   []byte
}

func (a *anyType) GetTypeUrl() string {
	if a == nil {
		return ""
	}
	return a.typeURL
}

func (a *anyType) GetValue() []byte {
	if a == nil {
		return nil
	}
	return a.value
}

// Register a type with a base URL for JSON marshaling. When the MarshalAny and
// UnmarshalAny functions are called they will treat the Any type value as JSON.
// To use protocol buffers for handling the Any value the proto.Register
// function should be used instead of this function.
func Register(v interface{}, args ...string) {
	var (
		t = tryDereference(v)
		p = path.Join(args...)
	)
	mu.Lock()
	defer mu.Unlock()
	if et, ok := registry[t]; ok {
		if et != p {
			panic(fmt.Errorf("type registered with alternate path %q != %q", et, p))
		}
		return
	}
	registry[t] = p
}

// TypeURL returns the type url for a registered type.
func TypeURL(v interface{}) (string, error) {
	mu.RLock()
	u, ok := registry[tryDereference(v)]
	mu.RUnlock()
	if !ok {
		switch t := v.(type) {
		case proto.Message:
			return string(t.ProtoReflect().Descriptor().FullName()), nil
		default:
			for _, h := range handlers {
				if u := h.TypeURL(v); u != "" {
					return u, nil
				}
			}
			return "", fmt.Errorf("type %s: %w", reflect.TypeOf(v), ErrNotFound)
		}
	}
	return u, nil
}

// Is returns true if the type of the Any is the same as v.
func Is(any Any, v interface{}) bool {
	if any == nil {
		return false
	}
	// call to check that v is a pointer
	tryDereference(v)
	url, err := TypeURL(v)
	if err != nil {
		return false
	}
	return any.GetTypeUrl() == url
}

// MarshalAny marshals the value v into an any with the correct TypeUrl.
// If the provided object is already a proto.Any message, then it will be
// returned verbatim. If it is of type proto.Message, it will be marshaled as a
// protocol buffer. Otherwise, the object will be marshaled to json.
func MarshalAny(v interface{}) (Any, error) {
	var marshal func(v interface{}) ([]byte, error)
	switch t := v.(type) {
	case Any:
		// avoid reserializing the type if we have an any.
		return t, nil
	case proto.Message:
		marshal = func(v interface{}) ([]byte, error) {
			return proto.Marshal(t)
		}
	default:
		for _, h := range handlers {
			if m := h.Marshaller(v); m != nil {
				marshal = func(v interface{}) ([]byte, error) {
					return m()
				}
				break
			}
		}

		if marshal == nil {
			marshal = json.Marshal
		}
	}

	url, err := TypeURL(v)
	if err != nil {
		return nil, err
	}

	data, err := marshal(v)
	if err != nil {
		return nil, err
	}
	return &anyType{
		typeURL: url,
		value:   data,
	}, nil
}

// UnmarshalAny unmarshals the any type into a concrete type.
func UnmarshalAny(any Any) (interface{}, error) {
	return UnmarshalByTypeURL(any.GetTypeUrl(), any.GetValue())
}

// UnmarshalByTypeURL unmarshals the given type and value to into a concrete type.
func UnmarshalByTypeURL(typeURL string, value []byte) (interface{}, error) {
	return unmarshal(typeURL, value, nil)
}

// UnmarshalTo unmarshals the any type into a concrete type passed in the out
// argument. It is identical to UnmarshalAny, but lets clients provide a
// destination type through the out argument.
func UnmarshalTo(any Any, out interface{}) error {
	return UnmarshalToByTypeURL(any.GetTypeUrl(), any.GetValue(), out)
}

// UnmarshalToByTypeURL unmarshals the given type and value into a concrete type passed
// in the out argument. It is identical to UnmarshalByTypeURL, but lets clients
// provide a destination type through the out argument.
func UnmarshalToByTypeURL(typeURL string, value []byte, out interface{}) error {
	_, err := unmarshal(typeURL, value, out)
	return err
}

// MarshalProto converts typeurl.Any to google.golang.org/protobuf/types/known/anypb.Any.
func MarshalProto(from Any) *anypb.Any {
	if from == nil {
		return nil
	}

	if pbany, ok := from.(*anypb.Any); ok {
		return pbany
	}

	return &anypb.Any{
		TypeUrl: from.GetTypeUrl(),
		Value:   from.GetValue(),
	}
}

// MarshalAnyToProto converts an arbitrary interface to google.golang.org/protobuf/types/known/anypb.Any.
func MarshalAnyToProto(from interface{}) (*anypb.Any, error) {
	anyType, err := MarshalAny(from)
	if err != nil {
		return nil, err
	}
	return MarshalProto(anyType), nil
}

func unmarshal(typeURL string, value []byte, v interface{}) (interface{}, error) {
	t, isProto, err := getTypeByUrl(typeURL)
	if err != nil {
		return nil, err
	}

	if v == nil {
		v = reflect.New(t).Interface()
	} else {
		// Validate interface type provided by client
		vURL, err := TypeURL(v)
		if err != nil {
			return nil, err
		}
		if typeURL != vURL {
			return nil, fmt.Errorf("can't unmarshal type %q to output %q", typeURL, vURL)
		}
	}

	if isProto {
		pm, ok := v.(proto.Message)
		if ok {
			return v, proto.Unmarshal(value, pm)
		}

		for _, h := range handlers {
			if unmarshal := h.Unmarshaller(v); unmarshal != nil {
				return v, unmarshal(value)
			}
		}
	}

	// fallback to json unmarshaller
	return v, json.Unmarshal(value, v)
}

func getTypeByUrl(url string) (_ reflect.Type, isProto bool, _ error) {
	mu.RLock()
	for t, u := range registry {
		if u == url {
			mu.RUnlock()
			return t, false, nil
		}
	}
	mu.RUnlock()
	mt, err := protoregistry.GlobalTypes.FindMessageByURL(url)
	if err != nil {
		if errors.Is(err, protoregistry.NotFound) {
			for _, h := range handlers {
				if t, isProto := h.GetType(url); t != nil {
					return t, isProto, nil
				}
			}
		}
		return nil, false, fmt.Errorf("type with url %s: %w", url, ErrNotFound)
	}
	empty := mt.New().Interface()
	return reflect.TypeOf(empty).Elem(), true, nil
}

func tryDereference(v interface{}) reflect.Type {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		// require check of pointer but dereference to register
		return t.Elem()
	}
	panic("v is not a pointer to a type")
}
