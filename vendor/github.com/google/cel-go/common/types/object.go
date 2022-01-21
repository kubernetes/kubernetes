// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

type protoObj struct {
	ref.TypeAdapter
	value     proto.Message
	typeDesc  *pb.TypeDescription
	typeValue *TypeValue
}

// NewObject returns an object based on a proto.Message value which handles
// conversion between protobuf type values and expression type values.
// Objects support indexing and iteration.
//
// Note: the type value is pulled from the list of registered types within the
// type provider. If the proto type is not registered within the type provider,
// then this will result in an error within the type adapter / provider.
func NewObject(adapter ref.TypeAdapter,
	typeDesc *pb.TypeDescription,
	typeValue *TypeValue,
	value proto.Message) ref.Val {
	return &protoObj{
		TypeAdapter: adapter,
		value:       value,
		typeDesc:    typeDesc,
		typeValue:   typeValue}
}

func (o *protoObj) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	pb := o.value
	if reflect.TypeOf(pb).AssignableTo(typeDesc) {
		return pb, nil
	}
	if reflect.TypeOf(o).AssignableTo(typeDesc) {
		return o, nil
	}
	switch typeDesc {
	case anyValueType:
		_, isAny := pb.(*anypb.Any)
		if isAny {
			return pb, nil
		}
		return anypb.New(pb)
	case jsonValueType:
		// Marshal the proto to JSON first, and then rehydrate as protobuf.Value as there is no
		// support for direct conversion from proto.Message to protobuf.Value.
		bytes, err := protojson.Marshal(pb)
		if err != nil {
			return nil, err
		}
		json := &structpb.Value{}
		err = protojson.Unmarshal(bytes, json)
		if err != nil {
			return nil, err
		}
		return json, nil
	default:
		if typeDesc == o.typeDesc.ReflectType() {
			return o.value, nil
		}
		if typeDesc.Kind() == reflect.Ptr {
			val := reflect.New(typeDesc.Elem()).Interface()
			dstPB, ok := val.(proto.Message)
			if ok {
				proto.Merge(dstPB, pb)
				return dstPB, nil
			}
		}
	}
	return nil, fmt.Errorf("type conversion error from '%T' to '%v'", o.value, typeDesc)
}

func (o *protoObj) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	default:
		if o.Type().TypeName() == typeVal.TypeName() {
			return o
		}
	case TypeType:
		return o.typeValue
	}
	return NewErr("type conversion error from '%s' to '%s'", o.typeDesc.Name(), typeVal)
}

func (o *protoObj) Equal(other ref.Val) ref.Val {
	if o.typeDesc.Name() != other.Type().TypeName() {
		return MaybeNoSuchOverloadErr(other)
	}
	return Bool(proto.Equal(o.value, other.Value().(proto.Message)))
}

// IsSet tests whether a field which is defined is set to a non-default value.
func (o *protoObj) IsSet(field ref.Val) ref.Val {
	protoFieldName, ok := field.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(field)
	}
	protoFieldStr := string(protoFieldName)
	fd, found := o.typeDesc.FieldByName(protoFieldStr)
	if !found {
		return NewErr("no such field '%s'", field)
	}
	if fd.IsSet(o.value) {
		return True
	}
	return False
}

func (o *protoObj) Get(index ref.Val) ref.Val {
	protoFieldName, ok := index.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(index)
	}
	protoFieldStr := string(protoFieldName)
	fd, found := o.typeDesc.FieldByName(protoFieldStr)
	if !found {
		return NewErr("no such field '%s'", index)
	}
	fv, err := fd.GetFrom(o.value)
	if err != nil {
		return NewErr(err.Error())
	}
	return o.NativeToValue(fv)
}

func (o *protoObj) Type() ref.Type {
	return o.typeValue
}

func (o *protoObj) Value() interface{} {
	return o.value
}
