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

package ttrpc

import (
	"fmt"

	spb "google.golang.org/genproto/googleapis/rpc/status"
)

type Request struct {
	Service     string      `protobuf:"bytes,1,opt,name=service,proto3"`
	Method      string      `protobuf:"bytes,2,opt,name=method,proto3"`
	Payload     []byte      `protobuf:"bytes,3,opt,name=payload,proto3"`
	TimeoutNano int64       `protobuf:"varint,4,opt,name=timeout_nano,proto3"`
	Metadata    []*KeyValue `protobuf:"bytes,5,rep,name=metadata,proto3"`
}

func (r *Request) Reset()         { *r = Request{} }
func (r *Request) String() string { return fmt.Sprintf("%+#v", r) }
func (r *Request) ProtoMessage()  {}

type Response struct {
	Status  *spb.Status `protobuf:"bytes,1,opt,name=status,proto3"`
	Payload []byte      `protobuf:"bytes,2,opt,name=payload,proto3"`
}

func (r *Response) Reset()         { *r = Response{} }
func (r *Response) String() string { return fmt.Sprintf("%+#v", r) }
func (r *Response) ProtoMessage()  {}

type StringList struct {
	List []string `protobuf:"bytes,1,rep,name=list,proto3"`
}

func (r *StringList) Reset()         { *r = StringList{} }
func (r *StringList) String() string { return fmt.Sprintf("%+#v", r) }
func (r *StringList) ProtoMessage()  {}

func makeStringList(item ...string) StringList { return StringList{List: item} }

type KeyValue struct {
	Key   string `protobuf:"bytes,1,opt,name=key,proto3"`
	Value string `protobuf:"bytes,2,opt,name=value,proto3"`
}

func (m *KeyValue) Reset()         { *m = KeyValue{} }
func (*KeyValue) ProtoMessage()    {}
func (m *KeyValue) String() string { return fmt.Sprintf("%+#v", m) }
