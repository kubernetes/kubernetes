/*
Copyright 2015 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupResource specifies a Group and a Resource, but does not force a version.  This is useful for identifying
// concepts during lookup stages without having partially valid types
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type GroupResource struct {
	Group    string `protobuf:"bytes,1,opt,name=group"`
	Resource string `protobuf:"bytes,2,opt,name=resource"`
}

func (gr *GroupResource) String() string {
	if len(gr.Group) == 0 {
		return gr.Resource
	}
	return gr.Resource + "." + gr.Group
}

// GroupVersionResource unambiguously identifies a resource.  It doesn't anonymously include GroupVersion
// to avoid automatic coersion.  It doesn't use a GroupVersion to avoid custom marshalling
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type GroupVersionResource struct {
	Group    string `protobuf:"bytes,1,opt,name=group"`
	Version  string `protobuf:"bytes,2,opt,name=version"`
	Resource string `protobuf:"bytes,3,opt,name=resource"`
}

func (gvr *GroupVersionResource) String() string {
	return strings.Join([]string{gvr.Group, "/", gvr.Version, ", Resource=", gvr.Resource}, "")
}

// GroupKind specifies a Group and a Kind, but does not force a version.  This is useful for identifying
// concepts during lookup stages without having partially valid types
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type GroupKind struct {
	Group string `protobuf:"bytes,1,opt,name=group"`
	Kind  string `protobuf:"bytes,2,opt,name=kind"`
}

func (gk *GroupKind) String() string {
	if len(gk.Group) == 0 {
		return gk.Kind
	}
	return gk.Kind + "." + gk.Group
}

// GroupVersionKind unambiguously identifies a kind.  It doesn't anonymously include GroupVersion
// to avoid automatic coersion.  It doesn't use a GroupVersion to avoid custom marshalling
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type GroupVersionKind struct {
	Group   string `protobuf:"bytes,1,opt,name=group"`
	Version string `protobuf:"bytes,2,opt,name=version"`
	Kind    string `protobuf:"bytes,3,opt,name=kind"`
}

func (gvk GroupVersionKind) String() string {
	return gvk.Group + "/" + gvk.Version + ", Kind=" + gvk.Kind
}

// GroupVersion contains the "group" and the "version", which uniquely identifies the API.
//
// +protobuf.options.(gogoproto.goproto_stringer)=false
type GroupVersion struct {
	Group   string `protobuf:"bytes,1,opt,name=group"`
	Version string `protobuf:"bytes,2,opt,name=version"`
}

// Empty returns true if group and version are empty
func (gv GroupVersion) Empty() bool {
	return len(gv.Group) == 0 && len(gv.Version) == 0
}

// String puts "group" and "version" into a single "group/version" string. For the legacy v1
// it returns "v1".
func (gv GroupVersion) String() string {
	// special case the internal apiVersion for the legacy kube types
	if gv.Empty() {
		return ""
	}

	// special case of "v1" for backward compatibility
	if len(gv.Group) == 0 && gv.Version == "v1" {
		return gv.Version
	}
	if len(gv.Group) > 0 {
		return gv.Group + "/" + gv.Version
	}
	return gv.Version
}

// MarshalJSON implements the json.Marshaller interface.
func (gv GroupVersion) MarshalJSON() ([]byte, error) {
	s := gv.String()
	if strings.Count(s, "/") > 1 {
		return []byte{}, fmt.Errorf("illegal GroupVersion %v: contains more than one /", s)
	}
	return json.Marshal(s)
}

func (gv *GroupVersion) unmarshal(value []byte) error {
	var s string
	if err := json.Unmarshal(value, &s); err != nil {
		return err
	}
	parsed, err := schema.ParseGroupVersion(s)
	if err != nil {
		return err
	}
	gv.Group, gv.Version = parsed.Group, parsed.Version
	return nil
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (gv *GroupVersion) UnmarshalJSON(value []byte) error {
	return gv.unmarshal(value)
}

// UnmarshalTEXT implements the Ugorji's encoding.TextUnmarshaler interface.
func (gv *GroupVersion) UnmarshalText(value []byte) error {
	return gv.unmarshal(value)
}
