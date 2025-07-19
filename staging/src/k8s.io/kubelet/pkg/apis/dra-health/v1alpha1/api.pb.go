/*
Copyright 2025 The Kubernetes Authors.

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

package v1alpha1

import (
	reflect "reflect"
	sync "sync"
	unsafe "unsafe"

	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type WatchResourcesRequest struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WatchResourcesRequest) Reset() {
	*x = WatchResourcesRequest{}
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WatchResourcesRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WatchResourcesRequest) ProtoMessage() {}

func (x *WatchResourcesRequest) ProtoReflect() protoreflect.Message {
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WatchResourcesRequest.ProtoReflect.Descriptor instead.
func (*WatchResourcesRequest) Descriptor() ([]byte, []int) {
	return file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescGZIP(), []int{0}
}

type DeviceHealth struct {
	state protoimpl.MessageState `protogen:"open.v1"`
	// resource_name is the full system-wide identifier: "<driver name>/<pool name>/<device name>".
	ResourceName string `protobuf:"bytes,1,opt,name=resource_name,json=resourceName,proto3" json:"resource_name,omitempty"`
	// pool_name identifies the pool within the driver.
	PoolName string `protobuf:"bytes,2,opt,name=pool_name,json=poolName,proto3" json:"pool_name,omitempty"`
	// device_name identifies the device within the pool.
	DeviceName string `protobuf:"bytes,3,opt,name=device_name,json=deviceName,proto3" json:"device_name,omitempty"`
	// health is the device's status: "Healthy", "Unhealthy", "Unknown".
	Health string `protobuf:"bytes,4,opt,name=health,proto3" json:"health,omitempty"`
	// last_updated is the Unix timestamp (seconds) of the last health update.
	LastUpdated   int64 `protobuf:"varint,5,opt,name=last_updated,json=lastUpdated,proto3" json:"last_updated,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *DeviceHealth) Reset() {
	*x = DeviceHealth{}
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *DeviceHealth) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeviceHealth) ProtoMessage() {}

func (x *DeviceHealth) ProtoReflect() protoreflect.Message {
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[1]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeviceHealth.ProtoReflect.Descriptor instead.
func (*DeviceHealth) Descriptor() ([]byte, []int) {
	return file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescGZIP(), []int{1}
}

func (x *DeviceHealth) GetResourceName() string {
	if x != nil {
		return x.ResourceName
	}
	return ""
}

func (x *DeviceHealth) GetPoolName() string {
	if x != nil {
		return x.PoolName
	}
	return ""
}

func (x *DeviceHealth) GetDeviceName() string {
	if x != nil {
		return x.DeviceName
	}
	return ""
}

func (x *DeviceHealth) GetHealth() string {
	if x != nil {
		return x.Health
	}
	return ""
}

func (x *DeviceHealth) GetLastUpdated() int64 {
	if x != nil {
		return x.LastUpdated
	}
	return 0
}

type WatchResourcesResponse struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Devices       []*DeviceHealth        `protobuf:"bytes,1,rep,name=devices,proto3" json:"devices,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *WatchResourcesResponse) Reset() {
	*x = WatchResourcesResponse{}
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *WatchResourcesResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WatchResourcesResponse) ProtoMessage() {}

func (x *WatchResourcesResponse) ProtoReflect() protoreflect.Message {
	mi := &file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes[2]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WatchResourcesResponse.ProtoReflect.Descriptor instead.
func (*WatchResourcesResponse) Descriptor() ([]byte, []int) {
	return file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescGZIP(), []int{2}
}

func (x *WatchResourcesResponse) GetDevices() []*DeviceHealth {
	if x != nil {
		return x.Devices
	}
	return nil
}

var File_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto protoreflect.FileDescriptor

var file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDesc = string([]byte{
	0x0a, 0x41, 0x73, 0x74, 0x61, 0x67, 0x69, 0x6e, 0x67, 0x2f, 0x73, 0x72, 0x63, 0x2f, 0x6b, 0x38,
	0x73, 0x2e, 0x69, 0x6f, 0x2f, 0x6b, 0x75, 0x62, 0x65, 0x6c, 0x65, 0x74, 0x2f, 0x70, 0x6b, 0x67,
	0x2f, 0x61, 0x70, 0x69, 0x73, 0x2f, 0x64, 0x72, 0x61, 0x2d, 0x68, 0x65, 0x61, 0x6c, 0x74, 0x68,
	0x2f, 0x76, 0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x2f, 0x61, 0x70, 0x69, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x12, 0x08, 0x76, 0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x22, 0x17, 0x0a,
	0x15, 0x57, 0x61, 0x74, 0x63, 0x68, 0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x73, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x22, 0xac, 0x01, 0x0a, 0x0c, 0x44, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x48, 0x65, 0x61, 0x6c, 0x74, 0x68, 0x12, 0x23, 0x0a, 0x0d, 0x72, 0x65, 0x73, 0x6f, 0x75,
	0x72, 0x63, 0x65, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c,
	0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x1b, 0x0a, 0x09,
	0x70, 0x6f, 0x6f, 0x6c, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x08, 0x70, 0x6f, 0x6f, 0x6c, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x1f, 0x0a, 0x0b, 0x64, 0x65, 0x76,
	0x69, 0x63, 0x65, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a,
	0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x16, 0x0a, 0x06, 0x68, 0x65,
	0x61, 0x6c, 0x74, 0x68, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x68, 0x65, 0x61, 0x6c,
	0x74, 0x68, 0x12, 0x21, 0x0a, 0x0c, 0x6c, 0x61, 0x73, 0x74, 0x5f, 0x75, 0x70, 0x64, 0x61, 0x74,
	0x65, 0x64, 0x18, 0x05, 0x20, 0x01, 0x28, 0x03, 0x52, 0x0b, 0x6c, 0x61, 0x73, 0x74, 0x55, 0x70,
	0x64, 0x61, 0x74, 0x65, 0x64, 0x22, 0x4a, 0x0a, 0x16, 0x57, 0x61, 0x74, 0x63, 0x68, 0x52, 0x65,
	0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12,
	0x30, 0x0a, 0x07, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b,
	0x32, 0x16, 0x2e, 0x76, 0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x2e, 0x44, 0x65, 0x76, 0x69,
	0x63, 0x65, 0x48, 0x65, 0x61, 0x6c, 0x74, 0x68, 0x52, 0x07, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65,
	0x73, 0x32, 0x65, 0x0a, 0x0a, 0x4e, 0x6f, 0x64, 0x65, 0x48, 0x65, 0x61, 0x6c, 0x74, 0x68, 0x12,
	0x57, 0x0a, 0x0e, 0x57, 0x61, 0x74, 0x63, 0x68, 0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
	0x73, 0x12, 0x1f, 0x2e, 0x76, 0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x2e, 0x57, 0x61, 0x74,
	0x63, 0x68, 0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x1a, 0x20, 0x2e, 0x76, 0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x2e, 0x57, 0x61,
	0x74, 0x63, 0x68, 0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x73, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x30, 0x01, 0x42, 0x2d, 0x5a, 0x2b, 0x6b, 0x38, 0x73, 0x2e,
	0x69, 0x6f, 0x2f, 0x6b, 0x75, 0x62, 0x65, 0x6c, 0x65, 0x74, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x61,
	0x70, 0x69, 0x73, 0x2f, 0x64, 0x72, 0x61, 0x2d, 0x68, 0x65, 0x61, 0x6c, 0x74, 0x68, 0x2f, 0x76,
	0x31, 0x61, 0x6c, 0x70, 0x68, 0x61, 0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
})

var (
	file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescOnce sync.Once
	file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescData []byte
)

func file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescGZIP() []byte {
	file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescOnce.Do(func() {
		file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescData = protoimpl.X.CompressGZIP(unsafe.Slice(unsafe.StringData(file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDesc), len(file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDesc)))
	})
	return file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDescData
}

var file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_goTypes = []any{
	(*WatchResourcesRequest)(nil),  // 0: v1alpha1.WatchResourcesRequest
	(*DeviceHealth)(nil),           // 1: v1alpha1.DeviceHealth
	(*WatchResourcesResponse)(nil), // 2: v1alpha1.WatchResourcesResponse
}
var file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_depIdxs = []int32{
	1, // 0: v1alpha1.WatchResourcesResponse.devices:type_name -> v1alpha1.DeviceHealth
	0, // 1: v1alpha1.NodeHealth.WatchResources:input_type -> v1alpha1.WatchResourcesRequest
	2, // 2: v1alpha1.NodeHealth.WatchResources:output_type -> v1alpha1.WatchResourcesResponse
	2, // [2:3] is the sub-list for method output_type
	1, // [1:2] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_init() }
func file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_init() {
	if File_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDesc), len(file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_rawDesc)),
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_goTypes,
		DependencyIndexes: file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_depIdxs,
		MessageInfos:      file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_msgTypes,
	}.Build()
	File_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto = out.File
	file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_goTypes = nil
	file_staging_src_k8s_io_kubelet_pkg_apis_dra_health_v1alpha1_api_proto_depIdxs = nil
}
