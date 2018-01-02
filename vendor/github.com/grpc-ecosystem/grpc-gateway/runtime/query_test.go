package runtime_test

import (
	"net/url"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"github.com/grpc-ecosystem/grpc-gateway/utilities"
)

func TestPopulateParameters(t *testing.T) {
	for _, spec := range []struct {
		values url.Values
		filter *utilities.DoubleArray
		want   proto.Message
	}{
		{
			values: url.Values{
				"float_value":    {"1.5"},
				"double_value":   {"2.5"},
				"int64_value":    {"-1"},
				"int32_value":    {"-2"},
				"uint64_value":   {"3"},
				"uint32_value":   {"4"},
				"bool_value":     {"true"},
				"string_value":   {"str"},
				"repeated_value": {"a", "b", "c"},
			},
			filter: utilities.NewDoubleArray(nil),
			want: &proto3Message{
				FloatValue:    1.5,
				DoubleValue:   2.5,
				Int64Value:    -1,
				Int32Value:    -2,
				Uint64Value:   3,
				Uint32Value:   4,
				BoolValue:     true,
				StringValue:   "str",
				RepeatedValue: []string{"a", "b", "c"},
			},
		},
		{
			values: url.Values{
				"float_value":    {"1.5"},
				"double_value":   {"2.5"},
				"int64_value":    {"-1"},
				"int32_value":    {"-2"},
				"uint64_value":   {"3"},
				"uint32_value":   {"4"},
				"bool_value":     {"true"},
				"string_value":   {"str"},
				"repeated_value": {"a", "b", "c"},
			},
			filter: utilities.NewDoubleArray(nil),
			want: &proto2Message{
				FloatValue:    proto.Float32(1.5),
				DoubleValue:   proto.Float64(2.5),
				Int64Value:    proto.Int64(-1),
				Int32Value:    proto.Int32(-2),
				Uint64Value:   proto.Uint64(3),
				Uint32Value:   proto.Uint32(4),
				BoolValue:     proto.Bool(true),
				StringValue:   proto.String("str"),
				RepeatedValue: []string{"a", "b", "c"},
			},
		},
		{
			values: url.Values{
				"nested.nested.nested.repeated_value": {"a", "b", "c"},
				"nested.nested.nested.string_value":   {"s"},
				"nested.nested.string_value":          {"t"},
				"nested.string_value":                 {"u"},
				"nested_non_null.string_value":        {"v"},
			},
			filter: utilities.NewDoubleArray(nil),
			want: &proto3Message{
				Nested: &proto2Message{
					Nested: &proto3Message{
						Nested: &proto2Message{
							RepeatedValue: []string{"a", "b", "c"},
							StringValue:   proto.String("s"),
						},
						StringValue: "t",
					},
					StringValue: proto.String("u"),
				},
				NestedNonNull: proto2Message{
					StringValue: proto.String("v"),
				},
			},
		},
		{
			values: url.Values{
				"uint64_value": {"1", "2", "3", "4", "5"},
			},
			filter: utilities.NewDoubleArray(nil),
			want: &proto3Message{
				Uint64Value: 1,
			},
		},
	} {
		msg := proto.Clone(spec.want)
		msg.Reset()
		err := runtime.PopulateQueryParameters(msg, spec.values, spec.filter)
		if err != nil {
			t.Errorf("runtime.PoplateQueryParameters(msg, %v, %v) failed with %v; want success", spec.values, spec.filter, err)
			continue
		}
		if got, want := msg, spec.want; !proto.Equal(got, want) {
			t.Errorf("runtime.PopulateQueryParameters(msg, %v, %v = %v; want %v", spec.values, spec.filter, got, want)
		}
	}
}

func TestPopulateParametersWithFilters(t *testing.T) {
	for _, spec := range []struct {
		values url.Values
		filter *utilities.DoubleArray
		want   proto.Message
	}{
		{
			values: url.Values{
				"bool_value":     {"true"},
				"string_value":   {"str"},
				"repeated_value": {"a", "b", "c"},
			},
			filter: utilities.NewDoubleArray([][]string{
				{"bool_value"}, {"repeated_value"},
			}),
			want: &proto3Message{
				StringValue: "str",
			},
		},
		{
			values: url.Values{
				"nested.nested.bool_value":   {"true"},
				"nested.nested.string_value": {"str"},
				"nested.string_value":        {"str"},
				"string_value":               {"str"},
			},
			filter: utilities.NewDoubleArray([][]string{
				{"nested"},
			}),
			want: &proto3Message{
				StringValue: "str",
			},
		},
		{
			values: url.Values{
				"nested.nested.bool_value":   {"true"},
				"nested.nested.string_value": {"str"},
				"nested.string_value":        {"str"},
				"string_value":               {"str"},
			},
			filter: utilities.NewDoubleArray([][]string{
				{"nested", "nested"},
			}),
			want: &proto3Message{
				Nested: &proto2Message{
					StringValue: proto.String("str"),
				},
				StringValue: "str",
			},
		},
		{
			values: url.Values{
				"nested.nested.bool_value":   {"true"},
				"nested.nested.string_value": {"str"},
				"nested.string_value":        {"str"},
				"string_value":               {"str"},
			},
			filter: utilities.NewDoubleArray([][]string{
				{"nested", "nested", "string_value"},
			}),
			want: &proto3Message{
				Nested: &proto2Message{
					StringValue: proto.String("str"),
					Nested: &proto3Message{
						BoolValue: true,
					},
				},
				StringValue: "str",
			},
		},
	} {
		msg := proto.Clone(spec.want)
		msg.Reset()
		err := runtime.PopulateQueryParameters(msg, spec.values, spec.filter)
		if err != nil {
			t.Errorf("runtime.PoplateQueryParameters(msg, %v, %v) failed with %v; want success", spec.values, spec.filter, err)
			continue
		}
		if got, want := msg, spec.want; !proto.Equal(got, want) {
			t.Errorf("runtime.PopulateQueryParameters(msg, %v, %v = %v; want %v", spec.values, spec.filter, got, want)
		}
	}
}

type proto3Message struct {
	Nested        *proto2Message `protobuf:"bytes,1,opt,name=nested" json:"nested,omitempty"`
	NestedNonNull proto2Message  `protobuf:"bytes,11,opt,name=nested_non_null" json:"nested_non_null,omitempty"`
	FloatValue    float32        `protobuf:"fixed32,2,opt,name=float_value" json:"float_value,omitempty"`
	DoubleValue   float64        `protobuf:"fixed64,3,opt,name=double_value" json:"double_value,omitempty"`
	Int64Value    int64          `protobuf:"varint,4,opt,name=int64_value" json:"int64_value,omitempty"`
	Int32Value    int32          `protobuf:"varint,5,opt,name=int32_value" json:"int32_value,omitempty"`
	Uint64Value   uint64         `protobuf:"varint,6,opt,name=uint64_value" json:"uint64_value,omitempty"`
	Uint32Value   uint32         `protobuf:"varint,7,opt,name=uint32_value" json:"uint32_value,omitempty"`
	BoolValue     bool           `protobuf:"varint,8,opt,name=bool_value" json:"bool_value,omitempty"`
	StringValue   string         `protobuf:"bytes,9,opt,name=string_value" json:"string_value,omitempty"`
	RepeatedValue []string       `protobuf:"bytes,10,rep,name=repeated_value" json:"repeated_value,omitempty"`
}

func (m *proto3Message) Reset()         { *m = proto3Message{} }
func (m *proto3Message) String() string { return proto.CompactTextString(m) }
func (*proto3Message) ProtoMessage()    {}

func (m *proto3Message) GetNested() *proto2Message {
	if m != nil {
		return m.Nested
	}
	return nil
}

type proto2Message struct {
	Nested           *proto3Message `protobuf:"bytes,1,opt,name=nested" json:"nested,omitempty"`
	FloatValue       *float32       `protobuf:"fixed32,2,opt,name=float_value" json:"float_value,omitempty"`
	DoubleValue      *float64       `protobuf:"fixed64,3,opt,name=double_value" json:"double_value,omitempty"`
	Int64Value       *int64         `protobuf:"varint,4,opt,name=int64_value" json:"int64_value,omitempty"`
	Int32Value       *int32         `protobuf:"varint,5,opt,name=int32_value" json:"int32_value,omitempty"`
	Uint64Value      *uint64        `protobuf:"varint,6,opt,name=uint64_value" json:"uint64_value,omitempty"`
	Uint32Value      *uint32        `protobuf:"varint,7,opt,name=uint32_value" json:"uint32_value,omitempty"`
	BoolValue        *bool          `protobuf:"varint,8,opt,name=bool_value" json:"bool_value,omitempty"`
	StringValue      *string        `protobuf:"bytes,9,opt,name=string_value" json:"string_value,omitempty"`
	RepeatedValue    []string       `protobuf:"bytes,10,rep,name=repeated_value" json:"repeated_value,omitempty"`
	XXX_unrecognized []byte         `json:"-"`
}

func (m *proto2Message) Reset()         { *m = proto2Message{} }
func (m *proto2Message) String() string { return proto.CompactTextString(m) }
func (*proto2Message) ProtoMessage()    {}

func (m *proto2Message) GetNested() *proto3Message {
	if m != nil {
		return m.Nested
	}
	return nil
}

func (m *proto2Message) GetFloatValue() float32 {
	if m != nil && m.FloatValue != nil {
		return *m.FloatValue
	}
	return 0
}

func (m *proto2Message) GetDoubleValue() float64 {
	if m != nil && m.DoubleValue != nil {
		return *m.DoubleValue
	}
	return 0
}

func (m *proto2Message) GetInt64Value() int64 {
	if m != nil && m.Int64Value != nil {
		return *m.Int64Value
	}
	return 0
}

func (m *proto2Message) GetInt32Value() int32 {
	if m != nil && m.Int32Value != nil {
		return *m.Int32Value
	}
	return 0
}

func (m *proto2Message) GetUint64Value() uint64 {
	if m != nil && m.Uint64Value != nil {
		return *m.Uint64Value
	}
	return 0
}

func (m *proto2Message) GetUint32Value() uint32 {
	if m != nil && m.Uint32Value != nil {
		return *m.Uint32Value
	}
	return 0
}

func (m *proto2Message) GetBoolValue() bool {
	if m != nil && m.BoolValue != nil {
		return *m.BoolValue
	}
	return false
}

func (m *proto2Message) GetStringValue() string {
	if m != nil && m.StringValue != nil {
		return *m.StringValue
	}
	return ""
}

func (m *proto2Message) GetRepeatedValue() []string {
	if m != nil {
		return m.RepeatedValue
	}
	return nil
}
