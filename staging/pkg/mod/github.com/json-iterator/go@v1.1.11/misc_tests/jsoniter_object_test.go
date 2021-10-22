package misc_tests

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"strings"
	"time"
)

func Test_empty_object(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `{}`)
	field := iter.ReadObject()
	should.Equal("", field)
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `{}`)
	iter.ReadObjectCB(func(iter *jsoniter.Iterator, field string) bool {
		should.FailNow("should not call")
		return true
	})
}

func Test_one_field(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `{"a": "stream"}`)
	field := iter.ReadObject()
	should.Equal("a", field)
	value := iter.ReadString()
	should.Equal("stream", value)
	field = iter.ReadObject()
	should.Equal("", field)
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `{"a": "stream"}`)
	should.True(iter.ReadObjectCB(func(iter *jsoniter.Iterator, field string) bool {
		should.Equal("a", field)
		iter.Skip()
		return true
	}))

}

func Test_two_field(t *testing.T) {
	should := require.New(t)
	iter := jsoniter.ParseString(jsoniter.ConfigDefault, `{ "a": "stream" , "c": "d" }`)
	field := iter.ReadObject()
	should.Equal("a", field)
	value := iter.ReadString()
	should.Equal("stream", value)
	field = iter.ReadObject()
	should.Equal("c", field)
	value = iter.ReadString()
	should.Equal("d", value)
	field = iter.ReadObject()
	should.Equal("", field)
	iter = jsoniter.ParseString(jsoniter.ConfigDefault, `{"field1": "1", "field2": 2}`)
	for field := iter.ReadObject(); field != ""; field = iter.ReadObject() {
		switch field {
		case "field1":
			iter.ReadString()
		case "field2":
			iter.ReadInt64()
		default:
			iter.ReportError("bind object", "unexpected field")
		}
	}
}

func Test_write_object(t *testing.T) {
	should := require.New(t)
	buf := &bytes.Buffer{}
	stream := jsoniter.NewStream(jsoniter.Config{IndentionStep: 2}.Froze(), buf, 4096)
	stream.WriteObjectStart()
	stream.WriteObjectField("hello")
	stream.WriteInt(1)
	stream.WriteMore()
	stream.WriteObjectField("world")
	stream.WriteInt(2)
	stream.WriteObjectEnd()
	stream.Flush()
	should.Nil(stream.Error)
	should.Equal("{\n  \"hello\": 1,\n  \"world\": 2\n}", buf.String())
}

func Test_reader_and_load_more(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		CreatedAt time.Time
	}
	reader := strings.NewReader(`
{
	"agency": null,
	"candidateId": 0,
	"candidate": "Blah Blah",
	"bookingId": 0,
	"shiftId": 1,
	"shiftTypeId": 0,
	"shift": "Standard",
	"bonus": 0,
	"bonusNI": 0,
	"days": [],
	"totalHours": 27,
	"expenses": [],
	"weekEndingDateSystem": "2016-10-09",
	"weekEndingDateClient": "2016-10-09",
	"submittedAt": null,
	"submittedById": null,
	"approvedAt": "2016-10-10T18:38:04Z",
	"approvedById": 0,
	"authorisedAt": "2016-10-10T18:38:04Z",
	"authorisedById": 0,
	"invoicedAt": "2016-10-10T20:00:00Z",
	"revokedAt": null,
	"revokedById": null,
	"revokeReason": null,
	"rejectedAt": null,
	"rejectedById": null,
	"rejectReasonCode": null,
	"rejectReason": null,
	"createdAt": "2016-10-03T00:00:00Z",
	"updatedAt": "2016-11-09T10:26:13Z",
	"updatedById": null,
	"overrides": [],
	"bookingApproverId": null,
	"bookingApprover": null,
	"status": "approved"
}
	`)
	decoder := jsoniter.ConfigCompatibleWithStandardLibrary.NewDecoder(reader)
	obj := TestObject{}
	should.Nil(decoder.Decode(&obj))
}

func Test_unmarshal_into_existing_value(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		Field1 int
		Field2 interface{}
	}
	var obj TestObject
	m := map[string]interface{}{}
	obj.Field2 = &m
	cfg := jsoniter.Config{UseNumber: true}.Froze()
	err := cfg.Unmarshal([]byte(`{"Field1":1,"Field2":{"k":"v"}}`), &obj)
	should.NoError(err)
	should.Equal(map[string]interface{}{
		"k": "v",
	}, m)
}

// for issue421
func Test_unmarshal_anonymous_struct_invalid(t *testing.T) {
	should := require.New(t)
	t0 := struct {
		Field1 string
	}{}

	cfg := jsoniter.ConfigCompatibleWithStandardLibrary
	err := cfg.UnmarshalFromString(`{"Field1":`, &t0)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t0).String())

	cfgCaseSensitive := jsoniter.Config{
		CaseSensitive: true,
	}.Froze()

	type TestObject1 struct {
		Field1 struct {
			InnerField1 string
		}
	}
	t1 := TestObject1{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field1":{"InnerField1"`, &t1)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t1.Field1).String())
	should.Contains(err.Error(), reflect.TypeOf(t1).String())

	type TestObject2 struct {
		Field1 int
		Field2 struct {
			InnerField1 string
			InnerField2 string
		}
	}
	t2 := TestObject2{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field2":{"InnerField2"`, &t2)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t2.Field2).String())
	should.Contains(err.Error(), reflect.TypeOf(t2).String())

	type TestObject3 struct {
		Field1 int
		Field2 int
		Field3 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
		}
	}
	t3 := TestObject3{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field3":{"InnerField3"`, &t3)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t3.Field3).String())
	should.Contains(err.Error(), reflect.TypeOf(t3).String())

	type TestObject4 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
		}
	}
	t4 := TestObject4{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field4":{"InnerField4"`, &t4)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t4.Field4).String())
	should.Contains(err.Error(), reflect.TypeOf(t4).String())

	type TestObject5 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 int
		Field5 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
			InnerField5 string
		}
	}
	t5 := TestObject5{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field5":{"InnerField5"`, &t5)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t5.Field5).String())
	should.Contains(err.Error(), reflect.TypeOf(t5).String())

	type TestObject6 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 int
		Field5 int
		Field6 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
			InnerField5 string
			InnerField6 string
		}
	}
	t6 := TestObject6{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field6":{"InnerField6"`, &t6)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t6.Field6).String())
	should.Contains(err.Error(), reflect.TypeOf(t6).String())

	type TestObject7 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 int
		Field5 int
		Field6 int
		Field7 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
			InnerField5 string
			InnerField6 string
			InnerField7 string
		}
	}
	t7 := TestObject7{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field7":{"InnerField7"`, &t7)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t7.Field7).String())
	should.Contains(err.Error(), reflect.TypeOf(t7).String())

	type TestObject8 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 int
		Field5 int
		Field6 int
		Field7 int
		Field8 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
			InnerField5 string
			InnerField6 string
			InnerField7 string
			InnerField8 string
		}
	}
	t8 := TestObject8{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field8":{"InnerField8"`, &t8)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t8.Field8).String())
	should.Contains(err.Error(), reflect.TypeOf(t8).String())

	type TestObject9 struct {
		Field1 int
		Field2 int
		Field3 int
		Field4 int
		Field5 int
		Field6 int
		Field7 int
		Field8 int
		Field9 struct {
			InnerField1 string
			InnerField2 string
			InnerField3 string
			InnerField4 string
			InnerField5 string
			InnerField6 string
			InnerField7 string
			InnerField8 string
			InnerField9 string
		}
	}
	t9 := TestObject9{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field9":{"InnerField9"`, &t9)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t9.Field9).String())
	should.Contains(err.Error(), reflect.TypeOf(t9).String())

	type TestObject10 struct {
		Field1  int
		Field2  int
		Field3  int
		Field4  int
		Field5  int
		Field6  int
		Field7  int
		Field8  int
		Field9  int
		Field10 struct {
			InnerField1  string
			InnerField2  string
			InnerField3  string
			InnerField4  string
			InnerField5  string
			InnerField6  string
			InnerField7  string
			InnerField8  string
			InnerField9  string
			InnerField10 string
		}
	}
	t10 := TestObject10{}
	err = cfgCaseSensitive.UnmarshalFromString(`{"Field10":{"InnerField10"`, &t10)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t10.Field10).String())
	should.Contains(err.Error(), reflect.TypeOf(t10).String())

	err = cfg.UnmarshalFromString(`{"Field10":{"InnerField10"`, &t10)
	should.NotNil(err)
	should.NotContains(err.Error(), reflect.TypeOf(t10.Field10).String())
	should.Contains(err.Error(), reflect.TypeOf(t10).String())
}
