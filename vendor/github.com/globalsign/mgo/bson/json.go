package bson

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/globalsign/mgo/internal/json"
)

// UnmarshalJSON unmarshals a JSON value that may hold non-standard
// syntax as defined in BSON's extended JSON specification.
func UnmarshalJSON(data []byte, value interface{}) error {
	d := json.NewDecoder(bytes.NewBuffer(data))
	d.Extend(&jsonExt)
	return d.Decode(value)
}

// MarshalJSON marshals a JSON value that may hold non-standard
// syntax as defined in BSON's extended JSON specification.
func MarshalJSON(value interface{}) ([]byte, error) {
	var buf bytes.Buffer
	e := json.NewEncoder(&buf)
	e.Extend(&jsonExt)
	err := e.Encode(value)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// jdec is used internally by the JSON decoding functions
// so they may unmarshal functions without getting into endless
// recursion due to keyed objects.
func jdec(data []byte, value interface{}) error {
	d := json.NewDecoder(bytes.NewBuffer(data))
	d.Extend(&funcExt)
	return d.Decode(value)
}

var jsonExt json.Extension
var funcExt json.Extension

// TODO
// - Shell regular expressions ("/regexp/opts")

func init() {
	jsonExt.DecodeUnquotedKeys(true)
	jsonExt.DecodeTrailingCommas(true)

	funcExt.DecodeFunc("BinData", "$binaryFunc", "$type", "$binary")
	jsonExt.DecodeKeyed("$binary", jdecBinary)
	jsonExt.DecodeKeyed("$binaryFunc", jdecBinary)
	jsonExt.EncodeType([]byte(nil), jencBinarySlice)
	jsonExt.EncodeType(Binary{}, jencBinaryType)

	funcExt.DecodeFunc("ISODate", "$dateFunc", "S")
	funcExt.DecodeFunc("new Date", "$dateFunc", "S")
	jsonExt.DecodeKeyed("$date", jdecDate)
	jsonExt.DecodeKeyed("$dateFunc", jdecDate)
	jsonExt.EncodeType(time.Time{}, jencDate)

	funcExt.DecodeFunc("Timestamp", "$timestamp", "t", "i")
	jsonExt.DecodeKeyed("$timestamp", jdecTimestamp)
	jsonExt.EncodeType(MongoTimestamp(0), jencTimestamp)

	funcExt.DecodeConst("undefined", Undefined)

	jsonExt.DecodeKeyed("$regex", jdecRegEx)
	jsonExt.EncodeType(RegEx{}, jencRegEx)

	funcExt.DecodeFunc("ObjectId", "$oidFunc", "Id")
	jsonExt.DecodeKeyed("$oid", jdecObjectId)
	jsonExt.DecodeKeyed("$oidFunc", jdecObjectId)
	jsonExt.EncodeType(ObjectId(""), jencObjectId)

	funcExt.DecodeFunc("DBRef", "$dbrefFunc", "$ref", "$id")
	jsonExt.DecodeKeyed("$dbrefFunc", jdecDBRef)

	funcExt.DecodeFunc("NumberLong", "$numberLongFunc", "N")
	jsonExt.DecodeKeyed("$numberLong", jdecNumberLong)
	jsonExt.DecodeKeyed("$numberLongFunc", jdecNumberLong)
	jsonExt.EncodeType(int64(0), jencNumberLong)
	jsonExt.EncodeType(int(0), jencInt)

	funcExt.DecodeConst("MinKey", MinKey)
	funcExt.DecodeConst("MaxKey", MaxKey)
	jsonExt.DecodeKeyed("$minKey", jdecMinKey)
	jsonExt.DecodeKeyed("$maxKey", jdecMaxKey)
	jsonExt.EncodeType(orderKey(0), jencMinMaxKey)

	jsonExt.DecodeKeyed("$undefined", jdecUndefined)
	jsonExt.EncodeType(Undefined, jencUndefined)

	jsonExt.Extend(&funcExt)
}

func fbytes(format string, args ...interface{}) []byte {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, format, args...)
	return buf.Bytes()
}

func jdecBinary(data []byte) (interface{}, error) {
	var v struct {
		Binary []byte `json:"$binary"`
		Type   string `json:"$type"`
		Func   struct {
			Binary []byte `json:"$binary"`
			Type   int64  `json:"$type"`
		} `json:"$binaryFunc"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}

	var binData []byte
	var binKind int64
	if v.Type == "" && v.Binary == nil {
		binData = v.Func.Binary
		binKind = v.Func.Type
	} else if v.Type == "" {
		return v.Binary, nil
	} else {
		binData = v.Binary
		binKind, err = strconv.ParseInt(v.Type, 0, 64)
		if err != nil {
			binKind = -1
		}
	}

	if binKind == 0 {
		return binData, nil
	}
	if binKind < 0 || binKind > 255 {
		return nil, fmt.Errorf("invalid type in binary object: %s", data)
	}

	return Binary{Kind: byte(binKind), Data: binData}, nil
}

func jencBinarySlice(v interface{}) ([]byte, error) {
	in := v.([]byte)
	out := make([]byte, base64.StdEncoding.EncodedLen(len(in)))
	base64.StdEncoding.Encode(out, in)
	return fbytes(`{"$binary":"%s","$type":"0x0"}`, out), nil
}

func jencBinaryType(v interface{}) ([]byte, error) {
	in := v.(Binary)
	out := make([]byte, base64.StdEncoding.EncodedLen(len(in.Data)))
	base64.StdEncoding.Encode(out, in.Data)
	return fbytes(`{"$binary":"%s","$type":"0x%x"}`, out, in.Kind), nil
}

const jdateFormat = "2006-01-02T15:04:05.999Z07:00"

func jdecDate(data []byte) (interface{}, error) {
	var v struct {
		S    string `json:"$date"`
		Func struct {
			S string
		} `json:"$dateFunc"`
	}
	_ = jdec(data, &v)
	if v.S == "" {
		v.S = v.Func.S
	}
	if v.S != "" {
		var errs []string
		for _, format := range []string{jdateFormat, "2006-01-02"} {
			t, err := time.Parse(format, v.S)
			if err == nil {
				return t, nil
			}
			errs = append(errs, err.Error())
		}
		return nil, fmt.Errorf("cannot parse date: %q [%s]", v.S, strings.Join(errs, ", "))
	}

	var vn struct {
		Date struct {
			N int64 `json:"$numberLong,string"`
		} `json:"$date"`
		Func struct {
			S int64
		} `json:"$dateFunc"`
	}
	err := jdec(data, &vn)
	if err != nil {
		return nil, fmt.Errorf("cannot parse date: %q", data)
	}
	n := vn.Date.N
	if n == 0 {
		n = vn.Func.S
	}
	return time.Unix(n/1000, n%1000*1e6).UTC(), nil
}

func jencDate(v interface{}) ([]byte, error) {
	t := v.(time.Time)
	return fbytes(`{"$date":%q}`, t.Format(jdateFormat)), nil
}

func jdecTimestamp(data []byte) (interface{}, error) {
	var v struct {
		Func struct {
			T int32 `json:"t"`
			I int32 `json:"i"`
		} `json:"$timestamp"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	return MongoTimestamp(uint64(v.Func.T)<<32 | uint64(uint32(v.Func.I))), nil
}

func jencTimestamp(v interface{}) ([]byte, error) {
	ts := uint64(v.(MongoTimestamp))
	return fbytes(`{"$timestamp":{"t":%d,"i":%d}}`, ts>>32, uint32(ts)), nil
}

func jdecRegEx(data []byte) (interface{}, error) {
	var v struct {
		Regex   string `json:"$regex"`
		Options string `json:"$options"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	return RegEx{v.Regex, v.Options}, nil
}

func jencRegEx(v interface{}) ([]byte, error) {
	re := v.(RegEx)
	type regex struct {
		Regex   string `json:"$regex"`
		Options string `json:"$options"`
	}
	return json.Marshal(regex{re.Pattern, re.Options})
}

func jdecObjectId(data []byte) (interface{}, error) {
	var v struct {
		Id   string `json:"$oid"`
		Func struct {
			Id string
		} `json:"$oidFunc"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	if v.Id == "" {
		v.Id = v.Func.Id
	}
	return ObjectIdHex(v.Id), nil
}

func jencObjectId(v interface{}) ([]byte, error) {
	return fbytes(`{"$oid":"%s"}`, v.(ObjectId).Hex()), nil
}

func jdecDBRef(data []byte) (interface{}, error) {
	// TODO Support unmarshaling $ref and $id into the input value.
	var v struct {
		Obj map[string]interface{} `json:"$dbrefFunc"`
	}
	// TODO Fix this. Must not be required.
	v.Obj = make(map[string]interface{})
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	return v.Obj, nil
}

func jdecNumberLong(data []byte) (interface{}, error) {
	var v struct {
		N    int64 `json:"$numberLong,string"`
		Func struct {
			N int64 `json:",string"`
		} `json:"$numberLongFunc"`
	}
	var vn struct {
		N    int64 `json:"$numberLong"`
		Func struct {
			N int64
		} `json:"$numberLongFunc"`
	}
	err := jdec(data, &v)
	if err != nil {
		err = jdec(data, &vn)
		v.N = vn.N
		v.Func.N = vn.Func.N
	}
	if err != nil {
		return nil, err
	}
	if v.N != 0 {
		return v.N, nil
	}
	return v.Func.N, nil
}

func jencNumberLong(v interface{}) ([]byte, error) {
	n := v.(int64)
	f := `{"$numberLong":"%d"}`
	if n <= 1<<53 {
		f = `{"$numberLong":%d}`
	}
	return fbytes(f, n), nil
}

func jencInt(v interface{}) ([]byte, error) {
	n := v.(int)
	f := `{"$numberLong":"%d"}`
	if int64(n) <= 1<<53 {
		f = `%d`
	}
	return fbytes(f, n), nil
}

func jdecMinKey(data []byte) (interface{}, error) {
	var v struct {
		N int64 `json:"$minKey"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	if v.N != 1 {
		return nil, fmt.Errorf("invalid $minKey object: %s", data)
	}
	return MinKey, nil
}

func jdecMaxKey(data []byte) (interface{}, error) {
	var v struct {
		N int64 `json:"$maxKey"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	if v.N != 1 {
		return nil, fmt.Errorf("invalid $maxKey object: %s", data)
	}
	return MaxKey, nil
}

func jencMinMaxKey(v interface{}) ([]byte, error) {
	switch v.(orderKey) {
	case MinKey:
		return []byte(`{"$minKey":1}`), nil
	case MaxKey:
		return []byte(`{"$maxKey":1}`), nil
	}
	panic(fmt.Sprintf("invalid $minKey/$maxKey value: %d", v))
}

func jdecUndefined(data []byte) (interface{}, error) {
	var v struct {
		B bool `json:"$undefined"`
	}
	err := jdec(data, &v)
	if err != nil {
		return nil, err
	}
	if !v.B {
		return nil, fmt.Errorf("invalid $undefined object: %s", data)
	}
	return Undefined, nil
}

func jencUndefined(v interface{}) ([]byte, error) {
	return []byte(`{"$undefined":true}`), nil
}
