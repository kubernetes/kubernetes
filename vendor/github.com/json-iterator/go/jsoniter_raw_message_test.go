package jsoniter

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func Test_json_RawMessage(t *testing.T) {
	should := require.New(t)
	var data json.RawMessage
	should.Nil(Unmarshal([]byte(`[1,2,3]`), &data))
	should.Equal(`[1,2,3]`, string(data))
	str, err := MarshalToString(data)
	should.Nil(err)
	should.Equal(`[1,2,3]`, str)
}

func Test_jsoniter_RawMessage(t *testing.T) {
	should := require.New(t)
	var data RawMessage
	should.Nil(Unmarshal([]byte(`[1,2,3]`), &data))
	should.Equal(`[1,2,3]`, string(data))
	str, err := MarshalToString(data)
	should.Nil(err)
	should.Equal(`[1,2,3]`, str)
}

func Test_json_RawMessage_in_struct(t *testing.T) {
	type TestObject struct {
		Field1 string
		Field2 json.RawMessage
	}
	should := require.New(t)
	var data TestObject
	should.Nil(Unmarshal([]byte(`{"field1": "hello", "field2": [1,2,3]}`), &data))
	should.Equal(` [1,2,3]`, string(data.Field2))
	should.Equal(`hello`, data.Field1)
}

func Test_decode_map_of_raw_message(t *testing.T) {
	should := require.New(t)
	type RawMap map[string]*json.RawMessage
	b := []byte("{\"test\":[{\"key\":\"value\"}]}")
	var rawMap RawMap
	should.Nil(Unmarshal(b, &rawMap))
	should.Equal(`[{"key":"value"}]`, string(*rawMap["test"]))
	type Inner struct {
		Key string `json:"key"`
	}
	var inner []Inner
	Unmarshal(*rawMap["test"], &inner)
	should.Equal("value", inner[0].Key)
}

func Test_encode_map_of_raw_message(t *testing.T) {
	should := require.New(t)
	type RawMap map[string]*json.RawMessage
	value := json.RawMessage("[]")
	rawMap := RawMap{"hello": &value}
	output, err := MarshalToString(rawMap)
	should.Nil(err)
	should.Equal(`{"hello":[]}`, output)
}

func Test_encode_map_of_jsoniter_raw_message(t *testing.T) {
	should := require.New(t)
	type RawMap map[string]*RawMessage
	value := RawMessage("[]")
	rawMap := RawMap{"hello": &value}
	output, err := MarshalToString(rawMap)
	should.Nil(err)
	should.Equal(`{"hello":[]}`, output)
}

func Test_marshal_invalid_json_raw_message(t *testing.T) {
	type A struct {
		Raw json.RawMessage `json:"raw"`
	}
	message := []byte(`{}`)

	a := A{}
	should := require.New(t)
	should.Nil(ConfigCompatibleWithStandardLibrary.Unmarshal(message, &a))
	aout, aouterr := ConfigCompatibleWithStandardLibrary.Marshal(&a)
	should.Equal(`{"raw":null}`, string(aout))
	should.Nil(aouterr)
}

func Test_raw_message_memory_not_copied_issue(t *testing.T) {
	jsonStream := `{"name":"xxxxx","bundle_id":"com.zonst.majiang","app_platform":"ios","app_category":"100103", "budget_day":1000,"bidding_min":1,"bidding_max":2,"bidding_type":"CPM", "freq":{"open":true,"type":"day","num":100},"speed":1, "targeting":{"vendor":{"open":true,"list":["zonst"]}, "geo_code":{"open":true,"list":["156110100"]},"app_category":{"open":true,"list":["100101"]}, "day_parting":{"open":true,"list":["100409","100410"]},"device_type":{"open":true,"list":["ipad"]}, "os_version":{"open":true,"list":[10]},"carrier":{"open":true,"list":["mobile"]}, "network":{"open":true,"list":["4G"]}},"url":{"tracking_imp_url":"http://www.baidu.com", "tracking_clk_url":"http://www.baidu.com","jump_url":"http://www.baidu.com","deep_link_url":"http://www.baidu.com"}}`
	type IteratorObject struct {
		Name        *string     `json:"name"`
		BundleId    *string     `json:"bundle_id"`
		AppCategory *string     `json:"app_category"`
		AppPlatform *string     `json:"app_platform"`
		BudgetDay   *float32    `json:"budget_day"`
		BiddingMax  *float32    `json:"bidding_max"`
		BiddingMin  *float32    `json:"bidding_min"`
		BiddingType *string     `json:"bidding_type"`
		Freq        *RawMessage `json:"freq"`
		Targeting   *RawMessage `json:"targeting"`
		Url         *RawMessage `json:"url"`
		Speed       *int        `json:"speed" db:"speed"`
	}

	obj := &IteratorObject{}
	decoder := NewDecoder(strings.NewReader(jsonStream))
	err := decoder.Decode(obj)
	should := require.New(t)
	should.Nil(err)
	should.Equal(`{"open":true,"type":"day","num":100}`, string(*obj.Freq))
}
