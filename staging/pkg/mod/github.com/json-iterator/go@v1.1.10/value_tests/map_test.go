package test

import (
	"encoding/json"
	"fmt"
	"math/big"
	"time"
)

func init() {
	var pRawMessage = func(val json.RawMessage) *json.RawMessage {
		return &val
	}
	nilMap := map[string]string(nil)
	marshalCases = append(marshalCases,
		map[string]interface{}{"abc": 1},
		map[string]MyInterface{"hello": MyString("world")},
		map[*big.Float]string{big.NewFloat(1.2): "2"},
		map[string]interface{}{
			"3": 3,
			"1": 1,
			"2": 2,
		},
		map[uint64]interface{}{
			uint64(1): "a",
			uint64(2): "a",
			uint64(4): "a",
		},
		nilMap,
		&nilMap,
		map[string]*json.RawMessage{"hello": pRawMessage(json.RawMessage("[]"))},
		map[Date]bool{{}: true},
		map[Date2]bool{{}: true},
		map[customKey]string{customKey(1): "bar"},
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*map[string]string)(nil),
		input: `{"k\"ey": "val"}`,
	}, unmarshalCase{
		ptr:   (*map[string]string)(nil),
		input: `null`,
	}, unmarshalCase{
		ptr:   (*map[string]*json.RawMessage)(nil),
		input: "{\"test\":[{\"key\":\"value\"}]}",
	}, unmarshalCase{
		ptr: (*map[Date]bool)(nil),
		input: `{
        "2018-12-12": true,
        "2018-12-13": true,
        "2018-12-14": true
    	}`,
	}, unmarshalCase{
		ptr: (*map[Date2]bool)(nil),
		input: `{
        "2018-12-12": true,
        "2018-12-13": true,
        "2018-12-14": true
    	}`,
	}, unmarshalCase{
		ptr: (*map[customKey]string)(nil),
		input: `{"foo": "bar"}`,
	})
}

type MyInterface interface {
	Hello() string
}

type MyString string

func (ms MyString) Hello() string {
	return string(ms)
}

type Date struct {
	time.Time
}

func (d *Date) UnmarshalJSON(b []byte) error {
	dateStr := string(b) // something like `"2017-08-20"`

	if dateStr == "null" {
		return nil
	}

	t, err := time.Parse(`"2006-01-02"`, dateStr)
	if err != nil {
		return fmt.Errorf("cant parse date: %#v", err)
	}

	d.Time = t
	return nil
}

func (d *Date) MarshalJSON() ([]byte, error) {
	return []byte(d.Time.Format("2006-01-02")), nil
}

type Date2 struct {
	time.Time
}

func (d Date2) UnmarshalJSON(b []byte) error {
	dateStr := string(b) // something like `"2017-08-20"`

	if dateStr == "null" {
		return nil
	}

	t, err := time.Parse(`"2006-01-02"`, dateStr)
	if err != nil {
		return fmt.Errorf("cant parse date: %#v", err)
	}

	d.Time = t
	return nil
}

func (d Date2) MarshalJSON() ([]byte, error) {
	return []byte(d.Time.Format("2006-01-02")), nil
}

type customKey int32

func (c customKey) MarshalText() ([]byte, error) {
	return []byte("foo"), nil
}

func (c *customKey) UnmarshalText(value []byte) error {
	*c = 1
	return nil
}
