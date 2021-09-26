package test

import (
	"bytes"
	"encoding/json"
	"time"
)

func init() {
	var pString = func(val string) *string {
		return &val
	}
	epoch := time.Unix(0, 0)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr: (*struct {
			Field interface{}
		})(nil),
		input: `{"Field": "hello"}`,
	}, unmarshalCase{
		ptr: (*struct {
			Field interface{}
		})(nil),
		input: `{"Field": "hello"}       `,
	}, unmarshalCase{
		ptr: (*struct {
			Field int `json:"field"`
		})(nil),
		input: `{"field": null}`,
	}, unmarshalCase{
		ptr: (*struct {
			ID      int                    `json:"id"`
			Payload map[string]interface{} `json:"payload"`
			buf     *bytes.Buffer
		})(nil),
		input: ` {"id":1, "payload":{"account":"123","password":"456"}}`,
	}, unmarshalCase{
		ptr: (*struct {
			Field1 string
		})(nil),
		input: `{"Field\"1":"hello"}`,
	}, unmarshalCase{
		ptr: (*struct {
			Field1 string
		})(nil),
		input: `{"\u0046ield1":"hello"}`,
	}, unmarshalCase{
		ptr: (*struct {
			Field1 *string
			Field2 *string
		})(nil),
		input: `{"field1": null, "field2": "world"}`,
	}, unmarshalCase{
		ptr: (*struct {
			Field1 string
			Field2 json.RawMessage
		})(nil),
		input: `{"field1": "hello", "field2":[1,2,3]}`,
	}, unmarshalCase{
		ptr: (*struct {
			a int
			b <-chan int
			C int
			d *time.Timer
		})(nil),
		input: `{"a": 444, "b":"bad", "C":256, "d":{"not":"a timer"}}`,
	}, unmarshalCase{
		ptr: (*struct {
			A string
			B string
			C string
			D string
			E string
			F string
			G string
			H string
			I string
			J string
			K string
		})(nil),
		input: `{"a":"1","b":"2","c":"3","d":"4","e":"5","f":"6","g":"7","h":"8","i":"9","j":"10","k":"11"}`,
	}, unmarshalCase{
		ptr: (*struct {
			T float64 `json:"T"`
		})(nil),
		input: `{"t":10.0}`,
	}, unmarshalCase{
		ptr: (*struct {
			T float64 `json:"T"`
		})(nil),
		input: `{"T":10.0}`,
	}, unmarshalCase{
		ptr: (*struct {
			T float64 `json:"t"`
		})(nil),
		input: `{"T":10.0}`,
	}, unmarshalCase{
		ptr: (*struct {
			KeyString string       `json:"key_string"`
			Type      string       `json:"type"`
			Asks      [][2]float64 `json:"asks"`
		})(nil),
		input: `{"key_string": "KEYSTRING","type": "TYPE","asks": [[1e+66,1]]}`,
	})
	marshalCases = append(marshalCases,
		struct {
			Field map[string]interface{}
		}{
			map[string]interface{}{"hello": "world"},
		},
		struct {
			Field  map[string]interface{}
			Field2 string
		}{
			map[string]interface{}{"hello": "world"}, "",
		},
		struct {
			Field interface{}
		}{
			1024,
		},
		struct {
			Field MyInterface
		}{
			MyString("hello"),
		},
		struct {
			F *float64
		}{},
		struct {
			*time.Time
		}{&epoch},
		struct {
			*StructVarious
		}{&StructVarious{}},
		struct {
			*StructVarious
			Field int
		}{nil, 10},
		struct {
			Field1 int
			Field2 [1]*float64
		}{},
		struct {
			Field interface{} `json:"field,omitempty"`
		}{},
		struct {
			Field MyInterface `json:"field,omitempty"`
		}{},
		struct {
			Field MyInterface `json:"field,omitempty"`
		}{MyString("hello")},
		struct {
			Field json.Marshaler `json:"field"`
		}{},
		struct {
			Field MyInterface `json:"field"`
		}{},
		struct {
			Field MyInterface `json:"field"`
		}{MyString("hello")},
		struct {
			Field1 string `json:"field-1,omitempty"`
			Field2 func() `json:"-"`
		}{},
		structRecursive{},
		struct {
			*CacheItem

			// Omit bad keys
			OmitMaxAge omit `json:"cacheAge,omitempty"`

			// Add nice keys
			MaxAge int `json:"max_age"`
		}{
			CacheItem: &CacheItem{
				Key:    "value",
				MaxAge: 100,
			},
			MaxAge: 20,
		},
		structOrder{},
		struct {
			Field1 *string
			Field2 *string
		}{Field2: pString("world")},
		struct {
			a int
			b <-chan int
			C int
			d *time.Timer
		}{
			a: 42,
			b: make(<-chan int, 10),
			C: 21,
			d: time.NewTimer(10 * time.Second),
		},
		struct {
			_UnderscoreField string
		}{
			"should not marshal",
		},
	)
}

type StructVarious struct {
	Field0 string
	Field1 []string
	Field2 map[string]interface{}
}

type structRecursive struct {
	Field1 string
	Me     *structRecursive
}

type omit *struct{}
type CacheItem struct {
	Key    string `json:"key"`
	MaxAge int    `json:"cacheAge"`
}

type orderA struct {
	Field2 string
}

type orderC struct {
	Field5 string
}

type orderB struct {
	Field4 string
	orderC
	Field6 string
}

type structOrder struct {
	Field1 string
	orderA
	Field3 string
	orderB
	Field7 string
}
