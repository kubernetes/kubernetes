package test

func init() {
	var pEFace = func(val interface{}) *interface{} {
		return &val
	}
	var pInt = func(val int) *int {
		return &val
	}
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (**interface{})(nil),
		input: `"hello"`,
	}, unmarshalCase{
		ptr:   (**interface{})(nil),
		input: `1e1`,
	}, unmarshalCase{
		ptr:   (**interface{})(nil),
		input: `1.0e1`,
	}, unmarshalCase{
		ptr:   (*[]interface{})(nil),
		input: `[1.0e1]`,
	}, unmarshalCase{
		ptr: (*struct {
			Field interface{}
		})(nil),
		input: `{"field":"hello"}`,
	}, unmarshalCase{
		obj: func() interface{} {
			type TestData struct {
				Name string `json:"name"`
			}
			o := &TestData{}
			return &o
		},
		input: `{"name":"value"}`,
	}, unmarshalCase{
		obj: func() interface{} {
			b := true
			return &struct {
				Field interface{} `json:"field"`
			}{&b}
		},
		input: `{"field": null}`,
	}, unmarshalCase{
		obj: func() interface{} {
			var pb *bool
			return &struct {
				Field interface{} `json:"field"`
			}{&pb}
		},
		input: `{"field": null}`,
	}, unmarshalCase{
		obj: func() interface{} {
			b := true
			pb := &b
			return &struct {
				Field interface{} `json:"field"`
			}{&pb}
		},
		input: `{"field": null}`,
	})
	marshalCases = append(marshalCases,
		pEFace("hello"),
		struct {
			Field interface{}
		}{"hello"},
		struct {
			Field interface{}
		}{struct {
			field chan int
		}{}},
		struct {
			Field interface{}
		}{struct {
			Field *int
		}{pInt(100)}},
	)
}
