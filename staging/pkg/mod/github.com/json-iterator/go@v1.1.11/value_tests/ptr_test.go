package test

func init() {
	var pInt = func(val int) *int {
		return &val
	}
	marshalCases = append(marshalCases,
		(*int)(nil),
		pInt(100),
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		obj: func() interface{} {
			var i int
			return &i
		},
		input: "null",
	}, unmarshalCase{
		obj: func() interface{} {
			var i *int
			return &i
		},
		input: "10",
	}, unmarshalCase{
		obj: func() interface{} {
			var i int
			pi := &i
			return &pi
		},
		input: "null",
	})
}
