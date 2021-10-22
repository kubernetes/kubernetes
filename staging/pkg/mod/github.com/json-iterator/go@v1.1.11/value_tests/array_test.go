package test

func init() {
	two := float64(2)
	marshalCases = append(marshalCases,
		[1]*float64{nil},
		[1]*float64{&two},
		[2]*float64{},
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*[0]int)(nil),
		input: `[1]`,
	}, unmarshalCase{
		ptr:   (*[1]int)(nil),
		input: `[2]`,
	}, unmarshalCase{
		ptr:   (*[1]int)(nil),
		input: `[]`,
	})
}
