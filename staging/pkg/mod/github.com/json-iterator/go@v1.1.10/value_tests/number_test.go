package test

import "encoding/json"

func init() {
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*json.Number)(nil),
		input: `"500"`,
	}, unmarshalCase{
		ptr:   (*json.Number)(nil),
		input: `1`,
	}, unmarshalCase{
		ptr:   (*json.Number)(nil),
		input: `null`,
	})
	marshalCases = append(marshalCases, json.Number(""))
}
