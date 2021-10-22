package test

import (
	"encoding/json"
)

func init() {
	marshalCases = append(marshalCases,
		json.RawMessage("{}"),
		json.RawMessage("12345"),
		json.RawMessage("3.14"),
		json.RawMessage("-0.5e10"),
		struct {
			Env   string          `json:"env"`
			Extra json.RawMessage `json:"extra,omitempty"`
		}{
			Env: "jfdk",
		},
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*json.RawMessage)(nil),
		input: `[1,2,3]`,
	}, unmarshalCase{
		ptr:   (*json.RawMessage)(nil),
		input: `1.122e+250`,
	})
}
