package test

func init() {
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr: (*struct {
			Field bool `json:"field"`
		})(nil),
		input: `{"field": null}`,
	})
}
