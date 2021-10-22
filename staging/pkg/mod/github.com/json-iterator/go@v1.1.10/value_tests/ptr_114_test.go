// +build go1.14

package test

func init() {
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		obj: func() interface{} {
			var i int
			pi := &i
			ppi := &pi
			return &ppi
		},
		input: "null",
	})
}
