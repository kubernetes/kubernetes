package skip_tests

func init() {
	testCases = append(testCases, testCase{
		ptr: (*struct{})(nil),
		inputs: []string{
			`{}`,                         // valid
			`{"hello":"world"}`,          // valid
			`{hello:"world"}`,            // invalid
			`{"hello:"world"}`,           // invalid
			`{"hello","world"}`,          // invalid
			`{"hello":{}`,                // invalid
			`{"hello":{}}`,               // valid
			`{"hello":{}}}`,              // invalid
			`{"hello":  {  "hello": 1}}`, // valid
			`{abc}`,                      // invalid
		},
	})
}
