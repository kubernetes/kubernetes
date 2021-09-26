package skip_tests

func init() {
	testCases = append(testCases, testCase{
		ptr: (*string)(nil),
		inputs: []string{
			`""`,       // valid
			`"hello"`,  // valid
			`"`,        // invalid
			`"\"`,      // invalid
			`"\x00"`,   // invalid
			"\"\x00\"", // invalid
			"\"\t\"",   // invalid
			`"\t"`,     // valid
		},
	})
}
