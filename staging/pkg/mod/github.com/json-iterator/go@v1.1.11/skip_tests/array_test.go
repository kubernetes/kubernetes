package skip_tests

func init() {
	testCases = append(testCases, testCase{
		ptr: (*[]interface{})(nil),
		inputs: []string{
			`[]`,             // valid
			`[1]`,            // valid
			`[  1, "hello"]`, // valid
			`[abc]`,          // invalid
			`[`,              // invalid
			`[[]`,            // invalid
		},
	})
}
