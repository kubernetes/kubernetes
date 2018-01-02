package test

type typeForTest []interface{}

var inputs = []string{
	`[]`,             // valid
	`[1]`,            // valid
	`[  1, "hello"]`, // valid
	`[abc]`,          // invalid
	`[`,              // invalid
	`[[]`,            // invalid
}
