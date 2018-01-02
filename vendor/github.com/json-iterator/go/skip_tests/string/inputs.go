package test

type typeForTest string

var inputs = []string{
	`""`,       // valid
	`"hello"`,  // valid
	`"`,        // invalid
	`"\"`,      // invalid
	`"\x00"`,   // invalid
	"\"\x00\"", // invalid
	"\"\t\"",   // invalid
	`"\t"`,     // valid
}
