package test

type typeForTest float64

var inputs = []string{
	"+1",    // invalid
	"-a",    // invalid
	"-\x00", // invalid, zero byte
	"0.1",   // valid
	"0..1",  // invalid, more dot
	"1e+1",  // valid
	"1+1",   // invalid
	"1E1",   // valid, e or E
	"1ee1",  // invalid
	"100a",  // invalid
	"10.",   // invalid
}
