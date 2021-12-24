// +build gofuzz

package reference

// fuzzParseNormalizedNamed implements a fuzzer
// that targets ParseNormalizedNamed
// Export before building the fuzzer.
// nolint:deadcode
func fuzzParseNormalizedNamed(data []byte) int {
	_, _ = ParseNormalizedNamed(string(data))
	return 1
}
