//go:build gofuzz
// +build gofuzz

package configs

func FuzzUnmarshalJSON(data []byte) int {
	hooks := Hooks{}
	_ = hooks.UnmarshalJSON(data)
	return 1
}
