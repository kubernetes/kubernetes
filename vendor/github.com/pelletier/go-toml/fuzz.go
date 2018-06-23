// +build gofuzz

package toml

func Fuzz(data []byte) int {
	tree, err := LoadBytes(data)
	if err != nil {
		if tree != nil {
			panic("tree must be nil if there is an error")
		}
		return 0
	}

	str, err := tree.ToTomlString()
	if err != nil {
		if str != "" {
			panic(`str must be "" if there is an error`)
		}
		panic(err)
	}

	tree, err = Load(str)
	if err != nil {
		if tree != nil {
			panic("tree must be nil if there is an error")
		}
		return 0
	}

	return 1
}
