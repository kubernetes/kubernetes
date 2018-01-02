package denco

// NextSeparator returns an index of next separator in path.
func NextSeparator(path string, start int) int {
	for start < len(path) {
		if c := path[start]; c == '/' || c == TerminationCharacter {
			break
		}
		start++
	}
	return start
}
