package sliceutil

// IndexOf get the index of the given value in the given string slice,
// or -1 if not found.
func IndexOf(slice []string, value string) int {
	for i, v := range slice {
		if v == value {
			return i
		}
	}
	return -1
}

// Contains check if a string slice contains a value.
func Contains(slice []string, value string) bool {
	return IndexOf(slice, value) != -1
}
