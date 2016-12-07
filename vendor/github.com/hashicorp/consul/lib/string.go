package lib

// StrContains checks if a list contains a string
func StrContains(l []string, s string) bool {
	for _, v := range l {
		if v == s {
			return true
		}
	}
	return false
}
