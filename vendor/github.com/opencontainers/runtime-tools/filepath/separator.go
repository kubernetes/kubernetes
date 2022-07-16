package filepath

// Separator is an explicit-OS version of path/filepath's Separator.
func Separator(os string) rune {
	if os == "windows" {
		return '\\'
	}
	return '/'
}
