package check

func PrintLine(filename string, line int) (string, error) {
	return printLine(filename, line)
}

func Indent(s, with string) string {
	return indent(s, with)
}
