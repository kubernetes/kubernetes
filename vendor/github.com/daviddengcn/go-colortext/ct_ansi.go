// +build !windows

package ct

import (
	"fmt"
)

func resetColor() {
	fmt.Print("\x1b[0m")
}

func changeColor(fg Color, fgBright bool, bg Color, bgBright bool) {
	if fg == None && bg == None {
		return
	} // if

	s := ""
	if fg != None {
		s = fmt.Sprintf("%s%d", s, 30+(int)(fg-Black))
		if fgBright {
			s += ";1"
		} // if
	} // if

	if bg != None {
		if s != "" {
			s += ";"
		} // if
		s = fmt.Sprintf("%s%d", s, 40+(int)(bg-Black))
	} // if

	s = "\x1b[0;" + s + "m"
	fmt.Print(s)
}
