package debug

import "log"

type Logger bool

func (d Logger) Log(v ...interface{}) {
	if d {
		log.Print(v...)
	}
}

func (d Logger) Logf(s string, v ...interface{}) {
	if d {
		log.Printf(s, v...)
	}
}
