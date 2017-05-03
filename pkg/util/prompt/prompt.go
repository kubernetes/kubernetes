package prompt

import (
	"fmt"
	"io"

	"github.com/howeyc/gopass"
)

const (
	ShowEcho     = true
	DontShowEcho = false
	Mask         = true
	DontMask     = false
)

type Prompter struct {
	reader io.Reader
}

func NewPrompter(reader io.Reader) *Prompter {
	return &Prompter{reader: reader}
}

func (p Prompter) Prompt(field string, showEcho bool, mask bool) (result string, err error) {
	fmt.Printf("Please enter %s: ", field)
	if showEcho {
		_, err = fmt.Fscan(p.reader, &result)
	} else {
		var data []byte
		if mask {
			data, err = gopass.GetPasswdMasked()
		} else {
			data, err = gopass.GetPasswd()
		}
		result = string(data)
	}
	return result, err
}
