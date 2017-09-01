package ansiterm

type oscStringState struct {
	baseState
}

func (oscState oscStringState) Handle(b byte) (s state, e error) {
	logger.Infof("OscString::Handle %#x", b)
	nextState, err := oscState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case isOscStringTerminator(b):
		return oscState.parser.ground, nil
	}

	return oscState, nil
}

// See below for OSC string terminators for linux
// http://man7.org/linux/man-pages/man4/console_codes.4.html
func isOscStringTerminator(b byte) bool {

	if b == ANSI_BEL || b == 0x5C {
		return true
	}

	return false
}
