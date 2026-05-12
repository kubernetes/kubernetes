package ansiterm

type oscStringState struct {
	baseState
}

func (oscState oscStringState) Handle(b byte) (s state, e error) {
	oscState.parser.logf("OscString::Handle %#x", b)
	nextState, err := oscState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	// There are several control characters and sequences which can
	// terminate an OSC string. Most of them are handled by the baseState
	// handler. The ANSI_BEL character is a special case which behaves as a
	// terminator only for an OSC string.
	if b == ANSI_BEL {
		return oscState.parser.ground, nil
	}

	return oscState, nil
}
