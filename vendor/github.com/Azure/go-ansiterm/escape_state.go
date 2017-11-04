package ansiterm

type escapeState struct {
	baseState
}

func (escState escapeState) Handle(b byte) (s state, e error) {
	logger.Infof("escapeState::Handle %#x", b)
	nextState, err := escState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case b == ANSI_ESCAPE_SECONDARY:
		return escState.parser.csiEntry, nil
	case b == ANSI_OSC_STRING_ENTRY:
		return escState.parser.oscString, nil
	case sliceContains(executors, b):
		return escState, escState.parser.execute()
	case sliceContains(escapeToGroundBytes, b):
		return escState.parser.ground, nil
	case sliceContains(intermeds, b):
		return escState.parser.escapeIntermediate, nil
	}

	return escState, nil
}

func (escState escapeState) Transition(s state) error {
	logger.Infof("Escape::Transition %s --> %s", escState.Name(), s.Name())
	escState.baseState.Transition(s)

	switch s {
	case escState.parser.ground:
		return escState.parser.escDispatch()
	case escState.parser.escapeIntermediate:
		return escState.parser.collectInter()
	}

	return nil
}

func (escState escapeState) Enter() error {
	escState.parser.clear()
	return nil
}
