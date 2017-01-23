package ansiterm

type EscapeState struct {
	BaseState
}

func (escState EscapeState) Handle(b byte) (s State, e error) {
	logger.Infof("EscapeState::Handle %#x", b)
	nextState, err := escState.BaseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case b == ANSI_ESCAPE_SECONDARY:
		return escState.parser.CsiEntry, nil
	case b == ANSI_OSC_STRING_ENTRY:
		return escState.parser.OscString, nil
	case sliceContains(Executors, b):
		return escState, escState.parser.execute()
	case sliceContains(EscapeToGroundBytes, b):
		return escState.parser.Ground, nil
	case sliceContains(Intermeds, b):
		return escState.parser.EscapeIntermediate, nil
	}

	return escState, nil
}

func (escState EscapeState) Transition(s State) error {
	logger.Infof("Escape::Transition %s --> %s", escState.Name(), s.Name())
	escState.BaseState.Transition(s)

	switch s {
	case escState.parser.Ground:
		return escState.parser.escDispatch()
	case escState.parser.EscapeIntermediate:
		return escState.parser.collectInter()
	}

	return nil
}

func (escState EscapeState) Enter() error {
	escState.parser.clear()
	return nil
}
