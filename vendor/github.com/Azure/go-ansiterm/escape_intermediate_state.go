package ansiterm

type EscapeIntermediateState struct {
	BaseState
}

func (escState EscapeIntermediateState) Handle(b byte) (s State, e error) {
	logger.Infof("EscapeIntermediateState::Handle %#x", b)
	nextState, err := escState.BaseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(Intermeds, b):
		return escState, escState.parser.collectInter()
	case sliceContains(Executors, b):
		return escState, escState.parser.execute()
	case sliceContains(EscapeIntermediateToGroundBytes, b):
		return escState.parser.Ground, nil
	}

	return escState, nil
}

func (escState EscapeIntermediateState) Transition(s State) error {
	logger.Infof("EscapeIntermediateState::Transition %s --> %s", escState.Name(), s.Name())
	escState.BaseState.Transition(s)

	switch s {
	case escState.parser.Ground:
		return escState.parser.escDispatch()
	}

	return nil
}
