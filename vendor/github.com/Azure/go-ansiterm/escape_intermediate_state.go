package ansiterm

type escapeIntermediateState struct {
	baseState
}

func (escState escapeIntermediateState) Handle(b byte) (s state, e error) {
	escState.parser.logf("escapeIntermediateState::Handle %#x", b)
	nextState, err := escState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(intermeds, b):
		return escState, escState.parser.collectInter()
	case sliceContains(executors, b):
		return escState, escState.parser.execute()
	case sliceContains(escapeIntermediateToGroundBytes, b):
		return escState.parser.ground, nil
	}

	return escState, nil
}

func (escState escapeIntermediateState) Transition(s state) error {
	escState.parser.logf("escapeIntermediateState::Transition %s --> %s", escState.Name(), s.Name())
	escState.baseState.Transition(s)

	switch s {
	case escState.parser.ground:
		return escState.parser.escDispatch()
	}

	return nil
}
