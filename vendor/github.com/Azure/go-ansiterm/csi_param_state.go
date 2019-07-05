package ansiterm

type csiParamState struct {
	baseState
}

func (csiState csiParamState) Handle(b byte) (s state, e error) {
	csiState.parser.logf("CsiParam::Handle %#x", b)

	nextState, err := csiState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(alphabetics, b):
		return csiState.parser.ground, nil
	case sliceContains(csiCollectables, b):
		csiState.parser.collectParam()
		return csiState, nil
	case sliceContains(executors, b):
		return csiState, csiState.parser.execute()
	}

	return csiState, nil
}

func (csiState csiParamState) Transition(s state) error {
	csiState.parser.logf("CsiParam::Transition %s --> %s", csiState.Name(), s.Name())
	csiState.baseState.Transition(s)

	switch s {
	case csiState.parser.ground:
		return csiState.parser.csiDispatch()
	}

	return nil
}
