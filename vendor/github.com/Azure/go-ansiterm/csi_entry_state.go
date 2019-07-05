package ansiterm

type csiEntryState struct {
	baseState
}

func (csiState csiEntryState) Handle(b byte) (s state, e error) {
	csiState.parser.logf("CsiEntry::Handle %#x", b)

	nextState, err := csiState.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(alphabetics, b):
		return csiState.parser.ground, nil
	case sliceContains(csiCollectables, b):
		return csiState.parser.csiParam, nil
	case sliceContains(executors, b):
		return csiState, csiState.parser.execute()
	}

	return csiState, nil
}

func (csiState csiEntryState) Transition(s state) error {
	csiState.parser.logf("CsiEntry::Transition %s --> %s", csiState.Name(), s.Name())
	csiState.baseState.Transition(s)

	switch s {
	case csiState.parser.ground:
		return csiState.parser.csiDispatch()
	case csiState.parser.csiParam:
		switch {
		case sliceContains(csiParams, csiState.parser.context.currentChar):
			csiState.parser.collectParam()
		case sliceContains(intermeds, csiState.parser.context.currentChar):
			csiState.parser.collectInter()
		}
	}

	return nil
}

func (csiState csiEntryState) Enter() error {
	csiState.parser.clear()
	return nil
}
