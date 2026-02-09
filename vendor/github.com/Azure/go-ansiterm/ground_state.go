package ansiterm

type groundState struct {
	baseState
}

func (gs groundState) Handle(b byte) (s state, e error) {
	gs.parser.context.currentChar = b

	nextState, err := gs.baseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(printables, b):
		return gs, gs.parser.print()

	case sliceContains(executors, b):
		return gs, gs.parser.execute()
	}

	return gs, nil
}
