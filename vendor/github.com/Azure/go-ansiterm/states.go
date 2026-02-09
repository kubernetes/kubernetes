package ansiterm

type stateID int

type state interface {
	Enter() error
	Exit() error
	Handle(byte) (state, error)
	Name() string
	Transition(state) error
}

type baseState struct {
	name   string
	parser *AnsiParser
}

func (base baseState) Enter() error {
	return nil
}

func (base baseState) Exit() error {
	return nil
}

func (base baseState) Handle(b byte) (s state, e error) {

	switch {
	case b == CSI_ENTRY:
		return base.parser.csiEntry, nil
	case b == DCS_ENTRY:
		return base.parser.dcsEntry, nil
	case b == ANSI_ESCAPE_PRIMARY:
		return base.parser.escape, nil
	case b == OSC_STRING:
		return base.parser.oscString, nil
	case sliceContains(toGroundBytes, b):
		return base.parser.ground, nil
	}

	return nil, nil
}

func (base baseState) Name() string {
	return base.name
}

func (base baseState) Transition(s state) error {
	if s == base.parser.ground {
		execBytes := []byte{0x18}
		execBytes = append(execBytes, 0x1A)
		execBytes = append(execBytes, getByteRange(0x80, 0x8F)...)
		execBytes = append(execBytes, getByteRange(0x91, 0x97)...)
		execBytes = append(execBytes, 0x99)
		execBytes = append(execBytes, 0x9A)

		if sliceContains(execBytes, base.parser.context.currentChar) {
			return base.parser.execute()
		}
	}

	return nil
}

type dcsEntryState struct {
	baseState
}

type errorState struct {
	baseState
}
