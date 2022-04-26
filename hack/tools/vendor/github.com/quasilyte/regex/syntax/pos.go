package syntax

type Position struct {
	Begin uint16
	End   uint16
}

func combinePos(begin, end Position) Position {
	return Position{Begin: begin.Begin, End: end.End}
}
