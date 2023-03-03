package ansiterm

func (ap *AnsiParser) collectParam() error {
	currChar := ap.context.currentChar
	ap.logf("collectParam %#x", currChar)
	ap.context.paramBuffer = append(ap.context.paramBuffer, currChar)
	return nil
}

func (ap *AnsiParser) collectInter() error {
	currChar := ap.context.currentChar
	ap.logf("collectInter %#x", currChar)
	ap.context.paramBuffer = append(ap.context.interBuffer, currChar)
	return nil
}

func (ap *AnsiParser) escDispatch() error {
	cmd, _ := parseCmd(*ap.context)
	intermeds := ap.context.interBuffer
	ap.logf("escDispatch currentChar: %#x", ap.context.currentChar)
	ap.logf("escDispatch: %v(%v)", cmd, intermeds)

	switch cmd {
	case "D": // IND
		return ap.eventHandler.IND()
	case "E": // NEL, equivalent to CRLF
		err := ap.eventHandler.Execute(ANSI_CARRIAGE_RETURN)
		if err == nil {
			err = ap.eventHandler.Execute(ANSI_LINE_FEED)
		}
		return err
	case "M": // RI
		return ap.eventHandler.RI()
	}

	return nil
}

func (ap *AnsiParser) csiDispatch() error {
	cmd, _ := parseCmd(*ap.context)
	params, _ := parseParams(ap.context.paramBuffer)
	ap.logf("Parsed params: %v with length: %d", params, len(params))

	ap.logf("csiDispatch: %v(%v)", cmd, params)

	switch cmd {
	case "@":
		return ap.eventHandler.ICH(getInt(params, 1))
	case "A":
		return ap.eventHandler.CUU(getInt(params, 1))
	case "B":
		return ap.eventHandler.CUD(getInt(params, 1))
	case "C":
		return ap.eventHandler.CUF(getInt(params, 1))
	case "D":
		return ap.eventHandler.CUB(getInt(params, 1))
	case "E":
		return ap.eventHandler.CNL(getInt(params, 1))
	case "F":
		return ap.eventHandler.CPL(getInt(params, 1))
	case "G":
		return ap.eventHandler.CHA(getInt(params, 1))
	case "H":
		ints := getInts(params, 2, 1)
		x, y := ints[0], ints[1]
		return ap.eventHandler.CUP(x, y)
	case "J":
		param := getEraseParam(params)
		return ap.eventHandler.ED(param)
	case "K":
		param := getEraseParam(params)
		return ap.eventHandler.EL(param)
	case "L":
		return ap.eventHandler.IL(getInt(params, 1))
	case "M":
		return ap.eventHandler.DL(getInt(params, 1))
	case "P":
		return ap.eventHandler.DCH(getInt(params, 1))
	case "S":
		return ap.eventHandler.SU(getInt(params, 1))
	case "T":
		return ap.eventHandler.SD(getInt(params, 1))
	case "c":
		return ap.eventHandler.DA(params)
	case "d":
		return ap.eventHandler.VPA(getInt(params, 1))
	case "f":
		ints := getInts(params, 2, 1)
		x, y := ints[0], ints[1]
		return ap.eventHandler.HVP(x, y)
	case "h":
		return ap.hDispatch(params)
	case "l":
		return ap.lDispatch(params)
	case "m":
		return ap.eventHandler.SGR(getInts(params, 1, 0))
	case "r":
		ints := getInts(params, 2, 1)
		top, bottom := ints[0], ints[1]
		return ap.eventHandler.DECSTBM(top, bottom)
	default:
		ap.logf("ERROR: Unsupported CSI command: '%s', with full context:  %v", cmd, ap.context)
		return nil
	}

}

func (ap *AnsiParser) print() error {
	return ap.eventHandler.Print(ap.context.currentChar)
}

func (ap *AnsiParser) clear() error {
	ap.context = &ansiContext{}
	return nil
}

func (ap *AnsiParser) execute() error {
	return ap.eventHandler.Execute(ap.context.currentChar)
}
