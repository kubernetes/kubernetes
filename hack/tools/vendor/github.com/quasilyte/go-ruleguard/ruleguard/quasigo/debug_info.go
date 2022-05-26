package quasigo

type debugInfo struct {
	funcs map[*Func]funcDebugInfo
}

type funcDebugInfo struct {
	paramNames []string
	localNames []string
}

func newDebugInfo() *debugInfo {
	return &debugInfo{
		funcs: make(map[*Func]funcDebugInfo),
	}
}
