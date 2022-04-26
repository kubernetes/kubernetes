package quasigo

import (
	"fmt"
	"strings"
)

// TODO(quasilyte): generate extra opcode info so we can simplify disasm function?

func disasm(env *Env, fn *Func) string {
	var out strings.Builder

	dbg, ok := env.debug.funcs[fn]
	if !ok {
		return "<unknown>\n"
	}

	code := fn.code
	labels := map[int]string{}
	walkBytecode(code, func(pc int, op opcode) {
		switch op {
		case opJumpTrue, opJumpFalse, opJump:
			offset := decode16(code, pc+1)
			targetPC := pc + offset
			if _, ok := labels[targetPC]; !ok {
				labels[targetPC] = fmt.Sprintf("L%d", len(labels))
			}
		}
	})

	walkBytecode(code, func(pc int, op opcode) {
		if l := labels[pc]; l != "" {
			fmt.Fprintf(&out, "%s:\n", l)
		}
		var arg interface{}
		var comment string
		switch op {
		case opCallNative:
			id := decode16(code, pc+1)
			arg = id
			comment = env.nativeFuncs[id].name
		case opPushParam, opPushIntParam:
			index := int(code[pc+1])
			arg = index
			comment = dbg.paramNames[index]
		case opSetLocal, opSetIntLocal, opPushLocal, opPushIntLocal, opIncLocal, opDecLocal:
			index := int(code[pc+1])
			arg = index
			comment = dbg.localNames[index]
		case opSetVariadicLen:
			arg = int(code[pc+1])
		case opPushConst:
			arg = int(code[pc+1])
			comment = fmt.Sprintf("value=%#v", fn.constants[code[pc+1]])
		case opPushIntConst:
			arg = int(code[pc+1])
			comment = fmt.Sprintf("value=%#v", fn.intConstants[code[pc+1]])
		case opJumpTrue, opJumpFalse, opJump:
			offset := decode16(code, pc+1)
			targetPC := pc + offset
			arg = offset
			comment = labels[targetPC]
		}

		if comment != "" {
			comment = " # " + comment
		}
		if arg == nil {
			fmt.Fprintf(&out, "  %s%s\n", op, comment)
		} else {
			fmt.Fprintf(&out, "  %s %#v%s\n", op, arg, comment)
		}
	})

	return out.String()
}
