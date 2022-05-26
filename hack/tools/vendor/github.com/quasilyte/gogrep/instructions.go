package gogrep

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

type bitmap64 uint64

func (m bitmap64) IsSet(pos int) bool {
	return m&(1<<pos) != 0
}

type valueKind uint8

const (
	emptyValue   valueKind = iota
	stringValue            // Extra values only; value is stored in program.strings
	ifaceValue             // Extra values only; value is stored in program.ifaces
	tokenValue             // token.Token
	chandirValue           // ast.CharDir
	intValue               // int
)

type program struct {
	insts   []instruction
	strings []string
	ifaces  []interface{}
}

func formatProgram(p *program) []string {
	var parts []string
	insts := p.insts

	nextInst := func() instruction {
		inst := insts[0]
		insts = insts[1:]
		return inst
	}
	peekOp := func() operation {
		return insts[0].op
	}

	var walk func(int)
	walk = func(depth int) {
		if len(insts) == 0 {
			return
		}
		inst := nextInst()

		part := strings.Repeat(" • ", depth) + formatInstruction(p, inst)
		parts = append(parts, part)

		info := operationInfoTable[inst.op]
		for i := 0; i < info.NumArgs; i++ {
			if i == info.SliceIndex {
				for j := 0; j < int(inst.value); j++ {
					walk(depth + 1)
				}
				continue
			}
			if !info.VariadicMap.IsSet(i) {
				walk(depth + 1)
				continue
			}
			for {
				isEnd := peekOp() == opEnd
				walk(depth + 1)
				if isEnd {
					break
				}
			}
		}
	}

	walk(0)
	return parts
}

func formatInstruction(p *program, inst instruction) string {
	parts := []string{inst.op.String()}

	info := operationInfoTable[inst.op]

	switch info.ValueKind {
	case chandirValue:
		dir := ast.ChanDir(inst.value)
		if dir&ast.SEND != 0 {
			parts = append(parts, "send")
		}
		if dir&ast.RECV != 0 {
			parts = append(parts, "recv")
		}
	case tokenValue:
		parts = append(parts, token.Token(inst.value).String())
	case intValue:
		parts = append(parts, fmt.Sprint(inst.value))
	}

	switch info.ExtraValueKind {
	case ifaceValue:
		parts = append(parts, fmt.Sprintf("%#v", p.ifaces[inst.valueIndex]))
	case stringValue:
		parts = append(parts, p.strings[inst.valueIndex])
	}

	return strings.Join(parts, " ")
}

type instruction struct {
	op         operation
	value      uint8
	valueIndex uint8
}
