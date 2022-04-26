//go:build main
// +build main

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"strings"
	"text/template"
)

var opcodePrototypes = []opcodeProto{
	{"Pop", "op", "(value) -> ()"},
	{"Dup", "op", "(x) -> (x x)"},

	{"PushParam", "op index:u8", "() -> (value)"},
	{"PushIntParam", "op index:u8", "() -> (value:int)"},
	{"PushLocal", "op index:u8", "() -> (value)"},
	{"PushIntLocal", "op index:u8", "() -> (value:int)"},
	{"PushFalse", "op", "() -> (false)"},
	{"PushTrue", "op", "() -> (true)"},
	{"PushConst", "op constid:u8", "() -> (const)"},
	{"PushIntConst", "op constid:u8", "() -> (const:int)"},

	{"ConvIntToIface", "op", "(value:int) -> (value)"},

	{"SetLocal", "op index:u8", "(value) -> ()"},
	{"SetIntLocal", "op index:u8", "(value:int) -> ()"},
	{"IncLocal", "op index:u8", stackUnchanged},
	{"DecLocal", "op index:u8", stackUnchanged},

	{"ReturnTop", "op", "(value) -> (value)"},
	{"ReturnIntTop", "op", "(value) -> (value)"},
	{"ReturnFalse", "op", stackUnchanged},
	{"ReturnTrue", "op", stackUnchanged},
	{"Return", "op", stackUnchanged},

	{"Jump", "op offset:i16", stackUnchanged},
	{"JumpFalse", "op offset:i16", "(cond:bool) -> ()"},
	{"JumpTrue", "op offset:i16", "(cond:bool) -> ()"},

	{"SetVariadicLen", "op len:u8", stackUnchanged},
	{"CallNative", "op funcid:u16", "(args...) -> (results...)"},

	{"IsNil", "op", "(value) -> (result:bool)"},
	{"IsNotNil", "op", "(value) -> (result:bool)"},

	{"Not", "op", "(value:bool) -> (result:bool)"},

	{"EqInt", "op", "(x:int y:int) -> (result:bool)"},
	{"NotEqInt", "op", "(x:int y:int) -> (result:bool)"},
	{"GtInt", "op", "(x:int y:int) -> (result:bool)"},
	{"GtEqInt", "op", "(x:int y:int) -> (result:bool)"},
	{"LtInt", "op", "(x:int y:int) -> (result:bool)"},
	{"LtEqInt", "op", "(x:int y:int) -> (result:bool)"},

	{"EqString", "op", "(x:string y:string) -> (result:bool)"},
	{"NotEqString", "op", "(x:string y:string) -> (result:bool)"},

	{"Concat", "op", "(x:string y:string) -> (result:string)"},
	{"Add", "op", "(x:int y:int) -> (result:int)"},
	{"Sub", "op", "(x:int y:int) -> (result:int)"},

	{"StringSlice", "op", "(s:string from:int to:int) -> (result:string)"},
	{"StringSliceFrom", "op", "(s:string from:int) -> (result:string)"},
	{"StringSliceTo", "op", "(s:string to:int) -> (result:string)"},
	{"StringLen", "op", "(s:string) -> (result:int)"},
}

type opcodeProto struct {
	name  string
	enc   string
	stack string
}

type encodingInfo struct {
	width int
	parts int
}

type opcodeInfo struct {
	Opcode    byte
	Name      string
	Enc       string
	EncString string
	Stack     string
	Width     int
}

const stackUnchanged = ""

var fileTemplate = template.Must(template.New("opcodes.go").Parse(`// Code generated "gen_opcodes.go"; DO NOT EDIT.

package quasigo

//go:generate stringer -type=opcode -trimprefix=op
type opcode byte

const (
	opInvalid opcode = 0
{{ range .Opcodes }}
	// Encoding: {{.EncString}}
	// Stack effect: {{ if .Stack}}{{.Stack}}{{else}}unchanged{{end}}
	op{{ .Name }} opcode = {{.Opcode}}
{{ end -}}
)

type opcodeInfo struct {
	width int
}

var opcodeInfoTable = [256]opcodeInfo{
	opInvalid: {width: 1},

{{ range .Opcodes -}}
	op{{.Name}}: {width: {{.Width}}},
{{ end }}
}
`))

func main() {
	opcodes := make([]opcodeInfo, len(opcodePrototypes))
	for i, proto := range opcodePrototypes {
		opcode := byte(i + 1)
		encInfo := decodeEnc(proto.enc)
		var encString string
		if encInfo.parts == 1 {
			encString = fmt.Sprintf("0x%02x (width=%d)", opcode, encInfo.width)
		} else {
			encString = fmt.Sprintf("0x%02x %s (width=%d)",
				opcode, strings.TrimPrefix(proto.enc, "op "), encInfo.width)
		}

		opcodes[i] = opcodeInfo{
			Opcode:    opcode,
			Name:      proto.name,
			Enc:       proto.enc,
			EncString: encString,
			Stack:     proto.stack,
			Width:     encInfo.width,
		}
	}

	var buf bytes.Buffer
	err := fileTemplate.Execute(&buf, map[string]interface{}{
		"Opcodes": opcodes,
	})
	if err != nil {
		log.Panicf("execute template: %v", err)
	}
	writeFile("opcodes.gen.go", buf.Bytes())
}

func decodeEnc(enc string) encodingInfo {
	fields := strings.Fields(enc)
	width := 0
	for _, f := range fields {
		parts := strings.Split(f, ":")
		var typ string
		if len(parts) == 2 {
			typ = parts[1]
		} else {
			typ = "u8"
		}
		switch typ {
		case "i8", "u8":
			width++
		case "i16", "u16":
			width += 2
		default:
			panic(fmt.Sprintf("unknown op argument type: %s", typ))
		}
	}
	return encodingInfo{width: width, parts: len(fields)}
}

func writeFile(filename string, data []byte) {
	pretty, err := format.Source(data)
	if err != nil {
		log.Panicf("gofmt: %v", err)
	}
	if err := ioutil.WriteFile(filename, pretty, 0666); err != nil {
		log.Panicf("write %s: %v", filename, err)
	}
}
