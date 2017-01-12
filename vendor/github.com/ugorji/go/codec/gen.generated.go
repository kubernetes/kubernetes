// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// DO NOT EDIT. THIS FILE IS AUTO-GENERATED FROM gen-dec-(map|array).go.tmpl

const genDecMapTmpl = `
{{var "v"}} := *{{ .Varname }}
{{var "l"}} := r.ReadMapStart()
{{var "bh"}} := z.DecBasicHandle()
if {{var "v"}} == nil {
	{{var "rl"}}, _ := z.DecInferLen({{var "l"}}, {{var "bh"}}.MaxInitLen, {{ .Size }})
	{{var "v"}} = make(map[{{ .KTyp }}]{{ .Typ }}, {{var "rl"}})
	*{{ .Varname }} = {{var "v"}}
}
var {{var "mk"}} {{ .KTyp }}
var {{var "mv"}} {{ .Typ }}
var {{var "mg"}} {{if decElemKindPtr}}, {{var "ms"}}, {{var "mok"}}{{end}} bool
if {{var "bh"}}.MapValueReset {
	{{if decElemKindPtr}}{{var "mg"}} = true
	{{else if decElemKindIntf}}if !{{var "bh"}}.InterfaceReset { {{var "mg"}} = true }
	{{else if not decElemKindImmutable}}{{var "mg"}} = true
	{{end}} }
if {{var "l"}} > 0  {
for {{var "j"}} := 0; {{var "j"}} < {{var "l"}}; {{var "j"}}++ {
	z.DecSendContainerState(codecSelfer_containerMapKey{{ .Sfx }})
	{{ $x := printf "%vmk%v" .TempVar .Rand }}{{ decLineVarK $x }}
{{ if eq .KTyp "interface{}" }}{{/* // special case if a byte array. */}}if {{var "bv"}}, {{var "bok"}} := {{var "mk"}}.([]byte); {{var "bok"}} {
		{{var "mk"}} = string({{var "bv"}})
	}{{ end }}{{if decElemKindPtr}}
	{{var "ms"}} = true{{end}}
	if {{var "mg"}} {
		{{if decElemKindPtr}}{{var "mv"}}, {{var "mok"}} = {{var "v"}}[{{var "mk"}}] 
		if {{var "mok"}} {
			{{var "ms"}} = false
		} {{else}}{{var "mv"}} = {{var "v"}}[{{var "mk"}}] {{end}}
	} {{if not decElemKindImmutable}}else { {{var "mv"}} = {{decElemZero}} }{{end}}
	z.DecSendContainerState(codecSelfer_containerMapValue{{ .Sfx }})
	{{ $x := printf "%vmv%v" .TempVar .Rand }}{{ decLineVar $x }}
	if {{if decElemKindPtr}} {{var "ms"}} && {{end}} {{var "v"}} != nil {
		{{var "v"}}[{{var "mk"}}] = {{var "mv"}}
	}
}
} else if {{var "l"}} < 0  {
for {{var "j"}} := 0; !r.CheckBreak(); {{var "j"}}++ {
	z.DecSendContainerState(codecSelfer_containerMapKey{{ .Sfx }})
	{{ $x := printf "%vmk%v" .TempVar .Rand }}{{ decLineVarK $x }}
{{ if eq .KTyp "interface{}" }}{{/* // special case if a byte array. */}}if {{var "bv"}}, {{var "bok"}} := {{var "mk"}}.([]byte); {{var "bok"}} {
		{{var "mk"}} = string({{var "bv"}})
	}{{ end }}{{if decElemKindPtr}}
	{{var "ms"}} = true {{ end }}
	if {{var "mg"}} {
		{{if decElemKindPtr}}{{var "mv"}}, {{var "mok"}} = {{var "v"}}[{{var "mk"}}] 
		if {{var "mok"}} {
			{{var "ms"}} = false
		} {{else}}{{var "mv"}} = {{var "v"}}[{{var "mk"}}] {{end}}
	} {{if not decElemKindImmutable}}else { {{var "mv"}} = {{decElemZero}} }{{end}}
	z.DecSendContainerState(codecSelfer_containerMapValue{{ .Sfx }})
	{{ $x := printf "%vmv%v" .TempVar .Rand }}{{ decLineVar $x }}
	if {{if decElemKindPtr}} {{var "ms"}} && {{end}} {{var "v"}} != nil {
		{{var "v"}}[{{var "mk"}}] = {{var "mv"}}
	}
}
} // else len==0: TODO: Should we clear map entries?
z.DecSendContainerState(codecSelfer_containerMapEnd{{ .Sfx }})
`

const genDecListTmpl = `
{{var "v"}} := {{if not isArray}}*{{end}}{{ .Varname }}
{{var "h"}}, {{var "l"}} := z.DecSliceHelperStart() {{/* // helper, containerLenS */}}{{if not isArray}}
var {{var "c"}} bool {{/* // changed */}}
_ = {{var "c"}}{{end}}
if {{var "l"}} == 0 {
	{{if isSlice }}if {{var "v"}} == nil {
		{{var "v"}} = []{{ .Typ }}{}
		{{var "c"}} = true
	} else if len({{var "v"}}) != 0 {
		{{var "v"}} = {{var "v"}}[:0]
		{{var "c"}} = true
	} {{end}} {{if isChan }}if {{var "v"}} == nil {
		{{var "v"}} = make({{ .CTyp }}, 0)
		{{var "c"}} = true
	} {{end}}
} else if {{var "l"}} > 0 {
	{{if isChan }}if {{var "v"}} == nil {
		{{var "rl"}}, _ = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }})
		{{var "v"}} = make({{ .CTyp }}, {{var "rl"}})
		{{var "c"}} = true
	}
	for {{var "r"}} := 0; {{var "r"}} < {{var "l"}}; {{var "r"}}++ {
		{{var "h"}}.ElemContainerState({{var "r"}})
		var {{var "t"}} {{ .Typ }}
		{{ $x := printf "%st%s" .TempVar .Rand }}{{ decLineVar $x }}
		{{var "v"}} <- {{var "t"}}
	}
	{{ else }}	var {{var "rr"}}, {{var "rl"}} int {{/* // num2read, length of slice/array/chan */}}
	var {{var "rt"}} bool {{/* truncated */}}
	_, _ = {{var "rl"}}, {{var "rt"}}
	{{var "rr"}} = {{var "l"}} // len({{var "v"}})
	if {{var "l"}} > cap({{var "v"}}) {
		{{if isArray }}z.DecArrayCannotExpand(len({{var "v"}}), {{var "l"}})
		{{ else }}{{if not .Immutable }}
		{{var "rg"}} := len({{var "v"}}) > 0
		{{var "v2"}} := {{var "v"}} {{end}}
		{{var "rl"}}, {{var "rt"}} = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }})
		if {{var "rt"}} {
			if {{var "rl"}} <= cap({{var "v"}}) {
				{{var "v"}} = {{var "v"}}[:{{var "rl"}}]
			} else {
				{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
			}
		} else {
			{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
		}
		{{var "c"}} = true
		{{var "rr"}} = len({{var "v"}}) {{if not .Immutable }}
			if {{var "rg"}} { copy({{var "v"}}, {{var "v2"}}) } {{end}} {{end}}{{/* end not Immutable, isArray */}}
	} {{if isSlice }} else if {{var "l"}} != len({{var "v"}}) {
		{{var "v"}} = {{var "v"}}[:{{var "l"}}]
		{{var "c"}} = true
	} {{end}}	{{/* end isSlice:47 */}} 
	{{var "j"}} := 0
	for ; {{var "j"}} < {{var "rr"}} ; {{var "j"}}++ {
		{{var "h"}}.ElemContainerState({{var "j"}})
		{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
	}
	{{if isArray }}for ; {{var "j"}} < {{var "l"}} ; {{var "j"}}++ {
		{{var "h"}}.ElemContainerState({{var "j"}})
		z.DecSwallow()
	}
	{{ else }}if {{var "rt"}} {
		for ; {{var "j"}} < {{var "l"}} ; {{var "j"}}++ {
			{{var "v"}} = append({{var "v"}}, {{ zero}})
			{{var "h"}}.ElemContainerState({{var "j"}})
			{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
		}
	} {{end}} {{/* end isArray:56 */}}
	{{end}} {{/* end isChan:16 */}}
} else { {{/* len < 0 */}}
	{{var "j"}} := 0
	for ; !r.CheckBreak(); {{var "j"}}++ {
		{{if isChan }}
		{{var "h"}}.ElemContainerState({{var "j"}})
		var {{var "t"}} {{ .Typ }}
		{{ $x := printf "%st%s" .TempVar .Rand }}{{ decLineVar $x }}
		{{var "v"}} <- {{var "t"}} 
		{{ else }}
		if {{var "j"}} >= len({{var "v"}}) {
			{{if isArray }}z.DecArrayCannotExpand(len({{var "v"}}), {{var "j"}}+1)
			{{ else }}{{var "v"}} = append({{var "v"}}, {{zero}})// var {{var "z"}} {{ .Typ }}
			{{var "c"}} = true {{end}}
		}
		{{var "h"}}.ElemContainerState({{var "j"}})
		if {{var "j"}} < len({{var "v"}}) {
			{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
		} else {
			z.DecSwallow()
		}
		{{end}}
	}
	{{if isSlice }}if {{var "j"}} < len({{var "v"}}) {
		{{var "v"}} = {{var "v"}}[:{{var "j"}}]
		{{var "c"}} = true
	} else if {{var "j"}} == 0 && {{var "v"}} == nil {
		{{var "v"}} = []{{ .Typ }}{}
		{{var "c"}} = true
	}{{end}}
}
{{var "h"}}.End()
{{if not isArray }}if {{var "c"}} { 
	*{{ .Varname }} = {{var "v"}}
}{{end}}
`

