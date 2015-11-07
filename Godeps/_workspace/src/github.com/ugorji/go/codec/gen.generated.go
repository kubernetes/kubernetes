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
var {{var "mg"}} bool
if {{var "bh"}}.MapValueReset {
	{{if decElemKindPtr}}{{var "mg"}} = true
	{{else if decElemKindIntf}}if !{{var "bh"}}.InterfaceReset { {{var "mg"}} = true }
	{{else if not decElemKindImmutable}}{{var "mg"}} = true
	{{end}} }
if {{var "l"}} > 0  {
for {{var "j"}} := 0; {{var "j"}} < {{var "l"}}; {{var "j"}}++ {
	{{ $x := printf "%vmk%v" .TempVar .Rand }}{{ decLineVarK $x }}
{{ if eq .KTyp "interface{}" }}{{/* // special case if a byte array. */}}if {{var "bv"}}, {{var "bok"}} := {{var "mk"}}.([]byte); {{var "bok"}} {
		{{var "mk"}} = string({{var "bv"}})
	}{{ end }}
	if {{var "mg"}} {
		{{var "mv"}} = {{var "v"}}[{{var "mk"}}]
	} {{if not decElemKindImmutable}}else { {{var "mv"}} = {{decElemZero}} }{{end}}
	{{ $x := printf "%vmv%v" .TempVar .Rand }}{{ decLineVar $x }}
	if {{var "v"}} != nil {
		{{var "v"}}[{{var "mk"}}] = {{var "mv"}}
	}
}
} else if {{var "l"}} < 0  {
for {{var "j"}} := 0; !r.CheckBreak(); {{var "j"}}++ {
	{{ $x := printf "%vmk%v" .TempVar .Rand }}{{ decLineVarK $x }}
{{ if eq .KTyp "interface{}" }}{{/* // special case if a byte array. */}}if {{var "bv"}}, {{var "bok"}} := {{var "mk"}}.([]byte); {{var "bok"}} {
		{{var "mk"}} = string({{var "bv"}})
	}{{ end }}
	if {{var "mg"}} {
		{{var "mv"}} = {{var "v"}}[{{var "mk"}}]
	} {{if not decElemKindImmutable}}else { {{var "mv"}} = {{decElemZero}} }{{end}}
	{{ $x := printf "%vmv%v" .TempVar .Rand }}{{ decLineVar $x }}
	if {{var "v"}} != nil {
		{{var "v"}}[{{var "mk"}}] = {{var "mv"}}
	}
}
r.ReadEnd()
} // else len==0: TODO: Should we clear map entries?
`

const genDecListTmpl = `
{{var "v"}} := {{ if not isArray}}*{{ end }}{{ .Varname }}
{{var "h"}}, {{var "l"}} := z.DecSliceHelperStart() {{/* // helper, containerLenS */}}

var {{var "rr"}}, {{var "rl"}} int {{/* // num2read, length of slice/array/chan */}}
var {{var "c"}}, {{var "rt"}} bool {{/* // changed, truncated */}}
_, _, _ = {{var "c"}}, {{var "rt"}}, {{var "rl"}}
{{var "rr"}} = {{var "l"}}
{{/* rl is NOT used. Only used for getting DecInferLen. len(r) used directly in code */}}

{{ if not isArray }}if {{var "v"}} == nil {
	if {{var "rl"}}, {{var "rt"}} = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }}); {{var "rt"}} {
		{{var "rr"}} = {{var "rl"}}
	}
	{{var "v"}} = make({{ .CTyp }}, {{var "rl"}})
	{{var "c"}} = true 
} 
{{ end }}
if {{var "l"}} == 0 { {{ if isSlice }}
	if len({{var "v"}}) != 0 { 
		{{var "v"}} = {{var "v"}}[:0] 
		{{var "c"}} = true 
	} {{ end }}
} else if {{var "l"}} > 0 {
	{{ if isChan }}
	for {{var "r"}} := 0; {{var "r"}} < {{var "l"}}; {{var "r"}}++ {
		var {{var "t"}} {{ .Typ }}
		{{ $x := printf "%st%s" .TempVar .Rand }}{{ decLineVar $x }}
		{{var "v"}} <- {{var "t"}} 
	{{ else }} 
	if {{var "l"}} > cap({{var "v"}}) {
		{{ if isArray }}z.DecArrayCannotExpand(len({{var "v"}}), {{var "l"}})
		{{ else }}{{var "rl"}}, {{var "rt"}} = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }})
		{{ if .Immutable }}
		{{var "v2"}} := {{var "v"}}
		{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
		if len({{var "v"}}) > 0 {
			copy({{var "v"}}, {{var "v2"}}[:cap({{var "v2"}})])
		}
		{{ else }}{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
		{{ end }}{{var "c"}} = true 
		{{ end }}
		{{var "rr"}} = len({{var "v"}})
	} else if {{var "l"}} != len({{var "v"}}) {
		{{ if isSlice }}{{var "v"}} = {{var "v"}}[:{{var "l"}}]
		{{var "c"}} = true {{ end }}
	}
	{{var "j"}} := 0
	for ; {{var "j"}} < {{var "rr"}} ; {{var "j"}}++ {
		{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
	}
	{{ if isArray }}for ; {{var "j"}} < {{var "l"}} ; {{var "j"}}++ {
		z.DecSwallow()
	}
	{{ else }}if {{var "rt"}} { {{/* means that it is mutable and slice */}}
		for ; {{var "j"}} < {{var "l"}} ; {{var "j"}}++ {
			{{var "v"}} = append({{var "v"}}, {{ zero}})
			{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
		}
	}
	{{ end }}
	{{ end }}{{/* closing 'if not chan' */}}
} else {
	for {{var "j"}} := 0; !r.CheckBreak(); {{var "j"}}++ {
		if {{var "j"}} >= len({{var "v"}}) {
			{{ if isArray }}z.DecArrayCannotExpand(len({{var "v"}}), {{var "j"}}+1)
			{{ else if isSlice}}{{var "v"}} = append({{var "v"}}, {{zero}})// var {{var "z"}} {{ .Typ }}
			{{var "c"}} = true {{ end }}
		}
		{{ if isChan}}
		var {{var "t"}} {{ .Typ }}
		{{ $x := printf "%st%s" .TempVar .Rand }}{{ decLineVar $x }}
		{{var "v"}} <- {{var "t"}} 
		{{ else }}
		if {{var "j"}} < len({{var "v"}}) {
			{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
		} else {
			z.DecSwallow()
		}
		{{ end }}
	}
	{{var "h"}}.End()
}
{{ if not isArray }}if {{var "c"}} { 
	*{{ .Varname }} = {{var "v"}}
}{{ end }}

`

