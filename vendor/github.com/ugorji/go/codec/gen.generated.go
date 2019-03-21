// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// DO NOT EDIT. THIS FILE IS AUTO-GENERATED FROM gen-dec-(map|array).go.tmpl

const genDecMapTmpl = `
{{var "v"}} := *{{ .Varname }}
{{var "l"}} := r.ReadMapStart()
{{var "bh"}} := z.DecBasicHandle()
if {{var "v"}} == nil {
	{{var "rl"}} := z.DecInferLen({{var "l"}}, {{var "bh"}}.MaxInitLen, {{ .Size }})
	{{var "v"}} = make(map[{{ .KTyp }}]{{ .Typ }}, {{var "rl"}})
	*{{ .Varname }} = {{var "v"}}
}
var {{var "mk"}} {{ .KTyp }}
var {{var "mv"}} {{ .Typ }}
var {{var "mg"}}, {{var "mdn"}} {{if decElemKindPtr}}, {{var "ms"}}, {{var "mok"}}{{end}} bool
if {{var "bh"}}.MapValueReset {
	{{if decElemKindPtr}}{{var "mg"}} = true
	{{else if decElemKindIntf}}if !{{var "bh"}}.InterfaceReset { {{var "mg"}} = true }
	{{else if not decElemKindImmutable}}{{var "mg"}} = true
	{{end}} }
if {{var "l"}} != 0 {
{{var "hl"}} := {{var "l"}} > 0 
	for {{var "j"}} := 0; ({{var "hl"}} && {{var "j"}} < {{var "l"}}) || !({{var "hl"}} || r.CheckBreak()); {{var "j"}}++ {
	r.ReadMapElemKey() {{/* z.DecSendContainerState(codecSelfer_containerMapKey{{ .Sfx }}) */}}
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
	r.ReadMapElemValue() {{/* z.DecSendContainerState(codecSelfer_containerMapValue{{ .Sfx }}) */}}
	{{var "mdn"}} = false
	{{ $x := printf "%vmv%v" .TempVar .Rand }}{{ $y := printf "%vmdn%v" .TempVar .Rand }}{{ decLineVar $x $y }}
	if {{var "mdn"}} {
		if {{ var "bh" }}.DeleteOnNilMapValue { delete({{var "v"}}, {{var "mk"}}) } else { {{var "v"}}[{{var "mk"}}] = {{decElemZero}} }
	} else if {{if decElemKindPtr}} {{var "ms"}} && {{end}} {{var "v"}} != nil {
		{{var "v"}}[{{var "mk"}}] = {{var "mv"}}
	}
}
} // else len==0: TODO: Should we clear map entries?
r.ReadMapEnd() {{/* z.DecSendContainerState(codecSelfer_containerMapEnd{{ .Sfx }}) */}}
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
} else {
	{{var "hl"}} := {{var "l"}} > 0
	var {{var "rl"}} int; _ =  {{var "rl"}}
	{{if isSlice }} if {{var "hl"}} {
	if {{var "l"}} > cap({{var "v"}}) {
		{{var "rl"}} = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }})
		if {{var "rl"}} <= cap({{var "v"}}) {
			{{var "v"}} = {{var "v"}}[:{{var "rl"}}]
		} else {
			{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
		}
		{{var "c"}} = true
	} else if {{var "l"}} != len({{var "v"}}) {
		{{var "v"}} = {{var "v"}}[:{{var "l"}}]
		{{var "c"}} = true
	}
	} {{end}}
	var {{var "j"}} int 
    // var {{var "dn"}} bool 
	for ; ({{var "hl"}} && {{var "j"}} < {{var "l"}}) || !({{var "hl"}} || r.CheckBreak()); {{var "j"}}++ {
		{{if not isArray}} if {{var "j"}} == 0 && len({{var "v"}}) == 0 {
			if {{var "hl"}} {
				{{var "rl"}} = z.DecInferLen({{var "l"}}, z.DecBasicHandle().MaxInitLen, {{ .Size }})
			} else {
				{{var "rl"}} = 8
			}
			{{var "v"}} = make([]{{ .Typ }}, {{var "rl"}})
			{{var "c"}} = true 
		}{{end}}
		{{var "h"}}.ElemContainerState({{var "j"}})
        // {{var "dn"}} = r.TryDecodeAsNil()
        {{if isChan}}{{ $x := printf "%[1]vv%[2]v" .TempVar .Rand }}var {{var $x}} {{ .Typ }}
		{{ decLineVar $x }}
		{{var "v"}} <- {{ $x }}
        {{else}}
		// if indefinite, etc, then expand the slice if necessary
		var {{var "db"}} bool
		if {{var "j"}} >= len({{var "v"}}) {
			{{if isSlice }} {{var "v"}} = append({{var "v"}}, {{ zero }}); {{var "c"}} = true
			{{else}} z.DecArrayCannotExpand(len(v), {{var "j"}}+1); {{var "db"}} = true
			{{end}}
		}
		if {{var "db"}} {
			z.DecSwallow()
		} else {
			{{ $x := printf "%[1]vv%[2]v[%[1]vj%[2]v]" .TempVar .Rand }}{{ decLineVar $x }}
		}
        {{end}}
	}
	{{if isSlice}} if {{var "j"}} < len({{var "v"}}) {
		{{var "v"}} = {{var "v"}}[:{{var "j"}}]
		{{var "c"}} = true
	} else if {{var "j"}} == 0 && {{var "v"}} == nil {
		{{var "v"}} = make([]{{ .Typ }}, 0)
		{{var "c"}} = true
	} {{end}}
}
{{var "h"}}.End()
{{if not isArray }}if {{var "c"}} { 
	*{{ .Varname }} = {{var "v"}}
}{{end}}

`

