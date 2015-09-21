package swagger

// Copyright 2014 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

type ParameterSorter []Parameter

func (s ParameterSorter) Len() int {
	return len(s)
}
func (s ParameterSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

var typeToSortKey = map[string]string{
	"path":   "A",
	"query":  "B",
	"form":   "C",
	"header": "D",
	"body":   "E",
}

func (s ParameterSorter) Less(i, j int) bool {
	// use ordering path,query,form,header,body
	pi := s[i]
	pj := s[j]
	return typeToSortKey[pi.ParamType]+pi.Name < typeToSortKey[pj.ParamType]+pj.Name
}
