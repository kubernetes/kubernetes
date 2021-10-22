// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package report

import (
	"fmt"
	"sort"
	"strings"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/delta"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
)

type mdWriter struct {
	sb strings.Builder
	nl bool
}

func (md *mdWriter) checkNL() {
	if md.nl {
		md.sb.WriteString("\n")
		md.nl = false
	}
}

func (md *mdWriter) WriteHeader(h string) {
	md.checkNL()
	md.sb.WriteString("## ")
	md.sb.WriteString(h)
	md.sb.WriteString("\n\n")
}

func (md *mdWriter) WriteSubheader(sh string) {
	md.checkNL()
	md.sb.WriteString("### ")
	md.sb.WriteString(sh)
	md.sb.WriteString("\n\n")
}

func (md *mdWriter) WriteLine(s string) {
	md.nl = true
	md.sb.WriteString(s)
	md.sb.WriteString("\n")
}

func (md *mdWriter) String() string {
	return md.sb.String()
}

// creates a markdown-formatted report from the specified package
func formatAsMarkdown(p Package) string {
	md := mdWriter{}
	writeBreakingChanges(p, &md)
	writeNewContent(p, &md)
	return md.String()
}

// writes all breaking changes
func writeBreakingChanges(p Package, md *mdWriter) {
	if !p.HasBreakingChanges() {
		return
	}
	md.WriteHeader("Breaking Changes")
	writeRemovedContent(p.BreakingChanges.Removed, md)
	writeSigChanges(p.BreakingChanges, md)
}

// writes the subset of breaking changes pertaining to removed content
func writeRemovedContent(removed *delta.Content, md *mdWriter) {
	if removed == nil {
		return
	}
	writeConsts(removed.Consts, "Removed Constants", md)
	writeFuncs(removed.Funcs, "Removed Funcs", md)
	writeStructs(removed, "Removed Structs", "Removed Struct Fields", md)
}

// writes the subset of breaking changes pertaining to signature changes
func writeSigChanges(bc *BreakingChanges, md *mdWriter) {
	if len(bc.Consts) == 0 && len(bc.Funcs) == 0 && len(bc.Structs) == 0 {
		return
	}
	md.WriteHeader("Signature Changes")
	if len(bc.Consts) > 0 {
		items := make([]string, len(bc.Consts))
		i := 0
		for k, v := range bc.Consts {
			items[i] = fmt.Sprintf("1. %s changed type from %s to %s", k, v.From, v.To)
			i++
		}
		sort.Strings(items)
		md.WriteSubheader("Const Types")
		for _, item := range items {
			md.WriteLine(item)
		}
	}
	if len(bc.Funcs) > 0 {
		// first get all the funcs so we can sort them
		items := make([]string, len(bc.Funcs))
		i := 0
		for k := range bc.Funcs {
			items[i] = k
			i++
		}
		sort.Strings(items)
		md.WriteSubheader("Funcs")
		for _, item := range items {
			// now add params/returns info
			changes := bc.Funcs[item]
			if changes.Params != nil {
				item = fmt.Sprintf("%s\n\t- Params\n\t\t- From: %s\n\t\t- To: %s", item, changes.Params.From, changes.Params.To)
			}
			if changes.Returns != nil {
				item = fmt.Sprintf("%s\n\t- Returns\n\t\t- From: %s\n\t\t- To: %s", item, changes.Returns.From, changes.Returns.To)
			}
			md.WriteLine(fmt.Sprintf("1. %s", item))
		}
	}
	if len(bc.Structs) > 0 {
		items := make([]string, 0, len(bc.Structs))
		for k, v := range bc.Structs {
			for f, d := range v.Fields {
				items = append(items, fmt.Sprintf("1. %s.%s changed type from %s to %s", k, f, d.From, d.To))
			}
		}
		sort.Strings(items)
		md.WriteSubheader("Struct Fields")
		for _, item := range items {
			md.WriteLine(item)
		}
	}
}

// writes all new content
func writeNewContent(p Package, md *mdWriter) {
	if !p.HasAdditiveChanges() {
		return
	}
	md.WriteHeader("New Content")
	writeConsts(p.AdditiveChanges.Consts, "New Constants", md)
	writeFuncs(p.AdditiveChanges.Funcs, "New Funcs", md)
	writeStructs(p.AdditiveChanges, "New Structs", "New Struct Fields", md)
}

// writes out const information formatted as TypeName.ConstName
func writeConsts(co map[string]exports.Const, subheader string, md *mdWriter) {
	if len(co) == 0 {
		return
	}
	items := make([]string, len(co))
	i := 0
	for c, t := range co {
		items[i] = fmt.Sprintf("1. %s.%s", t.Type, c)
		i++
	}
	sort.Strings(items)
	md.WriteSubheader(subheader)
	for _, item := range items {
		md.WriteLine(item)
	}
}

// writes out func information formatted as [receiver].FuncName([params]) [returns]
func writeFuncs(funcs map[string]exports.Func, subheader string, md *mdWriter) {
	if len(funcs) == 0 {
		return
	}
	items := make([]string, len(funcs))
	i := 0
	for k, v := range funcs {
		params := ""
		if v.Params != nil {
			params = *v.Params
		}
		returns := ""
		if v.Returns != nil {
			returns = *v.Returns
			if strings.Index(returns, ",") > -1 {
				returns = fmt.Sprintf("(%s)", returns)
			}
		}
		items[i] = fmt.Sprintf("1. %s(%s) %s", k, params, returns)
		i++
	}
	sort.Strings(items)
	md.WriteSubheader(subheader)
	for _, item := range items {
		md.WriteLine(item)
	}
}

// writes out struct information
// sheader1 is for added/removed struct types formatted as TypeName
// sheader2 is for added/removed struct fields formatted as TypeName.FieldName
func writeStructs(content *delta.Content, sheader1, sheader2 string, md *mdWriter) {
	if len(content.Structs) == 0 {
		return
	}
	md.WriteHeader("Struct Changes")
	if len(content.CompleteStructs) > 0 {
		md.WriteSubheader(sheader1)
		for _, s := range content.CompleteStructs {
			md.WriteLine(fmt.Sprintf("1. %s", s))
		}
	}
	modified := content.GetModifiedStructs()
	if len(modified) > 0 {
		md.WriteSubheader(sheader2)
		items := make([]string, 0, len(content.Structs)-len(content.CompleteStructs))
		for s, f := range modified {
			for _, af := range f.AnonymousFields {
				items = append(items, fmt.Sprintf("1. %s.%s", s, af))
			}
			for f := range f.Fields {
				items = append(items, fmt.Sprintf("1. %s.%s", s, f))
			}
		}
		sort.Strings(items)
		for _, item := range items {
			md.WriteLine(item)
		}
	}
}
