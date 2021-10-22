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

package delta

import (
	"sort"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
)

// Content defines the set of exported constants, funcs, and structs.
type Content struct {
	exports.Content

	// contains the names of structs that are modified in whole (i.e. new/removed)
	CompleteStructs []string `json:"newStructs,omitempty"`
}

// NewContent returns an initialized Content object.
func NewContent() Content {
	return Content{
		Content: exports.NewContent(),
	}
}

// GetModifiedStructs returns the subset, if any, of structs that are modified.
func (c Content) GetModifiedStructs() map[string]exports.Struct {
	if len(c.CompleteStructs) == 0 {
		return c.Structs
	}
	ms := map[string]exports.Struct{}
	for k, v := range c.Structs {
		if contains(c.CompleteStructs, k) {
			continue
		}
		ms[k] = v
	}
	return ms
}

// returns true if sl contains x
func contains(sl []string, x string) bool {
	for _, s := range sl {
		if s == x {
			return true
		}
	}
	return false
}

// GetExports returns a exports.Content struct containing all exports in rhs that aren't in lhs.
// This includes any new fields added to structs or methods added to interfaces.
func GetExports(lhs, rhs exports.Content) Content {
	nc := NewContent()

	for n, v := range rhs.Consts {
		if _, ok := lhs.Consts[n]; !ok {
			nc.Consts[n] = v
		}
	}

	for n, v := range rhs.Funcs {
		if _, ok := lhs.Funcs[n]; !ok {
			nc.Funcs[n] = v
		}
	}

	for n, v := range rhs.Interfaces {
		if _, ok := lhs.Interfaces[n]; !ok {
			nc.Interfaces[n] = v
		}
	}

	for n, v := range rhs.Structs {
		if _, ok := lhs.Structs[n]; !ok {
			nc.Structs[n] = v
			nc.CompleteStructs = append(nc.CompleteStructs, n)
		}
	}

	structFields := GetStructFields(lhs, rhs)
	if len(structFields) > 0 {
		for k, v := range structFields {
			nc.Structs[k] = v
		}
	}

	intMethods := GetInterfaceMethods(lhs, rhs)
	if len(intMethods) > 0 {
		for k, v := range intMethods {
			nc.Interfaces[k] = v
		}
	}

	sort.Strings(nc.CompleteStructs)
	return nc
}

// GetStructFields returns structs common to lhs and rhs where structs in rhs contain
// fields not in lhs.  Key is the struct type name, value contains the added content.
func GetStructFields(lhs, rhs exports.Content) map[string]exports.Struct {
	nf := map[string]exports.Struct{}

	for rhsKey, rhsVal := range rhs.Structs {
		if lhsStruct, ok := lhs.Structs[rhsKey]; ok {
			nc := exports.Struct{}
			for _, rhsAnon := range rhsVal.AnonymousFields {
				found := false
				for _, lhsAnon := range lhsStruct.AnonymousFields {
					if lhsAnon == rhsAnon {
						found = true
						break
					}
				}
				if !found {
					nc.AnonymousFields = append(nc.AnonymousFields, rhsAnon)
				}
			}
			for fn, fv := range rhsVal.Fields {
				if _, ok := lhsStruct.Fields[fn]; !ok {
					if nc.Fields == nil {
						nc.Fields = map[string]string{}
					}
					nc.Fields[fn] = fv
				}
			}
			// only add it if there's new content
			if len(nc.AnonymousFields) > 0 || len(nc.Fields) > 0 {
				nf[rhsKey] = nc
			}
		}
	}
	return nf
}

// GetInterfaceMethods returns interfaces common to lhs and rhs where interfaces in rhs contain
// methods not in lhs.  Key is the interface type name, value contains the added content.
func GetInterfaceMethods(lhs, rhs exports.Content) map[string]exports.Interface {
	ni := map[string]exports.Interface{}

	for rhsKey, rhsValue := range rhs.Interfaces {
		if lhsInterface, ok := lhs.Interfaces[rhsKey]; ok {
			nc := exports.Interface{}
			for in, iv := range rhsValue.Methods {
				if _, ok := lhsInterface.Methods[in]; !ok {
					if nc.Methods == nil {
						nc.Methods = map[string]exports.Func{}
					}
					nc.Methods[in] = iv
				}
			}
			// only add it if there's new content
			if len(nc.Methods) > 0 {
				ni[rhsKey] = nc
			}
		}
	}
	return ni
}

// Signature contains the details of how a type signature changed (e.g. From:"int" To:"string").
type Signature struct {
	// From contains the originial signature.
	From string `json:"from"`

	// To contains the new signature.
	To string `json:"to"`
}

// GetConstTypeChanges returns a collection of const where the type has changed.
// Key is the const name, value contains the type change information.
func GetConstTypeChanges(lhs, rhs exports.Content) map[string]Signature {
	cc := map[string]Signature{}

	for rhsKey, rhsValue := range rhs.Consts {
		if _, ok := lhs.Consts[rhsKey]; !ok {
			continue
		}
		if lhs.Consts[rhsKey].Type != rhsValue.Type {
			cc[rhsKey] = Signature{
				From: lhs.Consts[rhsKey].Type,
				To:   rhsValue.Type,
			}
		}
	}
	return cc
}

// None is the value used for functions with no parameters and/or no return values.
const None = "<none>"

// FuncSig contains the details of how a function's signature changed.
type FuncSig struct {
	// Params contains the parameter signature changes, may be nil.
	Params *Signature `json:"params,omitempty"`

	// Returns contains the return signature changes, may be nil.
	Returns *Signature `json:"returns,omitempty"`
}

func (fs FuncSig) isEmpty() bool {
	return fs.Params == nil && fs.Returns == nil
}

// GetFuncSigChanges returns a collection of functions that contain signature changes (params and/or returns).
// Key is the function name, value contains the signature change information.
func GetFuncSigChanges(lhs, rhs exports.Content) map[string]FuncSig {
	fsc := map[string]FuncSig{}

	for rhsKey, rhsValue := range rhs.Funcs {
		if _, ok := lhs.Funcs[rhsKey]; !ok {
			continue
		}
		sig := FuncSig{}
		if !safeStrCmp(lhs.Funcs[rhsKey].Params, rhsValue.Params) {
			sig.Params = &Signature{
				From: safeFuncSig(lhs.Funcs[rhsKey].Params),
				To:   safeFuncSig(rhsValue.Params),
			}
		}
		if !safeStrCmp(lhs.Funcs[rhsKey].Returns, rhsValue.Returns) {
			sig.Returns = &Signature{
				From: safeFuncSig(lhs.Funcs[rhsKey].Returns),
				To:   safeFuncSig(rhsValue.Returns),
			}
		}

		if !sig.isEmpty() {
			fsc[rhsKey] = sig
		}
	}
	return fsc
}

// InterfaceDef contains a collection of interface methods with signature changes.
// Key is the method name, value contains the signature change information.
type InterfaceDef struct {
	MethodSigs map[string]FuncSig `json:"funcSig"`
}

// GetInterfaceMethodSigChanges returns a collection of interfaces with method signature changes.
// Key is the interface name, value contains the method signature change information.
func GetInterfaceMethodSigChanges(lhs, rhs exports.Content) map[string]InterfaceDef {
	isc := map[string]InterfaceDef{}

	for rhsKey, rhsValue := range rhs.Interfaces {
		if _, ok := lhs.Interfaces[rhsKey]; !ok {
			continue
		}
		id := InterfaceDef{}

		for rhsMethod, rhsSig := range rhsValue.Methods {
			if _, ok := lhs.Interfaces[rhsKey].Methods[rhsMethod]; !ok {
				continue
			}
			sig := FuncSig{}
			if !safeStrCmp(lhs.Interfaces[rhsKey].Methods[rhsMethod].Params, rhsSig.Params) {
				sig.Params = &Signature{
					From: safeFuncSig(lhs.Interfaces[rhsKey].Methods[rhsMethod].Params),
					To:   safeFuncSig(rhsSig.Params),
				}
			}
			if !safeStrCmp(lhs.Interfaces[rhsKey].Methods[rhsMethod].Returns, rhsSig.Returns) {
				sig.Returns = &Signature{
					From: safeFuncSig(lhs.Interfaces[rhsKey].Methods[rhsMethod].Returns),
					To:   safeFuncSig(rhsSig.Returns),
				}
			}

			if !sig.isEmpty() {
				if id.MethodSigs == nil {
					id.MethodSigs = map[string]FuncSig{}
				}
				id.MethodSigs[rhsMethod] = sig
			}
		}

		if len(id.MethodSigs) > 0 {
			isc[rhsKey] = id
		}
	}
	return isc
}

// StructDef contains a collection of fields within a struct where the field's type has changed.
// Key is the field name, value contains the signature change information.
type StructDef struct {
	Fields map[string]Signature `json:"fields"`
}

// GetStructFieldChanges returns a collection of structs with fields that changed their type.
// Key is the struct name, value contains fields with signature changes.
func GetStructFieldChanges(lhs, rhs exports.Content) map[string]StructDef {
	sfc := map[string]StructDef{}

	for rhsKey, rhsValue := range rhs.Structs {
		if _, ok := lhs.Structs[rhsKey]; !ok {
			continue
		}
		sd := StructDef{}

		for rhsField, rhsSig := range rhsValue.Fields {
			if _, ok := lhs.Structs[rhsKey].Fields[rhsField]; !ok {
				continue
			}
			if lhs.Structs[rhsKey].Fields[rhsField] != rhsSig {
				if sd.Fields == nil {
					sd.Fields = map[string]Signature{}
				}
				sd.Fields[rhsField] = Signature{
					From: lhs.Structs[rhsKey].Fields[rhsField],
					To:   rhsSig,
				}
			}
		}

		if len(sd.Fields) > 0 {
			sfc[rhsKey] = sd
		}
	}
	return sfc
}

func safeFuncSig(s *string) string {
	if s == nil {
		return None
	}
	return *s
}

func safeStrCmp(lhs, rhs *string) bool {
	if lhs == nil && rhs == nil {
		return true
	}
	if lhs == nil || rhs == nil {
		return false
	}
	return *lhs == *rhs
}
