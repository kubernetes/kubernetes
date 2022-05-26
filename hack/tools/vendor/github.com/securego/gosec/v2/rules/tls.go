// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
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

//go:generate tlsconfig

package rules

import (
	"crypto/tls"
	"fmt"
	"go/ast"
	"go/types"
	"strconv"

	"github.com/securego/gosec/v2"
)

type insecureConfigTLS struct {
	gosec.MetaData
	MinVersion       int64
	MaxVersion       int64
	requiredType     string
	goodCiphers      []string
	actualMinVersion int64
	actualMaxVersion int64
}

func (t *insecureConfigTLS) ID() string {
	return t.MetaData.ID
}

func stringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

func (t *insecureConfigTLS) processTLSCipherSuites(n ast.Node, c *gosec.Context) *gosec.Issue {
	if ciphers, ok := n.(*ast.CompositeLit); ok {
		for _, cipher := range ciphers.Elts {
			if ident, ok := cipher.(*ast.SelectorExpr); ok {
				if !stringInSlice(ident.Sel.Name, t.goodCiphers) {
					err := fmt.Sprintf("TLS Bad Cipher Suite: %s", ident.Sel.Name)
					return gosec.NewIssue(c, ident, t.ID(), err, gosec.High, gosec.High)
				}
			}
		}
	}
	return nil
}

func (t *insecureConfigTLS) processTLSConfVal(n *ast.KeyValueExpr, c *gosec.Context) *gosec.Issue {
	if ident, ok := n.Key.(*ast.Ident); ok {
		switch ident.Name {
		case "InsecureSkipVerify":
			if node, ok := n.Value.(*ast.Ident); ok {
				if node.Name != "false" {
					return gosec.NewIssue(c, n, t.ID(), "TLS InsecureSkipVerify set true.", gosec.High, gosec.High)
				}
			} else {
				// TODO(tk): symbol tab look up to get the actual value
				return gosec.NewIssue(c, n, t.ID(), "TLS InsecureSkipVerify may be true.", gosec.High, gosec.Low)
			}

		case "PreferServerCipherSuites":
			if node, ok := n.Value.(*ast.Ident); ok {
				if node.Name == "false" {
					return gosec.NewIssue(c, n, t.ID(), "TLS PreferServerCipherSuites set false.", gosec.Medium, gosec.High)
				}
			} else {
				// TODO(tk): symbol tab look up to get the actual value
				return gosec.NewIssue(c, n, t.ID(), "TLS PreferServerCipherSuites may be false.", gosec.Medium, gosec.Low)
			}

		case "MinVersion":
			if d, ok := n.Value.(*ast.Ident); ok {
				obj := d.Obj
				if obj == nil {
					for _, f := range c.PkgFiles {
						obj = f.Scope.Lookup(d.Name)
						if obj != nil {
							break
						}
					}
				}
				if vs, ok := obj.Decl.(*ast.ValueSpec); ok && len(vs.Values) > 0 {
					if s, ok := vs.Values[0].(*ast.SelectorExpr); ok {
						x := s.X.(*ast.Ident).Name
						sel := s.Sel.Name

						for _, imp := range c.Pkg.Imports() {
							if imp.Name() == x {
								tObj := imp.Scope().Lookup(sel)
								if cst, ok := tObj.(*types.Const); ok {
									// ..got the value check if this can be translated
									if minVersion, err := strconv.ParseInt(cst.Val().String(), 10, 64); err == nil {
										t.actualMinVersion = minVersion
									}
								}
							}
						}
					}
					if ival, ierr := gosec.GetInt(vs.Values[0]); ierr == nil {
						t.actualMinVersion = ival
					}
				}
			} else if ival, ierr := gosec.GetInt(n.Value); ierr == nil {
				t.actualMinVersion = ival
			} else {
				if se, ok := n.Value.(*ast.SelectorExpr); ok {
					if pkg, ok := se.X.(*ast.Ident); ok && pkg.Name == "tls" {
						t.actualMinVersion = t.mapVersion(se.Sel.Name)
					}
				}
			}

		case "MaxVersion":
			if ival, ierr := gosec.GetInt(n.Value); ierr == nil {
				t.actualMaxVersion = ival
			} else {
				if se, ok := n.Value.(*ast.SelectorExpr); ok {
					if pkg, ok := se.X.(*ast.Ident); ok && pkg.Name == "tls" {
						t.actualMaxVersion = t.mapVersion(se.Sel.Name)
					}
				}
			}

		case "CipherSuites":
			if ret := t.processTLSCipherSuites(n.Value, c); ret != nil {
				return ret
			}

		}
	}
	return nil
}

func (t *insecureConfigTLS) mapVersion(version string) int64 {
	var v int64
	switch version {
	case "VersionTLS13":
		v = tls.VersionTLS13
	case "VersionTLS12":
		v = tls.VersionTLS12
	case "VersionTLS11":
		v = tls.VersionTLS11
	case "VersionTLS10":
		v = tls.VersionTLS10
	}
	return v
}

func (t *insecureConfigTLS) checkVersion(n ast.Node, c *gosec.Context) *gosec.Issue {
	if t.actualMaxVersion == 0 && t.actualMinVersion >= t.MinVersion {
		// no warning is generated since the min version is greater than the secure min version
		return nil
	}
	if t.actualMinVersion < t.MinVersion {
		return gosec.NewIssue(c, n, t.ID(), "TLS MinVersion too low.", gosec.High, gosec.High)
	}
	if t.actualMaxVersion < t.MaxVersion {
		return gosec.NewIssue(c, n, t.ID(), "TLS MaxVersion too low.", gosec.High, gosec.High)
	}
	return nil
}

func (t *insecureConfigTLS) resetVersion() {
	t.actualMaxVersion = 0
	t.actualMinVersion = 0
}

func (t *insecureConfigTLS) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if complit, ok := n.(*ast.CompositeLit); ok && complit.Type != nil {
		actualType := c.Info.TypeOf(complit.Type)
		if actualType != nil && actualType.String() == t.requiredType {
			for _, elt := range complit.Elts {
				if kve, ok := elt.(*ast.KeyValueExpr); ok {
					issue := t.processTLSConfVal(kve, c)
					if issue != nil {
						return issue, nil
					}
				}
			}
			issue := t.checkVersion(complit, c)
			t.resetVersion()
			return issue, nil
		}
	}
	return nil, nil
}
