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

package rules

import (
	"go/ast"
	"go/types"

	"github.com/securego/gosec/v2"
)

type subprocess struct {
	gosec.MetaData
	gosec.CallList
}

func (r *subprocess) ID() string {
	return r.MetaData.ID
}

// TODO(gm) The only real potential for command injection with a Go project
// is something like this:
//
// syscall.Exec("/bin/sh", []string{"-c", tainted})
//
// E.g. Input is correctly escaped but the execution context being used
// is unsafe. For example:
//
// syscall.Exec("echo", "foobar" + tainted)
func (r *subprocess) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if node := r.ContainsPkgCallExpr(n, c, false); node != nil {
		args := node.Args
		if r.isContext(n, c) {
			args = args[1:]
		}
		for _, arg := range args {
			if ident, ok := arg.(*ast.Ident); ok {
				obj := c.Info.ObjectOf(ident)

				// need to cast and check whether it is for a variable ?
				_, variable := obj.(*types.Var)

				// .. indeed it is a variable then processing is different than a normal
				// field assignment
				if variable {
					// skip the check when the declaration is not available
					if ident.Obj == nil {
						continue
					}
					switch ident.Obj.Decl.(type) {
					case *ast.AssignStmt:
						_, assignment := ident.Obj.Decl.(*ast.AssignStmt)
						if variable && assignment {
							if !gosec.TryResolve(ident, c) {
								return gosec.NewIssue(c, n, r.ID(), "Subprocess launched with variable", gosec.Medium, gosec.High), nil
							}
						}
					case *ast.Field:
						_, field := ident.Obj.Decl.(*ast.Field)
						if variable && field {
							// check if the variable exist in the scope
							vv, vvok := obj.(*types.Var)

							if vvok && vv.Parent().Lookup(ident.Name) == nil {
								return gosec.NewIssue(c, n, r.ID(), "Subprocess launched with variable", gosec.Medium, gosec.High), nil
							}
						}
					}
				}
			} else if !gosec.TryResolve(arg, c) {
				// the arg is not a constant or a variable but instead a function call or os.Args[i]
				return gosec.NewIssue(c, n, r.ID(), "Subprocess launched with a potential tainted input or cmd arguments", gosec.Medium, gosec.High), nil
			}
		}
	}
	return nil, nil
}

// isContext checks whether or not the node is a CommandContext call or not
// Thi is required in order to skip the first argument from the check.
func (r *subprocess) isContext(n ast.Node, ctx *gosec.Context) bool {
	selector, indent, err := gosec.GetCallInfo(n, ctx)
	if err != nil {
		return false
	}
	if selector == "exec" && indent == "CommandContext" {
		return true
	}
	return false
}

// NewSubproc detects cases where we are forking out to an external process
func NewSubproc(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	rule := &subprocess{gosec.MetaData{ID: id}, gosec.NewCallList()}
	rule.Add("os/exec", "Command")
	rule.Add("os/exec", "CommandContext")
	rule.Add("syscall", "Exec")
	rule.Add("syscall", "ForkExec")
	rule.Add("syscall", "StartProcess")
	rule.Add("golang.org/x/sys/execabs", "Command")
	rule.Add("golang.org/x/sys/execabs", "CommandContext")
	return rule, []ast.Node{(*ast.CallExpr)(nil)}
}
