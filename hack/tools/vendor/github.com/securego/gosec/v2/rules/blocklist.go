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
	"strings"

	"github.com/securego/gosec/v2"
)

type blocklistedImport struct {
	gosec.MetaData
	Blocklisted map[string]string
}

func unquote(original string) string {
	copy := strings.TrimSpace(original)
	copy = strings.TrimLeft(copy, `"`)
	return strings.TrimRight(copy, `"`)
}

func (r *blocklistedImport) ID() string {
	return r.MetaData.ID
}

func (r *blocklistedImport) Match(n ast.Node, c *gosec.Context) (*gosec.Issue, error) {
	if node, ok := n.(*ast.ImportSpec); ok {
		if description, ok := r.Blocklisted[unquote(node.Path.Value)]; ok {
			return gosec.NewIssue(c, node, r.ID(), description, r.Severity, r.Confidence), nil
		}
	}
	return nil, nil
}

// NewBlocklistedImports reports when a blocklisted import is being used.
// Typically when a deprecated technology is being used.
func NewBlocklistedImports(id string, conf gosec.Config, blocklist map[string]string) (gosec.Rule, []ast.Node) {
	return &blocklistedImport{
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.Medium,
			Confidence: gosec.High,
		},
		Blocklisted: blocklist,
	}, []ast.Node{(*ast.ImportSpec)(nil)}
}

// NewBlocklistedImportMD5 fails if MD5 is imported
func NewBlocklistedImportMD5(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return NewBlocklistedImports(id, conf, map[string]string{
		"crypto/md5": "Blocklisted import crypto/md5: weak cryptographic primitive",
	})
}

// NewBlocklistedImportDES fails if DES is imported
func NewBlocklistedImportDES(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return NewBlocklistedImports(id, conf, map[string]string{
		"crypto/des": "Blocklisted import crypto/des: weak cryptographic primitive",
	})
}

// NewBlocklistedImportRC4 fails if DES is imported
func NewBlocklistedImportRC4(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return NewBlocklistedImports(id, conf, map[string]string{
		"crypto/rc4": "Blocklisted import crypto/rc4: weak cryptographic primitive",
	})
}

// NewBlocklistedImportCGI fails if CGI is imported
func NewBlocklistedImportCGI(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return NewBlocklistedImports(id, conf, map[string]string{
		"net/http/cgi": "Blocklisted import net/http/cgi: Go versions < 1.6.3 are vulnerable to Httpoxy attack: (CVE-2016-5386)",
	})
}

// NewBlocklistedImportSHA1 fails if SHA1 is imported
func NewBlocklistedImportSHA1(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	return NewBlocklistedImports(id, conf, map[string]string{
		"crypto/sha1": "Blocklisted import crypto/sha1: weak cryptographic primitive",
	})
}
