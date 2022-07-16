// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author           xeipuuv
// author-github    https://github.com/xeipuuv
// author-mail      xeipuuv@gmail.com
//
// repository-name  gojsonschema
// repository-desc  An implementation of JSON Schema, based on IETF's draft v4 - Go language.
//
// description      Pool of referenced schemas.
//
// created          25-06-2013

package gojsonschema

import (
	"fmt"
)

type schemaReferencePool struct {
	documents map[string]*subSchema
}

func newSchemaReferencePool() *schemaReferencePool {

	p := &schemaReferencePool{}
	p.documents = make(map[string]*subSchema)

	return p
}

func (p *schemaReferencePool) Get(ref string) (r *subSchema, o bool) {

	if internalLogEnabled {
		internalLog(fmt.Sprintf("Schema Reference ( %s )", ref))
	}

	if sch, ok := p.documents[ref]; ok {
		if internalLogEnabled {
			internalLog(fmt.Sprintf(" From pool"))
		}
		return sch, true
	}

	return nil, false
}

func (p *schemaReferencePool) Add(ref string, sch *subSchema) {

	if internalLogEnabled {
		internalLog(fmt.Sprintf("Add Schema Reference %s to pool", ref))
	}
	if _, ok := p.documents[ref]; !ok {
		p.documents[ref] = sch
	}
}
