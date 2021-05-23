// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

/*

import (
	"net/url"
	"os"
	"path"
	"path/filepath"

	"github.com/go-openapi/jsonpointer"
)

  // Some currently unused functions and definitions that
  // used to be part of the expander.

  // Moved here for the record and possible future reuse

var (
	idPtr, _  = jsonpointer.New("/id")
	refPtr, _ = jsonpointer.New("/$ref")
)

func idFromNode(node interface{}) (*Ref, error) {
	if idValue, _, err := idPtr.Get(node); err == nil {
		if refStr, ok := idValue.(string); ok && refStr != "" {
			idRef, err := NewRef(refStr)
			if err != nil {
				return nil, err
			}
			return &idRef, nil
		}
	}
	return nil, nil
}

func nextRef(startingNode interface{}, startingRef *Ref, ptr *jsonpointer.Pointer) *Ref {
	if startingRef == nil {
		return nil
	}

	if ptr == nil {
		return startingRef
	}

	ret := startingRef
	var idRef *Ref
	node := startingNode

	for _, tok := range ptr.DecodedTokens() {
		node, _, _ = jsonpointer.GetForToken(node, tok)
		if node == nil {
			break
		}

		idRef, _ = idFromNode(node)
		if idRef != nil {
			nw, err := ret.Inherits(*idRef)
			if err != nil {
				break
			}
			ret = nw
		}

		refRef, _, _ := refPtr.Get(node)
		if refRef != nil {
			var rf Ref
			switch value := refRef.(type) {
			case string:
				rf, _ = NewRef(value)
			}
			nw, err := ret.Inherits(rf)
			if err != nil {
				break
			}
			nwURL := nw.GetURL()
			if nwURL.Scheme == "file" || (nwURL.Scheme == "" && nwURL.Host == "") {
				nwpt := filepath.ToSlash(nwURL.Path)
				if filepath.IsAbs(nwpt) {
					_, err := os.Stat(nwpt)
					if err != nil {
						nwURL.Path = filepath.Join(".", nwpt)
					}
				}
			}

			ret = nw
		}

	}

	return ret
}

// basePathFromSchemaID returns a new basePath based on an existing basePath and a schema ID
func basePathFromSchemaID(oldBasePath, id string) string {
	u, err := url.Parse(oldBasePath)
	if err != nil {
		panic(err)
	}
	uid, err := url.Parse(id)
	if err != nil {
		panic(err)
	}

	if path.IsAbs(uid.Path) {
		return id
	}
	u.Path = path.Join(path.Dir(u.Path), uid.Path)
	return u.String()
}
*/

// type ExtraSchemaProps map[string]interface{}

// // JSONSchema represents a structure that is a json schema draft 04
// type JSONSchema struct {
// 	SchemaProps
// 	ExtraSchemaProps
// }

// // MarshalJSON marshal this to JSON
// func (s JSONSchema) MarshalJSON() ([]byte, error) {
// 	b1, err := json.Marshal(s.SchemaProps)
// 	if err != nil {
// 		return nil, err
// 	}
// 	b2, err := s.Ref.MarshalJSON()
// 	if err != nil {
// 		return nil, err
// 	}
// 	b3, err := s.Schema.MarshalJSON()
// 	if err != nil {
// 		return nil, err
// 	}
// 	b4, err := json.Marshal(s.ExtraSchemaProps)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return swag.ConcatJSON(b1, b2, b3, b4), nil
// }

// // UnmarshalJSON marshal this from JSON
// func (s *JSONSchema) UnmarshalJSON(data []byte) error {
// 	var sch JSONSchema
// 	if err := json.Unmarshal(data, &sch.SchemaProps); err != nil {
// 		return err
// 	}
// 	if err := json.Unmarshal(data, &sch.Ref); err != nil {
// 		return err
// 	}
// 	if err := json.Unmarshal(data, &sch.Schema); err != nil {
// 		return err
// 	}
// 	if err := json.Unmarshal(data, &sch.ExtraSchemaProps); err != nil {
// 		return err
// 	}
// 	*s = sch
// 	return nil
// }
