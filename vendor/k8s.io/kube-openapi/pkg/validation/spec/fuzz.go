/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package spec

import (
	"github.com/go-openapi/jsonreference"
	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
)

var SwaggerFuzzFuncs []interface{} = []interface{}{
	func(v *Responses, c fuzz.Continue) {
		c.FuzzNoCustom(v)
		if v.Default != nil {
			// Check if we hit maxDepth and left an incomplete value
			if v.Default.Description == "" {
				v.Default = nil
				v.StatusCodeResponses = nil
			}
		}

		// conversion has no way to discern empty statusCodeResponses from
		// nil, since "default" is always included in the map.
		// So avoid empty responses list
		if len(v.StatusCodeResponses) == 0 {
			v.StatusCodeResponses = nil
		}
	},
	func(v *Operation, c fuzz.Continue) {
		c.FuzzNoCustom(v)

		if v != nil {
			// force non-nil
			v.Responses = &Responses{}
			c.Fuzz(v.Responses)

			v.Schemes = nil
			if c.RandBool() {
				v.Schemes = append(v.Schemes, "http")
			}

			if c.RandBool() {
				v.Schemes = append(v.Schemes, "https")
			}

			if c.RandBool() {
				v.Schemes = append(v.Schemes, "ws")
			}

			if c.RandBool() {
				v.Schemes = append(v.Schemes, "wss")
			}

			// Gnostic unconditionally makes security values non-null
			// So do not fuzz null values into the array.
			for i, val := range v.Security {
				if val == nil {
					v.Security[i] = make(map[string][]string)
				}

				for k, v := range val {
					if v == nil {
						val[k] = make([]string, 0)
					}
				}
			}
		}
	},
	func(v map[int]Response, c fuzz.Continue) {
		n := 0
		c.Fuzz(&n)
		if n == 0 {
			// Test that fuzzer is not at maxDepth so we do not
			// end up with empty elements
			return
		}

		// Prevent negative numbers
		num := c.Intn(4)
		for i := 0; i < num+2; i++ {
			val := Response{}
			c.Fuzz(&val)

			val.Description = c.RandString() + "x"
			v[100*(i+1)+c.Intn(100)] = val
		}
	},
	func(v map[string]PathItem, c fuzz.Continue) {
		n := 0
		c.Fuzz(&n)
		if n == 0 {
			// Test that fuzzer is not at maxDepth so we do not
			// end up with empty elements
			return
		}

		num := c.Intn(5)
		for i := 0; i < num+2; i++ {
			val := PathItem{}
			c.Fuzz(&val)

			// Ref params are only allowed in certain locations, so
			// possibly add a few to PathItems
			numRefsToAdd := c.Intn(5)
			for i := 0; i < numRefsToAdd; i++ {
				theRef := Parameter{}
				c.Fuzz(&theRef.Refable)

				val.Parameters = append(val.Parameters, theRef)
			}

			v["/"+c.RandString()] = val
		}
	},
	func(v *SchemaOrArray, c fuzz.Continue) {
		*v = SchemaOrArray{}
		// gnostic parser just doesn't support more
		// than one Schema here
		v.Schema = &Schema{}
		c.Fuzz(&v.Schema)

	},
	func(v *SchemaOrBool, c fuzz.Continue) {
		*v = SchemaOrBool{}

		if c.RandBool() {
			v.Allows = c.RandBool()
		} else {
			v.Schema = &Schema{}
			v.Allows = true
			c.Fuzz(&v.Schema)
		}
	},
	func(v map[string]Response, c fuzz.Continue) {
		n := 0
		c.Fuzz(&n)
		if n == 0 {
			// Test that fuzzer is not at maxDepth so we do not
			// end up with empty elements
			return
		}

		// Response definitions are not allowed to
		// be refs
		for i := 0; i < c.Intn(5)+1; i++ {
			resp := &Response{}

			c.Fuzz(resp)
			resp.Ref = Ref{}
			resp.Description = c.RandString() + "x"

			// Response refs are not vendor extensible by gnostic
			resp.VendorExtensible.Extensions = nil
			v[c.RandString()+"x"] = *resp
		}
	},
	func(v *Header, c fuzz.Continue) {
		if v != nil {
			c.FuzzNoCustom(v)

			// descendant Items of Header may not be refs
			cur := v.Items
			for cur != nil {
				cur.Ref = Ref{}
				cur = cur.Items
			}
		}
	},
	func(v *Ref, c fuzz.Continue) {
		*v = Ref{}
		v.Ref, _ = jsonreference.New("http://asd.com/" + c.RandString())
	},
	func(v *Response, c fuzz.Continue) {
		*v = Response{}
		if c.RandBool() {
			v.Ref = Ref{}
			v.Ref.Ref, _ = jsonreference.New("http://asd.com/" + c.RandString())
		} else {
			c.Fuzz(&v.VendorExtensible)
			c.Fuzz(&v.Schema)
			c.Fuzz(&v.ResponseProps)

			v.Headers = nil
			v.Ref = Ref{}

			n := 0
			c.Fuzz(&n)
			if n != 0 {
				// Test that fuzzer is not at maxDepth so we do not
				// end up with empty elements
				num := c.Intn(4)
				for i := 0; i < num; i++ {
					if v.Headers == nil {
						v.Headers = make(map[string]Header)
					}
					hdr := Header{}
					c.Fuzz(&hdr)
					if hdr.Type == "" {
						// hit maxDepth, just abort trying to make haders
						v.Headers = nil
						break
					}
					v.Headers[c.RandString()+"x"] = hdr
				}
			} else {
				v.Headers = nil
			}
		}

		v.Description = c.RandString() + "x"

		// Gnostic parses empty as nil, so to keep avoid putting empty
		if len(v.Headers) == 0 {
			v.Headers = nil
		}
	},
	func(v **Info, c fuzz.Continue) {
		// Info is never nil
		*v = &Info{}
		c.FuzzNoCustom(*v)

		(*v).Title = c.RandString() + "x"
	},
	func(v *Extensions, c fuzz.Continue) {
		// gnostic parser only picks up x- vendor extensions
		numChildren := c.Intn(5)
		for i := 0; i < numChildren; i++ {
			if *v == nil {
				*v = Extensions{}
			}
			(*v)["x-"+c.RandString()] = c.RandString()
		}
	},
	func(v *Swagger, c fuzz.Continue) {
		c.FuzzNoCustom(v)

		if v.Paths == nil {
			// Force paths non-nil since it does not have omitempty in json tag.
			// This means a perfect roundtrip (via json) is impossible,
			// since we can't tell the difference between empty/unspecified paths
			v.Paths = &Paths{}
			c.Fuzz(v.Paths)
		}

		v.Swagger = "2.0"

		// Gnostic support serializing ID at all
		// unavoidable data loss
		v.ID = ""

		v.Schemes = nil
		if c.RandUint64()%2 == 1 {
			v.Schemes = append(v.Schemes, "http")
		}

		if c.RandUint64()%2 == 1 {
			v.Schemes = append(v.Schemes, "https")
		}

		if c.RandUint64()%2 == 1 {
			v.Schemes = append(v.Schemes, "ws")
		}

		if c.RandUint64()%2 == 1 {
			v.Schemes = append(v.Schemes, "wss")
		}

		// Gnostic unconditionally makes security values non-null
		// So do not fuzz null values into the array.
		for i, val := range v.Security {
			if val == nil {
				v.Security[i] = make(map[string][]string)
			}

			for k, v := range val {
				if v == nil {
					val[k] = make([]string, 0)
				}
			}
		}
	},
	func(v *SecurityScheme, c fuzz.Continue) {
		v.Description = c.RandString() + "x"
		c.Fuzz(&v.VendorExtensible)

		switch c.Intn(3) {
		case 0:
			v.Type = "basic"
		case 1:
			v.Type = "apiKey"
			switch c.Intn(2) {
			case 0:
				v.In = "header"
			case 1:
				v.In = "query"
			default:
				panic("unreachable")
			}
			v.Name = "x" + c.RandString()
		case 2:
			v.Type = "oauth2"

			switch c.Intn(4) {
			case 0:
				v.Flow = "accessCode"
				v.TokenURL = "https://" + c.RandString()
				v.AuthorizationURL = "https://" + c.RandString()
			case 1:
				v.Flow = "application"
				v.TokenURL = "https://" + c.RandString()
			case 2:
				v.Flow = "implicit"
				v.AuthorizationURL = "https://" + c.RandString()
			case 3:
				v.Flow = "password"
				v.TokenURL = "https://" + c.RandString()
			default:
				panic("unreachable")
			}
			c.Fuzz(&v.Scopes)
		default:
			panic("unreachable")
		}
	},
	func(v *interface{}, c fuzz.Continue) {
		*v = c.RandString() + "x"
	},
	func(v *string, c fuzz.Continue) {
		*v = c.RandString() + "x"
	},
	func(v *ExternalDocumentation, c fuzz.Continue) {
		v.Description = c.RandString() + "x"
		v.URL = c.RandString() + "x"
	},
	func(v *SimpleSchema, c fuzz.Continue) {
		c.FuzzNoCustom(v)

		switch c.Intn(5) {
		case 0:
			v.Type = "string"
		case 1:
			v.Type = "number"
		case 2:
			v.Type = "boolean"
		case 3:
			v.Type = "integer"
		case 4:
			v.Type = "array"
		default:
			panic("unreachable")
		}

		switch c.Intn(5) {
		case 0:
			v.CollectionFormat = "csv"
		case 1:
			v.CollectionFormat = "ssv"
		case 2:
			v.CollectionFormat = "tsv"
		case 3:
			v.CollectionFormat = "pipes"
		case 4:
			v.CollectionFormat = ""
		default:
			panic("unreachable")
		}

		// None of the types which include SimpleSchema in our definitions
		// actually support "example" in the official spec
		v.Example = nil

		// unsupported by openapi
		v.Nullable = false
	},
	func(v *int64, c fuzz.Continue) {
		c.Fuzz(v)

		// Gnostic does not differentiate between 0 and non-specified
		// so avoid using 0 for fuzzer
		if *v == 0 {
			*v = 1
		}
	},
	func(v *float64, c fuzz.Continue) {
		c.Fuzz(v)

		// Gnostic does not differentiate between 0 and non-specified
		// so avoid using 0 for fuzzer
		if *v == 0.0 {
			*v = 1.0
		}
	},
	func(v *Parameter, c fuzz.Continue) {
		if v == nil {
			return
		}
		c.Fuzz(&v.VendorExtensible)
		if c.RandBool() {
			// body param
			v.Description = c.RandString() + "x"
			v.Name = c.RandString() + "x"
			v.In = "body"
			c.Fuzz(&v.Description)
			c.Fuzz(&v.Required)

			v.Schema = &Schema{}
			c.Fuzz(&v.Schema)

		} else {
			c.Fuzz(&v.SimpleSchema)
			c.Fuzz(&v.CommonValidations)
			v.AllowEmptyValue = false
			v.Description = c.RandString() + "x"
			v.Name = c.RandString() + "x"

			switch c.Intn(4) {
			case 0:
				// Header param
				v.In = "header"
			case 1:
				// Form data param
				v.In = "formData"
				v.AllowEmptyValue = c.RandBool()
			case 2:
				// Query param
				v.In = "query"
				v.AllowEmptyValue = c.RandBool()
			case 3:
				// Path param
				v.In = "path"
				v.Required = true
			default:
				panic("unreachable")
			}

			// descendant Items of Parameter may not be refs
			cur := v.Items
			for cur != nil {
				cur.Ref = Ref{}
				cur = cur.Items
			}
		}
	},
	func(v *Schema, c fuzz.Continue) {
		if c.RandBool() {
			// file schema
			c.Fuzz(&v.Default)
			c.Fuzz(&v.Description)
			c.Fuzz(&v.Example)
			c.Fuzz(&v.ExternalDocs)

			c.Fuzz(&v.Format)
			c.Fuzz(&v.ReadOnly)
			c.Fuzz(&v.Required)
			c.Fuzz(&v.Title)
			v.Type = StringOrArray{"file"}

		} else {
			// normal schema
			c.Fuzz(&v.SchemaProps)
			c.Fuzz(&v.SwaggerSchemaProps)
			c.Fuzz(&v.VendorExtensible)
			// c.Fuzz(&v.ExtraProps)
			// ExtraProps will not roundtrip - gnostic throws out
			// unrecognized keys
		}

		// Not supported by official openapi v2 spec
		// and stripped by k8s apiserver
		v.ID = ""
		v.AnyOf = nil
		v.OneOf = nil
		v.Not = nil
		v.Nullable = false
		v.AdditionalItems = nil
		v.Schema = ""
		v.PatternProperties = nil
		v.Definitions = nil
		v.Dependencies = nil
	},
}

var SwaggerDiffOptions = []cmp.Option{
	// cmp.Diff panics on Ref since jsonreference.Ref uses unexported fields
	cmp.Comparer(func(a Ref, b Ref) bool {
		return a.String() == b.String()
	}),
}
