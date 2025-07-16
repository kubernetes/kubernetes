package spec3

import (
	"math/rand"
	"strings"

	"sigs.k8s.io/randfill"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

// refChance is the chance that a particular component will use a $ref
// instead of fuzzed. Expressed as a fraction 1/n, currently there is
// a 1/3 chance that a ref will be used.
const refChance = 3

const alphaNumChars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randAlphanumString() string {
	arr := make([]string, rand.Intn(10)+5)
	for i := 0; i < len(arr); i++ {
		arr[i] = string(alphaNumChars[rand.Intn(len(alphaNumChars))])
	}
	return strings.Join(arr, "")
}

var OpenAPIV3FuzzFuncs []interface{} = []interface{}{
	func(s *string, c randfill.Continue) {
		// All OpenAPI V3 map keys must follow the corresponding
		// regex. Note that this restricts the range for all other
		// string values as well.
		str := randAlphanumString()
		*s = str
	},
	func(o *OpenAPI, c randfill.Continue) {
		c.FillNoCustom(o)
		o.Version = "3.0.0"
		for i, val := range o.SecurityRequirement {
			if val == nil {
				o.SecurityRequirement[i] = make(map[string][]string)
			}

			for k, v := range val {
				if v == nil {
					val[k] = make([]string, 0)
				}
			}
		}

	},
	func(r *interface{}, c randfill.Continue) {
		switch c.Intn(3) {
		case 0:
			*r = nil
		case 1:
			n := c.String(0) + "x"
			*r = n
		case 2:
			n := c.Float64()
			*r = n
		}
	},
	func(v **spec.Info, c randfill.Continue) {
		// Info is never nil
		*v = &spec.Info{}
		c.FillNoCustom(*v)
		(*v).Title = c.String(0) + "x"
	},
	func(v *Paths, c randfill.Continue) {
		c.Fill(&v.VendorExtensible)
		num := c.Intn(5)
		if num > 0 {
			v.Paths = make(map[string]*Path)
		}
		for i := 0; i < num; i++ {
			val := Path{}
			c.Fill(&val)
			v.Paths["/"+c.String(0)] = &val
		}
	},
	func(v *SecurityScheme, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Refable)
			return
		}
		switch c.Intn(4) {
		case 0:
			v.Type = "apiKey"
			v.Name = c.String(0) + "x"
			switch c.Intn(3) {
			case 0:
				v.In = "query"
			case 1:
				v.In = "header"
			case 2:
				v.In = "cookie"
			}
		case 1:
			v.Type = "http"
		case 2:
			v.Type = "oauth2"
			v.Flows = make(map[string]*OAuthFlow)
			flow := OAuthFlow{}
			flow.AuthorizationUrl = c.String(0) + "x"
			v.Flows["implicit"] = &flow
			flow.Scopes = make(map[string]string)
			flow.Scopes["foo"] = "bar"
		case 3:
			v.Type = "openIdConnect"
			v.OpenIdConnectUrl = "https://" + c.String(0)
		}
		v.Scheme = "basic"
	},
	func(v *spec.Ref, c randfill.Continue) {
		switch c.Intn(7) {
		case 0:
			*v = spec.MustCreateRef("#/components/schemas/" + randAlphanumString())
		case 1:
			*v = spec.MustCreateRef("#/components/responses/" + randAlphanumString())
		case 2:
			*v = spec.MustCreateRef("#/components/headers/" + randAlphanumString())
		case 3:
			*v = spec.MustCreateRef("#/components/securitySchemes/" + randAlphanumString())
		case 5:
			*v = spec.MustCreateRef("#/components/parameters/" + randAlphanumString())
		case 6:
			*v = spec.MustCreateRef("#/components/requestBodies/" + randAlphanumString())
		}
	},
	func(v *Parameter, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Refable)
			return
		}
		c.Fill(&v.ParameterProps)
		c.Fill(&v.VendorExtensible)

		switch c.Intn(3) {
		case 0:
			// Header param
			v.In = "query"
		case 1:
			v.In = "header"
		case 2:
			v.In = "cookie"
		}
	},
	func(v *RequestBody, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Refable)
			return
		}
		c.Fill(&v.RequestBodyProps)
		c.Fill(&v.VendorExtensible)
	},
	func(v *Header, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Refable)
			return
		}
		c.Fill(&v.HeaderProps)
		c.Fill(&v.VendorExtensible)
	},
	func(v *ResponsesProps, c randfill.Continue) {
		c.Fill(&v.Default)
		n := c.Intn(5)
		for i := 0; i < n; i++ {
			r2 := Response{}
			c.Fill(&r2)
			// HTTP Status code in 100-599 Range
			code := c.Intn(500) + 100
			v.StatusCodeResponses = make(map[int]*Response)
			v.StatusCodeResponses[code] = &r2
		}
	},
	func(v *Response, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Refable)
			return
		}
		c.Fill(&v.ResponseProps)
		c.Fill(&v.VendorExtensible)
	},
	func(v *Operation, c randfill.Continue) {
		c.FillNoCustom(v)
		// Do not fuzz null values into the array.
		for i, val := range v.SecurityRequirement {
			if val == nil {
				v.SecurityRequirement[i] = make(map[string][]string)
			}

			for k, v := range val {
				if v == nil {
					val[k] = make([]string, 0)
				}
			}
		}
	},
	func(v *spec.Extensions, c randfill.Continue) {
		numChildren := c.Intn(5)
		for i := 0; i < numChildren; i++ {
			if *v == nil {
				*v = spec.Extensions{}
			}
			(*v)["x-"+c.String(0)] = c.String(0)
		}
	},
	func(v *spec.ExternalDocumentation, c randfill.Continue) {
		c.Fill(&v.Description)
		v.URL = "https://" + randAlphanumString()
	},
	func(v *spec.SchemaURL, c randfill.Continue) {
		*v = spec.SchemaURL("https://" + randAlphanumString())
	},
	func(v *spec.SchemaOrBool, c randfill.Continue) {
		*v = spec.SchemaOrBool{}

		if c.Bool() {
			v.Allows = c.Bool()
		} else {
			v.Schema = &spec.Schema{}
			v.Allows = true
			c.Fill(&v.Schema)
		}
	},
	func(v *spec.SchemaOrArray, c randfill.Continue) {
		*v = spec.SchemaOrArray{}
		if c.Bool() {
			schema := spec.Schema{}
			c.Fill(&schema)
			v.Schema = &schema
		} else {
			v.Schemas = []spec.Schema{}
			numChildren := c.Intn(5)
			for i := 0; i < numChildren; i++ {
				schema := spec.Schema{}
				c.Fill(&schema)
				v.Schemas = append(v.Schemas, schema)
			}

		}

	},
	func(v *spec.SchemaOrStringArray, c randfill.Continue) {
		if c.Bool() {
			*v = spec.SchemaOrStringArray{}
			if c.Bool() {
				c.Fill(&v.Property)
			} else {
				c.Fill(&v.Schema)
			}
		}
	},
	func(v *spec.Schema, c randfill.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fill(&v.Ref)
			return
		}
		if c.Bool() {
			// file schema
			c.Fill(&v.Default)
			c.Fill(&v.Description)
			c.Fill(&v.Example)
			c.Fill(&v.ExternalDocs)

			c.Fill(&v.Format)
			c.Fill(&v.ReadOnly)
			c.Fill(&v.Required)
			c.Fill(&v.Title)
			v.Type = spec.StringOrArray{"file"}

		} else {
			// normal schema
			c.Fill(&v.SchemaProps)
			c.Fill(&v.SwaggerSchemaProps)
			c.Fill(&v.VendorExtensible)
			c.Fill(&v.ExtraProps)
		}

	},
}
