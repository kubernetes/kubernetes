package spec3

import (
	"math/rand"
	"strings"

	fuzz "github.com/google/gofuzz"

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
	func(s *string, c fuzz.Continue) {
		// All OpenAPI V3 map keys must follow the corresponding
		// regex. Note that this restricts the range for all other
		// string values as well.
		str := randAlphanumString()
		*s = str
	},
	func(o *OpenAPI, c fuzz.Continue) {
		c.FuzzNoCustom(o)
		o.Version = "3.0.0"
	},
	func(r *interface{}, c fuzz.Continue) {
		switch c.Intn(3) {
		case 0:
			*r = nil
		case 1:
			n := c.RandString() + "x"
			*r = n
		case 2:
			n := c.Float64()
			*r = n
		}
	},
	func(v **spec.Info, c fuzz.Continue) {
		// Info is never nil
		*v = &spec.Info{}
		c.FuzzNoCustom(*v)
		(*v).Title = c.RandString() + "x"
	},
	func(v *Paths, c fuzz.Continue) {
		c.Fuzz(&v.VendorExtensible)
		num := c.Intn(5)
		if num > 0 {
			v.Paths = make(map[string]*Path)
		}
		for i := 0; i < num; i++ {
			val := Path{}
			c.Fuzz(&val)
			v.Paths["/"+c.RandString()] = &val
		}
	},
	func(v *SecurityScheme, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Refable)
			return
		}
		switch c.Intn(4) {
		case 0:
			v.Type = "apiKey"
			v.Name = c.RandString() + "x"
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
			flow.AuthorizationUrl = c.RandString() + "x"
			v.Flows["implicit"] = &flow
			flow.Scopes = make(map[string]string)
			flow.Scopes["foo"] = "bar"
		case 3:
			v.Type = "openIdConnect"
			v.OpenIdConnectUrl = "https://" + c.RandString()
		}
		v.Scheme = "basic"
	},
	func(v *spec.Ref, c fuzz.Continue) {
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
	func(v *Parameter, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Refable)
			return
		}
		c.Fuzz(&v.ParameterProps)
		c.Fuzz(&v.VendorExtensible)

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
	func(v *RequestBody, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Refable)
			return
		}
		c.Fuzz(&v.RequestBodyProps)
		c.Fuzz(&v.VendorExtensible)
	},
	func(v *Header, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Refable)
			return
		}
		c.Fuzz(&v.HeaderProps)
		c.Fuzz(&v.VendorExtensible)
	},
	func(v *ResponsesProps, c fuzz.Continue) {
		c.Fuzz(&v.Default)
		n := c.Intn(5)
		for i := 0; i < n; i++ {
			r2 := Response{}
			c.Fuzz(&r2)
			// HTTP Status code in 100-599 Range
			code := c.Intn(500) + 100
			v.StatusCodeResponses = make(map[int]*Response)
			v.StatusCodeResponses[code] = &r2
		}
	},
	func(v *Response, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Refable)
			return
		}
		c.Fuzz(&v.ResponseProps)
		c.Fuzz(&v.VendorExtensible)
	},
	func(v *spec.Extensions, c fuzz.Continue) {
		numChildren := c.Intn(5)
		for i := 0; i < numChildren; i++ {
			if *v == nil {
				*v = spec.Extensions{}
			}
			(*v)["x-"+c.RandString()] = c.RandString()
		}
	},
	func(v *spec.ExternalDocumentation, c fuzz.Continue) {
		c.Fuzz(&v.Description)
		v.URL = "https://" + randAlphanumString()
	},
	func(v *spec.SchemaURL, c fuzz.Continue) {
		*v = spec.SchemaURL("https://" + randAlphanumString())
	},
	func(v *spec.SchemaOrBool, c fuzz.Continue) {
		*v = spec.SchemaOrBool{}

		if c.RandBool() {
			v.Allows = c.RandBool()
		} else {
			v.Schema = &spec.Schema{}
			v.Allows = true
			c.Fuzz(&v.Schema)
		}
	},
	func(v *spec.SchemaOrArray, c fuzz.Continue) {
		*v = spec.SchemaOrArray{}
		if c.RandBool() {
			schema := spec.Schema{}
			c.Fuzz(&schema)
			v.Schema = &schema
		} else {
			v.Schemas = []spec.Schema{}
			numChildren := c.Intn(5)
			for i := 0; i < numChildren; i++ {
				schema := spec.Schema{}
				c.Fuzz(&schema)
				v.Schemas = append(v.Schemas, schema)
			}

		}

	},
	func(v *spec.SchemaOrStringArray, c fuzz.Continue) {
		if c.RandBool() {
			*v = spec.SchemaOrStringArray{}
			if c.RandBool() {
				c.Fuzz(&v.Property)
			} else {
				c.Fuzz(&v.Schema)
			}
		}
	},
	func(v *spec.Schema, c fuzz.Continue) {
		if c.Intn(refChance) == 0 {
			c.Fuzz(&v.Ref)
			return
		}
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
			v.Type = spec.StringOrArray{"file"}

		} else {
			// normal schema
			c.Fuzz(&v.SchemaProps)
			c.Fuzz(&v.SwaggerSchemaProps)
			c.Fuzz(&v.VendorExtensible)
			c.Fuzz(&v.ExtraProps)
		}

	},
}
