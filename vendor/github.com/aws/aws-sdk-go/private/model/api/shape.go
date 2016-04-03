package api

import (
	"bytes"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
	"text/template"

	"github.com/aws/aws-sdk-go/private/util"
)

// A ShapeRef defines the usage of a shape within the API.
type ShapeRef struct {
	API           *API   `json:"-"`
	Shape         *Shape `json:"-"`
	Documentation string
	ShapeName     string `json:"shape"`
	Location      string
	LocationName  string
	QueryName     string
	Flattened     bool
	Streaming     bool
	XMLAttribute  bool
	XMLNamespace  XMLInfo
	Payload       string
}

// A XMLInfo defines URL and prefix for Shapes when rendered as XML
type XMLInfo struct {
	Prefix string
	URI    string
}

// A Shape defines the definition of a shape type
type Shape struct {
	API           *API `json:"-"`
	ShapeName     string
	Documentation string
	MemberRefs    map[string]*ShapeRef `json:"members"`
	MemberRef     ShapeRef             `json:"member"`
	KeyRef        ShapeRef             `json:"key"`
	ValueRef      ShapeRef             `json:"value"`
	Required      []string
	Payload       string
	Type          string
	Exception     bool
	Enum          []string
	EnumConsts    []string
	Flattened     bool
	Streaming     bool
	Location      string
	LocationName  string
	XMLNamespace  XMLInfo
	Min           int // optional Minimum length (string, list) or value (number)
	Max           int // optional Minimum length (string, list) or value (number)

	refs       []*ShapeRef // References to this shape
	resolvePkg string      // use this package in the goType() if present
}

// Rename changes the name of the Shape to newName. Also updates
// the associated API's reference to use newName.
func (s *Shape) Rename(newName string) {
	for _, r := range s.refs {
		r.ShapeName = newName
	}

	delete(s.API.Shapes, s.ShapeName)
	s.API.Shapes[newName] = s
	s.ShapeName = newName
}

// MemberNames returns a slice of struct member names.
func (s *Shape) MemberNames() []string {
	i, names := 0, make([]string, len(s.MemberRefs))
	for n := range s.MemberRefs {
		names[i] = n
		i++
	}
	sort.Strings(names)
	return names
}

// GoTypeWithPkgName returns a shape's type as a string with the package name in
// <packageName>.<type> format. Package naming only applies to structures.
func (s *Shape) GoTypeWithPkgName() string {
	return goType(s, true)
}

// GoType returns a shape's Go type
func (s *Shape) GoType() string {
	return goType(s, false)
}

// GoType returns a shape ref's Go type.
func (ref *ShapeRef) GoType() string {
	if ref.Shape == nil {
		panic(fmt.Errorf("missing shape definition on reference for %#v", ref))
	}

	return ref.Shape.GoType()
}

// GoTypeWithPkgName returns a shape's type as a string with the package name in
// <packageName>.<type> format. Package naming only applies to structures.
func (ref *ShapeRef) GoTypeWithPkgName() string {
	if ref.Shape == nil {
		panic(fmt.Errorf("missing shape definition on reference for %#v", ref))
	}

	return ref.Shape.GoTypeWithPkgName()
}

// Returns a string version of the Shape's type.
// If withPkgName is true, the package name will be added as a prefix
func goType(s *Shape, withPkgName bool) string {
	switch s.Type {
	case "structure":
		if withPkgName || s.resolvePkg != "" {
			pkg := s.resolvePkg
			if pkg != "" {
				s.API.imports[pkg] = true
				pkg = path.Base(pkg)
			} else {
				pkg = s.API.PackageName()
			}
			return fmt.Sprintf("*%s.%s", pkg, s.ShapeName)
		}
		return "*" + s.ShapeName
	case "map":
		return "map[string]" + s.ValueRef.GoType()
	case "list":
		return "[]" + s.MemberRef.GoType()
	case "boolean":
		return "*bool"
	case "string", "character":
		return "*string"
	case "blob":
		return "[]byte"
	case "integer", "long":
		return "*int64"
	case "float", "double":
		return "*float64"
	case "timestamp":
		s.API.imports["time"] = true
		return "*time.Time"
	default:
		panic("Unsupported shape type: " + s.Type)
	}
}

// GoTypeElem returns the Go type for the Shape. If the shape type is a pointer just
// the type will be returned minus the pointer *.
func (s *Shape) GoTypeElem() string {
	t := s.GoType()
	if strings.HasPrefix(t, "*") {
		return t[1:]
	}
	return t
}

// GoTypeElem returns the Go type for the Shape. If the shape type is a pointer just
// the type will be returned minus the pointer *.
func (ref *ShapeRef) GoTypeElem() string {
	if ref.Shape == nil {
		panic(fmt.Errorf("missing shape definition on reference for %#v", ref))
	}

	return ref.Shape.GoTypeElem()
}

// GoTags returns the rendered tags string for the ShapeRef
func (ref *ShapeRef) GoTags(toplevel bool, isRequired bool) string {
	code := "`"
	if ref.Location != "" {
		code += `location:"` + ref.Location + `" `
	} else if ref.Shape.Location != "" {
		code += `location:"` + ref.Shape.Location + `" `
	}
	if ref.LocationName != "" {
		code += `locationName:"` + ref.LocationName + `" `
	} else if ref.Shape.LocationName != "" {
		code += `locationName:"` + ref.Shape.LocationName + `" `
	}
	if ref.QueryName != "" {
		code += `queryName:"` + ref.QueryName + `" `
	}
	if ref.Shape.MemberRef.LocationName != "" {
		code += `locationNameList:"` + ref.Shape.MemberRef.LocationName + `" `
	}
	if ref.Shape.KeyRef.LocationName != "" {
		code += `locationNameKey:"` + ref.Shape.KeyRef.LocationName + `" `
	}
	if ref.Shape.ValueRef.LocationName != "" {
		code += `locationNameValue:"` + ref.Shape.ValueRef.LocationName + `" `
	}
	if ref.Shape.Min > 0 {
		code += fmt.Sprintf(`min:"%d" `, ref.Shape.Min)
	}
	code += `type:"` + ref.Shape.Type + `" `

	// embed the timestamp type for easier lookups
	if ref.Shape.Type == "timestamp" {
		code += `timestampFormat:"`
		if ref.Location == "header" {
			code += "rfc822"
		} else {
			switch ref.API.Metadata.Protocol {
			case "json", "rest-json":
				code += "unix"
			case "rest-xml", "ec2", "query":
				code += "iso8601"
			}
		}
		code += `" `
	}

	if ref.Shape.Flattened || ref.Flattened {
		code += `flattened:"true" `
	}

	if ref.XMLAttribute {
		code += `xmlAttribute:"true" `
	}

	if isRequired {
		code += `required:"true" `
	}

	if ref.Shape.IsEnum() {
		code += `enum:"` + ref.ShapeName + `" `
	}

	if toplevel {
		if ref.Shape.Payload != "" {
			code += `payload:"` + ref.Shape.Payload + `" `
		}
		if ref.XMLNamespace.Prefix != "" {
			code += `xmlPrefix:"` + ref.XMLNamespace.Prefix + `" `
		} else if ref.Shape.XMLNamespace.Prefix != "" {
			code += `xmlPrefix:"` + ref.Shape.XMLNamespace.Prefix + `" `
		}
		if ref.XMLNamespace.URI != "" {
			code += `xmlURI:"` + ref.XMLNamespace.URI + `" `
		} else if ref.Shape.XMLNamespace.URI != "" {
			code += `xmlURI:"` + ref.Shape.XMLNamespace.URI + `" `
		}

	}

	return strings.TrimSpace(code) + "`"
}

// Docstring returns the godocs formated documentation
func (ref *ShapeRef) Docstring() string {
	if ref.Documentation != "" {
		return ref.Documentation
	}
	return ref.Shape.Docstring()
}

// Docstring returns the godocs formated documentation
func (s *Shape) Docstring() string {
	return s.Documentation
}

var goCodeStringerTmpl = template.Must(template.New("goCodeStringerTmpl").Parse(`
// String returns the string representation
func (s {{ .ShapeName }}) String() string {
	return awsutil.Prettify(s)
}
// GoString returns the string representation
func (s {{ .ShapeName }}) GoString() string {
	return s.String()
}
`))

func (s *Shape) goCodeStringers() string {
	w := bytes.Buffer{}
	if err := goCodeStringerTmpl.Execute(&w, s); err != nil {
		panic(fmt.Sprintln("Unexpected error executing goCodeStringers template", err))
	}

	return w.String()
}

var enumStrip = regexp.MustCompile(`[^a-zA-Z0-9_:\./-]`)
var enumDelims = regexp.MustCompile(`[-_:\./]+`)
var enumCamelCase = regexp.MustCompile(`([a-z])([A-Z])`)

// EnumName returns the Nth enum in the shapes Enum list
func (s *Shape) EnumName(n int) string {
	enum := s.Enum[n]
	enum = enumStrip.ReplaceAllLiteralString(enum, "")
	enum = enumCamelCase.ReplaceAllString(enum, "$1-$2")
	parts := enumDelims.Split(enum, -1)
	for i, v := range parts {
		v = strings.ToLower(v)
		parts[i] = ""
		if len(v) > 0 {
			parts[i] = strings.ToUpper(v[0:1])
		}
		if len(v) > 1 {
			parts[i] += v[1:]
		}
	}
	enum = strings.Join(parts, "")
	enum = strings.ToUpper(enum[0:1]) + enum[1:]
	return enum
}

// GoCode returns the rendered Go code for the Shape.
func (s *Shape) GoCode() string {
	code := s.Docstring()
	if !s.IsEnum() {
		code += "type " + s.ShapeName + " "
	}

	switch {
	case s.Type == "structure":
		ref := &ShapeRef{ShapeName: s.ShapeName, API: s.API, Shape: s}

		code += "struct {\n"
		code += "_ struct{} " + ref.GoTags(true, false) + "\n\n"
		for _, n := range s.MemberNames() {
			m := s.MemberRefs[n]
			code += m.Docstring()
			if (m.Streaming || m.Shape.Streaming) && s.Payload == n {
				rtype := "io.ReadSeeker"
				if len(s.refs) > 1 {
					rtype = "aws.ReaderSeekCloser"
				} else if strings.HasSuffix(s.ShapeName, "Output") {
					rtype = "io.ReadCloser"
				}

				s.API.imports["io"] = true
				code += n + " " + rtype + " " + m.GoTags(false, s.IsRequired(n)) + "\n\n"
			} else {
				code += n + " " + m.GoType() + " " + m.GoTags(false, s.IsRequired(n)) + "\n\n"
			}
		}
		code += "}"

		if !s.API.NoStringerMethods {
			code += s.goCodeStringers()
		}
	case s.IsEnum():
		code += "const (\n"
		for n, e := range s.Enum {
			code += fmt.Sprintf("\t// @enum %s\n\t%s = %q\n",
				s.ShapeName, s.EnumConsts[n], e)
		}
		code += ")"
	default:
		panic("Cannot generate toplevel shape for " + s.Type)
	}

	return util.GoFmt(code)
}

// IsEnum returns whether this shape is an enum list
func (s *Shape) IsEnum() bool {
	return s.Type == "string" && len(s.Enum) > 0
}

// IsRequired returns if member is a required field.
func (s *Shape) IsRequired(member string) bool {
	for _, n := range s.Required {
		if n == member {
			return true
		}
	}
	return false
}

// IsInternal returns whether the shape was defined in this package
func (s *Shape) IsInternal() bool {
	return s.resolvePkg == ""
}

// removeRef removes a shape reference from the list of references this
// shape is used in.
func (s *Shape) removeRef(ref *ShapeRef) {
	r := s.refs
	for i := 0; i < len(r); i++ {
		if r[i] == ref {
			j := i + 1
			copy(r[i:], r[j:])
			for k, n := len(r)-j+i, len(r); k < n; k++ {
				r[k] = nil // free up the end of the list
			} // for k
			s.refs = r[:len(r)-j+i]
			break
		}
	}
}
