// +build codegen

package api

import (
	"bytes"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
	"text/template"

	"github.com/aws/aws-sdk-go/private/protocol"
)

// ErrorInfo represents the error block of a shape's structure
type ErrorInfo struct {
	Type           string
	Code           string
	HTTPStatusCode int
}

// A XMLInfo defines URL and prefix for Shapes when rendered as XML
type XMLInfo struct {
	Prefix string
	URI    string
}

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
	// Ignore, if set, will not be sent over the wire
	Ignore              bool
	XMLNamespace        XMLInfo
	Payload             string
	IdempotencyToken    bool   `json:"idempotencyToken"`
	TimestampFormat     string `json:"timestampFormat"`
	JSONValue           bool   `json:"jsonvalue"`
	Deprecated          bool   `json:"deprecated"`
	DeprecatedMsg       string `json:"deprecatedMessage"`
	EndpointDiscoveryID bool   `json:"endpointdiscoveryid"`
	HostLabel           bool   `json:"hostLabel"`

	OrigShapeName string `json:"-"`

	GenerateGetter bool

	IsEventPayload bool `json:"eventpayload"`
	IsEventHeader  bool `json:"eventheader"`

	// Collection of custom tags the shape reference includes.
	CustomTags ShapeTags

	// Flags whether the member reference is a endpoint ARN
	EndpointARN bool

	// Flags whether the member reference is a Outpost ID
	OutpostIDMember bool

	// Flag whether the member reference is a Account ID when endpoint shape ARN is present
	AccountIDMemberWithARN bool
}

// A Shape defines the definition of a shape type
type Shape struct {
	API              *API `json:"-"`
	ShapeName        string
	Documentation    string
	MemberRefs       map[string]*ShapeRef `json:"members"`
	MemberRef        ShapeRef             `json:"member"` // List ref
	KeyRef           ShapeRef             `json:"key"`    // map key ref
	ValueRef         ShapeRef             `json:"value"`  // map value ref
	Required         []string
	Payload          string
	Type             string
	Exception        bool
	Enum             []string
	EnumConsts       []string
	Flattened        bool
	Streaming        bool
	Location         string
	LocationName     string
	IdempotencyToken bool   `json:"idempotencyToken"`
	TimestampFormat  string `json:"timestampFormat"`
	XMLNamespace     XMLInfo
	Min              float64 // optional Minimum length (string, list) or value (number)

	OutputEventStreamAPI *EventStreamAPI
	EventStream          *EventStream
	EventFor             []*EventStream `json:"-"`

	IsInputEventStream  bool `json:"-"`
	IsOutputEventStream bool `json:"-"`

	IsEventStream bool `json:"eventstream"`
	IsEvent       bool `json:"event"`

	refs       []*ShapeRef // References to this shape
	resolvePkg string      // use this package in the goType() if present

	OrigShapeName string `json:"-"`

	// Defines if the shape is a placeholder and should not be used directly
	Placeholder bool

	Deprecated    bool   `json:"deprecated"`
	DeprecatedMsg string `json:"deprecatedMessage"`

	Validations ShapeValidations

	// Error information that is set if the shape is an error shape.
	ErrorInfo ErrorInfo `json:"error"`

	// Flags that the shape cannot be rename. Prevents the shape from being
	// renamed further by the Input/Output.
	AliasedShapeName bool

	// Sensitive types should not be logged by SDK type loggers.
	Sensitive bool `json:"sensitive"`

	// Flags that a member of the shape is an EndpointARN
	HasEndpointARNMember bool

	// Flags that a member of the shape is an OutpostIDMember
	HasOutpostIDMember bool

	// Flags that the shape has an account id member along with EndpointARN member
	HasAccountIdMemberWithARN bool

	// Indicates the Shape is used as an operation input
	UsedAsInput bool

	// Indicates the Shape is used as an operation output
	UsedAsOutput bool
}

// CanBeEmpty returns if the shape value can sent request as an empty value.
// String, blob, list, and map are types must not be empty when the member is
// serialized to the URI path, or decorated with HostLabel.
func (ref *ShapeRef) CanBeEmpty() bool {
	switch ref.Shape.Type {
	case "string":
		return !(ref.Location == "uri" || ref.HostLabel)
	case "blob", "map", "list":
		return !(ref.Location == "uri")
	default:
		return true
	}
}

// ErrorCodeName will return the error shape's name formated for
// error code const.
func (s *Shape) ErrorCodeName() string {
	return "ErrCode" + s.ShapeName
}

// ErrorName will return the shape's name or error code if available based
// on the API's protocol. This is the error code string returned by the service.
func (s *Shape) ErrorName() string {
	name := s.ErrorInfo.Type
	switch s.API.Metadata.Protocol {
	case "query", "ec2query", "rest-xml":
		if len(s.ErrorInfo.Code) > 0 {
			name = s.ErrorInfo.Code
		}
	}

	if len(name) == 0 {
		name = s.OrigShapeName
	}
	if len(name) == 0 {
		name = s.ShapeName
	}

	return name
}

// PayloadRefName returns the payload member of the shape if there is one
// modeled. If no payload is modeled, empty string will be returned.
func (s *Shape) PayloadRefName() string {
	if name := s.Payload; len(name) != 0 {
		// Root shape
		return name
	}

	for name, ref := range s.MemberRefs {
		if ref.IsEventPayload {
			return name
		}
	}

	return ""
}

// GoTags returns the struct tags for a shape.
func (s *Shape) GoTags(root, required bool) string {
	ref := &ShapeRef{ShapeName: s.ShapeName, API: s.API, Shape: s}
	return ref.GoTags(root, required)
}

// Rename changes the name of the Shape to newName. Also updates
// the associated API's reference to use newName.
func (s *Shape) Rename(newName string) {
	if s.AliasedShapeName {
		panic(fmt.Sprintf("attempted to rename %s, but flagged as aliased",
			s.ShapeName))
	}

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

// HasMember will return whether or not the shape has a given
// member by name.
func (s *Shape) HasMember(name string) bool {
	_, ok := s.MemberRefs[name]
	return ok
}

// GoTypeWithPkgName returns a shape's type as a string with the package name in
// <packageName>.<type> format. Package naming only applies to structures.
func (s *Shape) GoTypeWithPkgName() string {
	return goType(s, true)
}

// GoTypeWithPkgNameElem returns the shapes type as a string with the "*"
// removed if there was one preset.
func (s *Shape) GoTypeWithPkgNameElem() string {
	t := goType(s, true)
	if strings.HasPrefix(t, "*") {
		return t[1:]
	}
	return t
}

// UseIndirection returns if the shape's reference should use indirection or not.
func (s *ShapeRef) UseIndirection() bool {
	switch s.Shape.Type {
	case "map", "list", "blob", "structure", "jsonvalue":
		return false
	}

	if s.Streaming || s.Shape.Streaming {
		return false
	}

	if s.JSONValue {
		return false
	}

	return true
}

func (s Shape) GetTimestampFormat() string {
	format := s.TimestampFormat

	if len(format) > 0 && !protocol.IsKnownTimestampFormat(format) {
		panic(fmt.Sprintf("Unknown timestampFormat %s, for %s",
			format, s.ShapeName))
	}

	return format
}

func (ref ShapeRef) GetTimestampFormat() string {
	format := ref.TimestampFormat

	if len(format) == 0 {
		format = ref.Shape.TimestampFormat
	}

	if len(format) > 0 && !protocol.IsKnownTimestampFormat(format) {
		panic(fmt.Sprintf("Unknown timestampFormat %s, for %s",
			format, ref.ShapeName))
	}

	return format
}

// GoStructValueType returns the Shape's Go type value instead of a pointer
// for the type.
func (s *Shape) GoStructValueType(name string, ref *ShapeRef) string {
	v := s.GoStructType(name, ref)

	if ref.UseIndirection() && v[0] == '*' {
		return v[1:]
	}

	return v
}

// GoStructType returns the type of a struct field based on the API
// model definition.
func (s *Shape) GoStructType(name string, ref *ShapeRef) string {
	if (ref.Streaming || ref.Shape.Streaming) && s.Payload == name {
		rtype := "io.ReadSeeker"
		if strings.HasSuffix(s.ShapeName, "Output") {
			rtype = "io.ReadCloser"
		}

		s.API.imports["io"] = true
		return rtype
	}

	if ref.JSONValue {
		s.API.AddSDKImport("aws")
		return "aws.JSONValue"
	}

	for _, v := range s.Validations {
		// TODO move this to shape validation resolution
		if (v.Ref.Shape.Type == "map" || v.Ref.Shape.Type == "list") && v.Type == ShapeValidationNested {
			s.API.imports["fmt"] = true
		}
	}

	return ref.GoType()
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
		return "map[string]" + goType(s.ValueRef.Shape, withPkgName)
	case "jsonvalue":
		return "aws.JSONValue"
	case "list":
		return "[]" + goType(s.MemberRef.Shape, withPkgName)
	case "boolean":
		return "*bool"
	case "string", "character":
		return "*string"
	case "blob":
		return "[]byte"
	case "byte", "short", "integer", "long":
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

// ShapeTag is a struct tag that will be applied to a shape's generated code
type ShapeTag struct {
	Key, Val string
}

// String returns the string representation of the shape tag
func (s ShapeTag) String() string {
	return fmt.Sprintf(`%s:"%s"`, s.Key, s.Val)
}

// ShapeTags is a collection of shape tags and provides serialization of the
// tags in an ordered list.
type ShapeTags []ShapeTag

// Join returns an ordered serialization of the shape tags with the provided
// separator.
func (s ShapeTags) Join(sep string) string {
	o := &bytes.Buffer{}
	for i, t := range s {
		o.WriteString(t.String())
		if i < len(s)-1 {
			o.WriteString(sep)
		}
	}

	return o.String()
}

// String is an alias for Join with the empty space separator.
func (s ShapeTags) String() string {
	return s.Join(" ")
}

// GoTags returns the rendered tags string for the ShapeRef
func (ref *ShapeRef) GoTags(toplevel bool, isRequired bool) string {
	tags := append(ShapeTags{}, ref.CustomTags...)

	if ref.Location != "" {
		tags = append(tags, ShapeTag{"location", ref.Location})
	} else if ref.Shape.Location != "" {
		tags = append(tags, ShapeTag{"location", ref.Shape.Location})
	} else if ref.IsEventHeader {
		tags = append(tags, ShapeTag{"location", "header"})
	}

	if ref.LocationName != "" {
		tags = append(tags, ShapeTag{"locationName", ref.LocationName})
	} else if ref.Shape.LocationName != "" {
		tags = append(tags, ShapeTag{"locationName", ref.Shape.LocationName})
	} else if len(ref.Shape.EventFor) != 0 && ref.API.Metadata.Protocol == "rest-xml" {
		// RPC JSON events need to have location name modeled for round trip testing.
		tags = append(tags, ShapeTag{"locationName", ref.Shape.OrigShapeName})
	}

	if ref.QueryName != "" {
		tags = append(tags, ShapeTag{"queryName", ref.QueryName})
	}
	if ref.Shape.MemberRef.LocationName != "" {
		tags = append(tags, ShapeTag{"locationNameList", ref.Shape.MemberRef.LocationName})
	}
	if ref.Shape.KeyRef.LocationName != "" {
		tags = append(tags, ShapeTag{"locationNameKey", ref.Shape.KeyRef.LocationName})
	}
	if ref.Shape.ValueRef.LocationName != "" {
		tags = append(tags, ShapeTag{"locationNameValue", ref.Shape.ValueRef.LocationName})
	}
	if ref.Shape.Min > 0 {
		tags = append(tags, ShapeTag{"min", fmt.Sprintf("%v", ref.Shape.Min)})
	}

	if ref.Deprecated || ref.Shape.Deprecated {
		tags = append(tags, ShapeTag{"deprecated", "true"})
	}

	// All shapes have a type
	tags = append(tags, ShapeTag{"type", ref.Shape.Type})

	// embed the timestamp type for easier lookups
	if ref.Shape.Type == "timestamp" {
		if format := ref.GetTimestampFormat(); len(format) > 0 {
			tags = append(tags, ShapeTag{
				Key: "timestampFormat",
				Val: format,
			})
		}
	}

	if ref.Shape.Flattened || ref.Flattened {
		tags = append(tags, ShapeTag{"flattened", "true"})
	}
	if ref.XMLAttribute {
		tags = append(tags, ShapeTag{"xmlAttribute", "true"})
	}
	if isRequired {
		tags = append(tags, ShapeTag{"required", "true"})
	}
	if ref.Shape.IsEnum() {
		tags = append(tags, ShapeTag{"enum", ref.ShapeName})
	}

	if toplevel {
		if name := ref.Shape.PayloadRefName(); len(name) > 0 {
			tags = append(tags, ShapeTag{"payload", name})
		}
	}

	if ref.XMLNamespace.Prefix != "" {
		tags = append(tags, ShapeTag{"xmlPrefix", ref.XMLNamespace.Prefix})
	} else if ref.Shape.XMLNamespace.Prefix != "" {
		tags = append(tags, ShapeTag{"xmlPrefix", ref.Shape.XMLNamespace.Prefix})
	}

	if ref.XMLNamespace.URI != "" {
		tags = append(tags, ShapeTag{"xmlURI", ref.XMLNamespace.URI})
	} else if ref.Shape.XMLNamespace.URI != "" {
		tags = append(tags, ShapeTag{"xmlURI", ref.Shape.XMLNamespace.URI})
	}

	if ref.IdempotencyToken || ref.Shape.IdempotencyToken {
		tags = append(tags, ShapeTag{"idempotencyToken", "true"})
	}

	if ref.Ignore {
		tags = append(tags, ShapeTag{"ignore", "true"})
	}

	if ref.Shape.Sensitive {
		tags = append(tags, ShapeTag{"sensitive", "true"})
	}

	return fmt.Sprintf("`%s`", tags)
}

// Docstring returns the godocs formated documentation
func (ref *ShapeRef) Docstring() string {
	if ref.Documentation != "" {
		return strings.Trim(ref.Documentation, "\n ")
	}
	return ref.Shape.Docstring()
}

// Docstring returns the godocs formated documentation
func (s *Shape) Docstring() string {
	return strings.Trim(s.Documentation, "\n ")
}

// IndentedDocstring is the indented form of the doc string.
func (ref *ShapeRef) IndentedDocstring() string {
	doc := ref.Docstring()
	return strings.Replace(doc, "// ", "//   ", -1)
}

var goCodeStringerTmpl = template.Must(template.New("goCodeStringerTmpl").Parse(`
// String returns the string representation
func (s {{ $.ShapeName }}) String() string {
	return awsutil.Prettify(s)
}
// GoString returns the string representation
func (s {{ $.ShapeName }}) GoString() string {
	return s.String()
}
`))

// GoCodeStringers renders the Stringers for API input/output shapes
func (s *Shape) GoCodeStringers() string {
	w := bytes.Buffer{}
	if err := goCodeStringerTmpl.Execute(&w, s); err != nil {
		panic(fmt.Sprintln("Unexpected error executing GoCodeStringers template", err))
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

// NestedShape returns the shape pointer value for the shape which is nested
// under the current shape. If the shape is not nested nil will be returned.
//
// strucutures, the current shape is returned
// map: the value shape of the map is returned
// list: the element shape of the list is returned
func (s *Shape) NestedShape() *Shape {
	var nestedShape *Shape
	switch s.Type {
	case "structure":
		nestedShape = s
	case "map":
		nestedShape = s.ValueRef.Shape
	case "list":
		nestedShape = s.MemberRef.Shape
	}

	return nestedShape
}

var structShapeTmpl = func() *template.Template {
	shapeTmpl := template.Must(
		template.New("structShapeTmpl").
			Funcs(template.FuncMap{
				"GetDeprecatedMsg": getDeprecatedMessage,
				"TrimExportedMembers": func(s *Shape) map[string]*ShapeRef {
					members := map[string]*ShapeRef{}
					for name, ref := range s.MemberRefs {
						if ref.Shape.IsEventStream {
							continue
						}
						members[name] = ref
					}
					return members
				},
			}).
			Parse(structShapeTmplDef),
	)

	template.Must(
		shapeTmpl.AddParseTree(
			"eventStreamEventShapeTmpl", eventStreamEventShapeTmpl.Tree),
	)
	template.Must(
		shapeTmpl.AddParseTree(
			"exceptionShapeMethodTmpl",
			exceptionShapeMethodTmpl.Tree),
	)
	shapeTmpl.Funcs(eventStreamEventShapeTmplFuncs)

	template.Must(
		shapeTmpl.AddParseTree(
			"hostLabelsShapeTmpl",
			hostLabelsShapeTmpl.Tree),
	)

	template.Must(
		shapeTmpl.AddParseTree(
			"endpointARNShapeTmpl",
			endpointARNShapeTmpl.Tree),
	)

	template.Must(
		shapeTmpl.AddParseTree(
			"outpostIDShapeTmpl",
			outpostIDShapeTmpl.Tree),
	)

	template.Must(
		shapeTmpl.AddParseTree(
			"accountIDWithARNShapeTmpl",
			accountIDWithARNShapeTmpl.Tree),
	)

	return shapeTmpl
}()

const structShapeTmplDef = `
{{ $.Docstring }}
{{ if $.Deprecated -}}
{{ if $.Docstring -}}
//
{{ end -}}
// Deprecated: {{ GetDeprecatedMsg $.DeprecatedMsg $.ShapeName }}
{{ end -}}
type {{ $.ShapeName }} struct {
	_ struct{} {{ $.GoTags true false }}

	{{- if $.Exception }}
		{{- $_ := $.API.AddSDKImport "private/protocol" }}
		RespMetadata protocol.ResponseMetadata` + "`json:\"-\" xml:\"-\"`" + `
	{{- end }}

	{{- if $.OutputEventStreamAPI }}

		{{ $.OutputEventStreamAPI.OutputMemberName }} *{{ $.OutputEventStreamAPI.Name }}
	{{- end }}

	{{- range $name, $elem := (TrimExportedMembers $) }}

		{{ $isBlob := $.WillRefBeBase64Encoded $name -}}
		{{ $isRequired := $.IsRequired $name -}}
		{{ $doc := $elem.Docstring -}}

		{{ if $doc -}}
			{{ $doc }}
			{{ if $elem.Deprecated -}}
			//
			// Deprecated: {{ GetDeprecatedMsg $elem.DeprecatedMsg $name }}
			{{ end -}}
		{{ end -}}
		{{ if $isBlob -}}
			{{ if $doc -}}
				//
			{{ end -}}
			// {{ $name }} is automatically base64 encoded/decoded by the SDK.
		{{ end -}}
		{{ if $isRequired -}}
			{{ if or $doc $isBlob -}}
				//
			{{ end -}}
			// {{ $name }} is a required field
		{{ end -}}
		{{ $name }} {{ $.GoStructType $name $elem }} {{ $elem.GoTags false $isRequired }}
	{{- end }}
}

{{- if not $.API.NoStringerMethods }}
	{{ $.GoCodeStringers }}
{{- end }}

{{- if not (or $.API.NoValidataShapeMethods $.Exception) }}
	{{- if $.Validations }}
		{{ $.Validations.GoCode $ }}
	{{- end }}
{{- end }}

{{- if not (or $.API.NoGenStructFieldAccessors $.Exception) }}
	{{- $builderShapeName := print $.ShapeName }}

	{{- range $name, $elem := (TrimExportedMembers $) }}
		// Set{{ $name }} sets the {{ $name }} field's value.
		func (s *{{ $builderShapeName }}) Set{{ $name }}(v {{ $.GoStructValueType $name $elem }}) *{{ $builderShapeName }} {
			{{- if $elem.UseIndirection }}
				s.{{ $name }} = &v
			{{- else }}
				s.{{ $name }} = v
			{{- end }}
			return s
		}

		{{- if $elem.GenerateGetter }}

			func (s *{{ $builderShapeName }}) get{{ $name }}() (v {{ $.GoStructValueType $name $elem }}) {
				{{- if $elem.UseIndirection }}
					if s.{{ $name }} == nil {
						return v
					}
					return *s.{{ $name }}
				{{- else }}
					return s.{{ $name }}
				{{- end }}
			}
		{{- end }}
	{{- end }}
{{- end }}

{{- if $.OutputEventStreamAPI }}
	{{- $esMemberName := $.OutputEventStreamAPI.OutputMemberName }}
	{{- if $.OutputEventStreamAPI.Legacy }}
		func (s *{{ $.ShapeName }}) Set{{ $esMemberName }}(v *{{ $.OutputEventStreamAPI.Name }}) *{{ $.ShapeName }} {
			s.{{ $esMemberName }} = v
			return s
		}
		func (s *{{ $.ShapeName }}) Get{{ $esMemberName }}() *{{ $.OutputEventStreamAPI.Name }} {
			return s.{{ $esMemberName }}
		}
	{{- end }}

	// GetStream returns the type to interact with the event stream.
	func (s *{{ $.ShapeName }}) GetStream() *{{ $.OutputEventStreamAPI.Name }} {
		return s.{{ $esMemberName }}
	}
{{- end }}

{{- if $.EventFor }}
	{{ template "eventStreamEventShapeTmpl" $ }}
{{- end }}

{{- if and $.Exception (or $.API.WithGeneratedTypedErrors $.EventFor) }}
	{{ template "exceptionShapeMethodTmpl" $ }}
{{- end }}

{{- if $.HasHostLabelMembers }}
	{{ template "hostLabelsShapeTmpl" $ }}
{{- end }}

{{- if $.HasEndpointARNMember }}
	{{ template "endpointARNShapeTmpl" $ }}
{{- end }}

{{- if $.HasOutpostIDMember }}
	{{ template "outpostIDShapeTmpl" $ }}
{{- end }}

{{- if $.HasAccountIdMemberWithARN }}
	{{ template "accountIDWithARNShapeTmpl" $ }}
{{- end }}

`

var exceptionShapeMethodTmpl = template.Must(
	template.New("exceptionShapeMethodTmpl").Parse(`
{{- $_ := $.API.AddImport "fmt" }}
{{/* TODO allow service custom input to be used */}}
func newError{{ $.ShapeName }}(v protocol.ResponseMetadata) error {
	return &{{ $.ShapeName }}{
		RespMetadata: v,
	}
}

// Code returns the exception type name.
func (s *{{ $.ShapeName }}) Code() string {
	return "{{ $.ErrorName }}"
}

// Message returns the exception's message.
func (s *{{ $.ShapeName }}) Message() string {
	{{- if index $.MemberRefs "Message_" }}
		if s.Message_ != nil {
			return *s.Message_
		}
	{{ end -}}
	return ""
}

// OrigErr always returns nil, satisfies awserr.Error interface.
func (s *{{ $.ShapeName }}) OrigErr() error {
	return nil
}

func (s *{{ $.ShapeName }}) Error() string {
	{{- if or (and (eq (len $.MemberRefs) 1) (not (index $.MemberRefs "Message_"))) (gt (len $.MemberRefs) 1) }}
		return fmt.Sprintf("%s: %s\n%s", s.Code(), s.Message(), s.String())
	{{- else }}
		return fmt.Sprintf("%s: %s", s.Code(), s.Message())
	{{- end }}
}

// Status code returns the HTTP status code for the request's response error.
func (s *{{ $.ShapeName }}) StatusCode() int {
	return s.RespMetadata.StatusCode
}

// RequestID returns the service's response RequestID for request.
func (s *{{ $.ShapeName }}) RequestID() string {
	return s.RespMetadata.RequestID
}
`))

var enumShapeTmpl = template.Must(template.New("EnumShape").Parse(`
{{ $.Docstring }}
const (
	{{ range $index, $elem := $.Enum -}}
		{{ $name := index $.EnumConsts $index -}}
		// {{ $name }} is a {{ $.ShapeName }} enum value
		{{ $name }} = "{{ $elem }}"

	{{ end }}
)

{{/* Enum iterators use non-idomatic _Values suffix to avoid naming collisions with other generated types, and enum values */}}
// {{ $.ShapeName }}_Values returns all elements of the {{ $.ShapeName }} enum
func {{ $.ShapeName }}_Values() []string {
	return []string{
		{{ range $index, $elem := $.Enum -}}
		{{ index $.EnumConsts $index }},
		{{ end }}
	}
}
`))

// GoCode returns the rendered Go code for the Shape.
func (s *Shape) GoCode() string {
	w := &bytes.Buffer{}

	switch {
	case s.IsEventStream:
		if err := renderEventStreamShape(w, s); err != nil {
			panic(
				fmt.Sprintf(
					"failed to generate eventstream API shape, %s, %v",
					s.ShapeName, err),
			)
		}
	case s.Type == "structure":
		if err := structShapeTmpl.Execute(w, s); err != nil {
			panic(
				fmt.Sprintf(
					"Failed to generate struct shape %s, %v",
					s.ShapeName, err),
			)
		}
	case s.IsEnum():
		if err := enumShapeTmpl.Execute(w, s); err != nil {
			panic(
				fmt.Sprintf(
					"Failed to generate enum shape %s, %v",
					s.ShapeName, err),
			)
		}
	default:
		panic(fmt.Sprintln("Cannot generate toplevel shape for", s.Type))
	}

	return w.String()
}

// IsEnum returns whether this shape is an enum list
func (s *Shape) IsEnum() bool {
	return s.Type == "string" && len(s.Enum) > 0
}

// IsRequired returns if member is a required field. Required fields are fields
// marked as required, hostLabels, or location of uri path.
func (s *Shape) IsRequired(member string) bool {
	ref, ok := s.MemberRefs[member]
	if !ok {
		panic(fmt.Sprintf(
			"attempted to check required for unknown member, %s.%s",
			s.ShapeName, member,
		))
	}
	if ref.IdempotencyToken || ref.Shape.IdempotencyToken {
		return false
	}
	if ref.Location == "uri" || ref.HostLabel {
		return true
	}
	for _, n := range s.Required {
		if n == member {
			if ref.Shape.IsEventStream {
				return false
			}
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

func (s *Shape) WillRefBeBase64Encoded(refName string) bool {
	payloadRefName := s.Payload
	if payloadRefName == refName {
		return false
	}

	ref, ok := s.MemberRefs[refName]
	if !ok {
		panic(fmt.Sprintf("shape %s does not contain %q refName", s.ShapeName, refName))
	}

	return ref.Shape.Type == "blob"
}

// Clone returns a cloned version of the shape with all references clones.
//
// Does not clone EventStream or Validate related values.
func (s *Shape) Clone(newName string) *Shape {
	if s.AliasedShapeName {
		panic(fmt.Sprintf("attempted to clone and rename %s, but flagged as aliased",
			s.ShapeName))
	}

	n := new(Shape)
	*n = *s

	debugLogger.Logln("cloning", s.ShapeName, "to", newName)

	n.MemberRefs = map[string]*ShapeRef{}
	for k, r := range s.MemberRefs {
		nr := new(ShapeRef)
		*nr = *r
		nr.Shape.refs = append(nr.Shape.refs, nr)
		n.MemberRefs[k] = nr
	}

	if n.MemberRef.Shape != nil {
		n.MemberRef.Shape.refs = append(n.MemberRef.Shape.refs, &n.MemberRef)
	}
	if n.KeyRef.Shape != nil {
		n.KeyRef.Shape.refs = append(n.KeyRef.Shape.refs, &n.KeyRef)
	}
	if n.ValueRef.Shape != nil {
		n.ValueRef.Shape.refs = append(n.ValueRef.Shape.refs, &n.ValueRef)
	}

	n.refs = []*ShapeRef{}

	n.Required = append([]string{}, n.Required...)
	n.Enum = append([]string{}, n.Enum...)
	n.EnumConsts = append([]string{}, n.EnumConsts...)

	n.API.Shapes[newName] = n
	n.ShapeName = newName
	n.OrigShapeName = s.OrigShapeName

	return n
}
