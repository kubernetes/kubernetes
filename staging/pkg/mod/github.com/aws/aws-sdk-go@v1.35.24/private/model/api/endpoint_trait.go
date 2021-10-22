// +build codegen

package api

import (
	"fmt"
	"text/template"
)

func setupEndpointHostPrefix(op *Operation) {
	op.API.AddSDKImport("private/protocol")

	buildHandler := fmt.Sprintf("protocol.NewHostPrefixHandler(%q, ",
		op.Endpoint.HostPrefix)

	if op.InputRef.Shape.HasHostLabelMembers() {
		buildHandler += "input.hostLabels"
	} else {
		buildHandler += "nil"
	}

	buildHandler += ")"

	op.CustomBuildHandlers = append(op.CustomBuildHandlers,
		buildHandler,
		"protocol.ValidateEndpointHostHandler",
	)
}

// HasHostLabelMembers returns true if the shape contains any members which are
// decorated with the hostLabel trait.
func (s *Shape) HasHostLabelMembers() bool {
	for _, ref := range s.MemberRefs {
		if ref.HostLabel {
			return true
		}
	}

	return false
}

var hostLabelsShapeTmpl = template.Must(
	template.New("hostLabelsShapeTmpl").
		Parse(hostLabelsShapeTmplDef),
)

const hostLabelsShapeTmplDef = `
{{- define "hostLabelsShapeTmpl" }}
func (s *{{ $.ShapeName }}) hostLabels() map[string]string {
	return map[string]string{
	{{- range $name, $ref := $.MemberRefs }}
		{{- if $ref.HostLabel }}
		"{{ $name }}": aws.StringValue(s.{{ $name }}),
		{{- end }}
	{{- end }}
	}
}
{{- end }}
`
