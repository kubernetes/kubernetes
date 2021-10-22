package api

import "text/template"

const endpointARNShapeTmplDef = `
{{- define "endpointARNShapeTmpl" }}
{{ range $_, $name := $.MemberNames -}}
	{{ $elem := index $.MemberRefs $name -}}
	{{ if $elem.EndpointARN -}}
		func (s *{{ $.ShapeName }}) getEndpointARN() (arn.Resource, error) {
			if s.{{ $name }} == nil {
				return nil, fmt.Errorf("member {{ $name }} is nil")
			}
			return parseEndpointARN(*s.{{ $name }})
		}

		func (s *{{ $.ShapeName }}) hasEndpointARN() bool {
			if s.{{ $name }} == nil {
				return false
			}
			return arn.IsARN(*s.{{ $name }})
		}

		// updateArnableField updates the value of the input field that 
		// takes an ARN as an input. This method is useful to backfill 
		// the parsed resource name from ARN into the input member.
		// It returns a pointer to a modified copy of input and an error.
		// Note that original input is not modified. 
		func (s {{ $.ShapeName }}) updateArnableField(v string) (interface{}, error) {
			if s.{{ $name }} == nil {
				return nil, fmt.Errorf("member {{ $name }} is nil")
			}
			s.{{ $name }} = aws.String(v)
			return &s, nil 
		}
	{{ end -}}
{{ end }}
{{ end }}
`

var endpointARNShapeTmpl = template.Must(
	template.New("endpointARNShapeTmpl").
		Parse(endpointARNShapeTmplDef),
)

const outpostIDShapeTmplDef = `
{{- define "outpostIDShapeTmpl" }}
{{ range $_, $name := $.MemberNames -}}
	{{ $elem := index $.MemberRefs $name -}}
	{{ if $elem.OutpostIDMember -}}
		func (s *{{ $.ShapeName }}) getOutpostID() (string, error) {
			if s.{{ $name }} == nil {
				return "", fmt.Errorf("member {{ $name }} is nil")
			}
			return *s.{{ $name }}, nil
		}

		func (s *{{ $.ShapeName }}) hasOutpostID() bool {
			if s.{{ $name }} == nil {
				return false
			}
			return true 
		}
	{{ end -}}
{{ end }}
{{ end }}
`

var outpostIDShapeTmpl = template.Must(
	template.New("outpostIDShapeTmpl").
		Parse(outpostIDShapeTmplDef),
)

const accountIDWithARNShapeTmplDef = `
{{- define "accountIDWithARNShapeTmpl" }}
{{ range $_, $name := $.MemberNames -}}
	{{ $elem := index $.MemberRefs $name -}}
	{{ if $elem.AccountIDMemberWithARN -}}
		// updateAccountID returns a pointer to a modified copy of input, 
		// if account id is not provided, we update the account id in modified input
		// if account id is provided, but doesn't match with the one in ARN, we throw an error
		// if account id is not updated, we return nil. Note that original input is not modified. 
		func (s {{ $.ShapeName }}) updateAccountID(accountId string) (interface{}, error) {
			if s.{{ $name }} == nil {
				s.{{ $name }} = aws.String(accountId)
				return &s, nil
			} else if *s.{{ $name }} != accountId  {
				return &s, fmt.Errorf("Account ID mismatch, the Account ID cannot be specified in an ARN and in the accountId field")
			}
			return nil, nil
		}
	{{ end -}}
{{ end }}
{{ end }}
`

var accountIDWithARNShapeTmpl = template.Must(
	template.New("accountIDWithARNShapeTmpl").
		Parse(accountIDWithARNShapeTmplDef),
)
