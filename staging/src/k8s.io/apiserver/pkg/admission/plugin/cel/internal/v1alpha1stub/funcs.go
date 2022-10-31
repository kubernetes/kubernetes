package v1alpha1stub

import (
	"fmt"
	"strings"
)

func (p *ParamKind) String() string {
	if p == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ParamKind{`,
		`APIVersion:` + fmt.Sprintf("%v", p.APIVersion) + `,`,
		`Kind:` + fmt.Sprintf("%v", p.Kind) + `,`,
		`}`,
	}, "")
	return s
}
