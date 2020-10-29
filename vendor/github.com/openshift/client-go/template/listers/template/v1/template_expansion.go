package v1

import (
	templateapiv1 "github.com/openshift/api/template/v1"
	"k8s.io/apimachinery/pkg/api/errors"
)

const TemplateUIDIndex = "templateuid"

// TemplateListerExpansion allows custom methods to be added to
// TemplateLister.
type TemplateListerExpansion interface {
	GetByUID(uid string) (*templateapiv1.Template, error)
}

// TemplateNamespaceListerExpansion allows custom methods to be added to
// TemplateNamespaceLister.
type TemplateNamespaceListerExpansion interface{}

func (s templateLister) GetByUID(uid string) (*templateapiv1.Template, error) {
	templates, err := s.indexer.ByIndex(TemplateUIDIndex, uid)
	if err != nil {
		return nil, err
	}
	if len(templates) == 0 {
		return nil, errors.NewNotFound(templateapiv1.Resource("template"), uid)
	}
	return templates[0].(*templateapiv1.Template), nil
}
