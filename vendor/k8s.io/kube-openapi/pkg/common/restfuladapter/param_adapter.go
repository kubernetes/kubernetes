package restfuladapter

import (
	"encoding/json"
	"github.com/emicklei/go-restful/v3"
	"k8s.io/kube-openapi/pkg/common"
)

var _ common.Parameter = &ParamAdapter{}

type ParamAdapter struct {
	Param *restful.Parameter
}

func (r *ParamAdapter) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.Param)
}

func (r *ParamAdapter) Name() string {
	return r.Param.Data().Name
}

func (r *ParamAdapter) Description() string {
	return r.Param.Data().Description
}

func (r *ParamAdapter) Required() bool {
	return r.Param.Data().Required
}

func (r *ParamAdapter) Kind() common.ParameterKind {
	switch r.Param.Kind() {
	case restful.PathParameterKind:
		return common.PathParameterKind
	case restful.QueryParameterKind:
		return common.QueryParameterKind
	case restful.BodyParameterKind:
		return common.BodyParameterKind
	case restful.HeaderParameterKind:
		return common.HeaderParameterKind
	case restful.FormParameterKind:
		return common.FormParameterKind
	default:
		return common.UnknownParameterKind
	}
}

func (r *ParamAdapter) DataType() string {
	return r.Param.Data().DataType
}

func (r *ParamAdapter) AllowMultiple() bool {
	return r.Param.Data().AllowMultiple
}
