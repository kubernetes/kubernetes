package restfuladapter

import (
	"github.com/emicklei/go-restful/v3"
	"k8s.io/kube-openapi/pkg/common"
)

var _ common.Route = &RouteAdapter{}

// RouteAdapter adapts a restful.Route to common.Route.
type RouteAdapter struct {
	Route *restful.Route
}

func (r *RouteAdapter) StatusCodeResponses() []common.StatusCodeResponse {
	// go-restful uses the ResponseErrors field to contain both error and regular responses.
	var responses []common.StatusCodeResponse
	for _, res := range r.Route.ResponseErrors {
		localRes := res
		responses = append(responses, &ResponseErrorAdapter{&localRes})
	}

	return responses
}

func (r *RouteAdapter) OperationName() string {
	return r.Route.Operation
}

func (r *RouteAdapter) Method() string {
	return r.Route.Method
}

func (r *RouteAdapter) Path() string {
	return r.Route.Path
}

func (r *RouteAdapter) Parameters() []common.Parameter {
	var params []common.Parameter
	for _, rParam := range r.Route.ParameterDocs {
		params = append(params, &ParamAdapter{rParam})
	}
	return params
}

func (r *RouteAdapter) Description() string {
	return r.Route.Doc
}

func (r *RouteAdapter) Consumes() []string {
	return r.Route.Consumes
}

func (r *RouteAdapter) Produces() []string {
	return r.Route.Produces
}

func (r *RouteAdapter) Metadata() map[string]interface{} {
	return r.Route.Metadata
}

func (r *RouteAdapter) RequestPayloadSample() interface{} {
	return r.Route.ReadSample
}

func (r *RouteAdapter) ResponsePayloadSample() interface{} {
	return r.Route.WriteSample
}
