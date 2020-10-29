package options

import (
	"k8s.io/apimachinery/pkg/api/meta"

	"github.com/openshift/library-go/pkg/client/openshiftrestmapper"
)

func NewAdmissionRESTMapper(delegate meta.RESTMapper) meta.RESTMapper {
	return openshiftrestmapper.NewOpenShiftHardcodedRESTMapper(delegate)
}
