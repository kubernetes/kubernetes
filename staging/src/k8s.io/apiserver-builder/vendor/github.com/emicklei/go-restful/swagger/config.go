package swagger

import (
	"net/http"
	"reflect"

	"github.com/emicklei/go-restful"
)

// PostBuildDeclarationMapFunc can be used to modify the api declaration map.
type PostBuildDeclarationMapFunc func(apiDeclarationMap *ApiDeclarationList)

// MapSchemaFormatFunc can be used to modify typeName at definition time.
type MapSchemaFormatFunc func(typeName string) string

// MapModelTypeNameFunc can be used to return the desired typeName for a given
// type. It will return false if the default name should be used.
type MapModelTypeNameFunc func(t reflect.Type) (string, bool)

type Config struct {
	// url where the services are available, e.g. http://localhost:8080
	// if left empty then the basePath of Swagger is taken from the actual request
	WebServicesUrl string
	// path where the JSON api is avaiable , e.g. /apidocs
	ApiPath string
	// [optional] path where the swagger UI will be served, e.g. /swagger
	SwaggerPath string
	// [optional] location of folder containing Swagger HTML5 application index.html
	SwaggerFilePath string
	// api listing is constructed from this list of restful WebServices.
	WebServices []*restful.WebService
	// will serve all static content (scripts,pages,images)
	StaticHandler http.Handler
	// [optional] on default CORS (Cross-Origin-Resource-Sharing) is enabled.
	DisableCORS bool
	// Top-level API version. Is reflected in the resource listing.
	ApiVersion string
	// If set then call this handler after building the complete ApiDeclaration Map
	PostBuildHandler PostBuildDeclarationMapFunc
	// Swagger global info struct
	Info Info
	// [optional] If set, model builder should call this handler to get addition typename-to-swagger-format-field conversion.
	SchemaFormatHandler MapSchemaFormatFunc
	// [optional] If set, model builder should call this handler to retrieve the name for a given type.
	ModelTypeNameHandler MapModelTypeNameFunc
}
