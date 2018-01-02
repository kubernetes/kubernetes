package api

import (
	"net/http"

	"github.com/Sirupsen/logrus"
	"github.com/gorilla/context"
	"github.com/gorilla/mux"
	"github.com/rancher/go-rancher/client"
)

func ApiHandler(schemas *client.Schemas, f http.Handler) http.Handler {
	return context.ClearHandler(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if err := CreateApiContext(rw, r, schemas); err != nil {
			logrus.WithField("err", err).Errorf("Failed to create API context")
			rw.WriteHeader(500)
			return
		}

		f.ServeHTTP(rw, r)
	}))
}

func VersionsHandler(schemas *client.Schemas, versions ...string) http.Handler {
	return ApiHandler(schemas, http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		apiContext := GetApiContext(r)

		collection := client.GenericCollection{
			Collection: client.Collection{
				Links: map[string]string{},
			},
			Data: []interface{}{},
		}

		for i, version := range versions {
			collection.Data = append(collection.Data, client.Resource{
				Id: version,
				Links: map[string]string{
					SELF: apiContext.UrlBuilder.Version(version),
				},
				Type: "apiVersion",
			})

			if i == len(versions)-1 {
				collection.Links[LATEST] = apiContext.UrlBuilder.Version(version)
			}
		}

		apiContext.Write(&collection)
	}))
}

func contains(list []string, item string) bool {
	for _, i := range list {
		if i == item {
			return true
		}
	}
	return false
}

func VersionHandler(schemas *client.Schemas, version string) http.Handler {
	return ApiHandler(schemas, http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		apiContext := GetApiContext(r)

		versionResource := client.Resource{
			Id:    version,
			Type:  "apiVersion",
			Links: map[string]string{},
		}

		for _, schema := range schemas.Data {
			if contains(schema.CollectionMethods, "GET") {
				versionResource.Links[schema.PluralName] = apiContext.UrlBuilder.Collection(schema.Id)
			}
		}

		apiContext.Write(&versionResource)
	}))
}

func SchemasHandler(schemas *client.Schemas) http.Handler {
	return ApiHandler(schemas, http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		apiContext := GetApiContext(r)
		schemasCopy := *schemas
		for i := range schemasCopy.Data {
			populateSchema(apiContext, &schemasCopy.Data[i])
		}
		apiContext.Write(&schemasCopy)
	}))
}

func populateSchema(apiContext *ApiContext, schema *client.Schema) {
	if contains(schema.CollectionMethods, "GET") {
		schema.Links["collection"] = apiContext.UrlBuilder.Collection(schema.Id)
	}
}

func SchemaHandler(schemas *client.Schemas) http.Handler {
	return ApiHandler(schemas, http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		apiContext := GetApiContext(r)

		schema := schemas.Schema(mux.Vars(r)["id"])

		populateSchema(apiContext, &schema)

		apiContext.Write(&schema)
	}))
}
