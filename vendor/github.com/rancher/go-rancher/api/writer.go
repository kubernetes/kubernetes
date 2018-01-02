package api

import (
	"fmt"
	"net/http"
	"reflect"

	"github.com/Sirupsen/logrus"
)

type ApiResponseWriter interface {
	Write(obj interface{}, rw http.ResponseWriter) error
}

func (a *ApiContext) Write(obj interface{}) {
	var err error
	collection := getCollection(obj)
	if collection != nil {
		err = a.WriteCollection(obj)
	} else {
		err = a.WriteResource(obj)
	}

	if err != nil {
		logrus.WithField("err", err).Errorf("Failed to write response")
		a.responseWriter.WriteHeader(500)
	}
}

func stringSetIfNot(data map[string]string, key, value string) map[string]string {
	if v, ok := data[key]; !ok || v == "" {
		data[key] = value
	}

	return data
}

func setIfNot(data map[string]interface{}, key string, value interface{}) map[string]interface{} {
	if v, ok := data[key]; !ok || v == nil {
		data[key] = value
	}

	return data
}

func (a *ApiContext) WriteCollection(obj interface{}) error {
	collectionData, resourcesData, err := CollectionToMap(obj, a.schemas)
	if err != nil {
		return err
	}

	a.populateCollection(collectionData, resourcesData)
	for _, resource := range resourcesData {
		a.populateResource(resource)
	}

	return a.apiResponseWriter.Write(collectionData, a.responseWriter)
}

func (a *ApiContext) WriteResource(obj interface{}) error {
	resourceData, err := ResourceToMap(obj, a.schemas)
	if err != nil {
		return err
	}

	a.populateResource(resourceData)
	return a.apiResponseWriter.Write(resourceData, a.responseWriter)
}

func mapInterfaceToString(input interface{}) map[string]string {
	result := map[string]string{}
	if input == nil {
		return result
	}
	switch i := input.(type) {
	case map[string]string:
		return i
	case map[string]interface{}:
		for k, v := range i {
			result[k] = fmt.Sprintf("%s", v)
		}
	default:
		logrus.Infof("Unknown type", reflect.TypeOf(input))
	}
	return result
}

func getString(data map[string]interface{}, key string) string {
	if v, ok := data[key]; ok {
		return fmt.Sprintf("%s", v)
	} else {
		return ""
	}

}

func getStringMap(data map[string]interface{}, key string) map[string]string {
	result := map[string]string{}
	if v, ok := data[key]; ok {
		result = mapInterfaceToString(v)
	}
	data[key] = result
	return result
}

func (a *ApiContext) populateCollection(c map[string]interface{}, data []map[string]interface{}) {
	stringSetIfNot(getStringMap(c, "links"), "self", a.UrlBuilder.Current())

	if _, ok := c["type"]; !ok {
		c["type"] = "collection"
	}

	if len(data) > 0 {
		setIfNot(c, "resourceType", data[0]["type"])
	}
}

func (a *ApiContext) populateResource(c map[string]interface{}) {
	resourceType := getString(c, "type")
	resourceId := getString(c, "id")

	if resourceType != "" && resourceId != "" {
		stringSetIfNot(getStringMap(c, "links"), "self", a.UrlBuilder.ReferenceByIdLink(resourceType, resourceId))
	}

	stringSetIfNot(getStringMap(c, "links"), "self", a.UrlBuilder.Current())
	// This will create an empty map if not
	getStringMap(c, "actions")
}
