package main

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"regexp"
	"strings"
	"text/template"

	yaml "gopkg.in/yaml.v2"

	"github.com/rancher/go-rancher/client"
)

var (
	commonFieldsMap        map[string]bool
	schemaMap              map[string]client.Schema
	referenceRegexp        *regexp.Regexp = regexp.MustCompile(`reference\[([a-zA-Z]+)\]`)
	descriptionsMap        map[string]string
	genericDescriptionsMap map[string]string
)

func init() {
	commonFieldsMap = make(map[string]bool)
	schemaMap = make(map[string]client.Schema)
	descriptionsMap = make(map[string]string)
	genericDescriptionsMap = make(map[string]string)
}

//APIField is to add in description and provide URL
type APIField struct {
	client.Field
	Description string `json:"description"`
	TypeURL     string
}

type APIAction struct {
	Input       ActionInput
	Output      string
	Description string `json:"description"`
	Method      string
	ActionURL   string
}

type ActionInput struct {
	Name      string
	FieldMap  map[string]APIField
	InputJSON string
}

func generateFiles() error {
	schemas, err := readCattleSchema()

	if err != nil {
		return err
	}

	if err = readBlacklistFiles(); err != nil {
		return err
	}

	if err = readInputFiles(); err != nil {
		return err
	}

	if err = setupDirectory(apiOutputDir + "/api-resources/"); err != nil {
		return err
	}

	//Create a map of all resources but exclude the blacklist items

	for _, resourceSchema := range schemas.Data {
		//Filter out any blacklist types
		if blacklistTypes[resourceSchema.Id] {
			continue
		}

		//Create a new Resource Action Map to eliminate any blacklist actions
		resourceActionMap := make(map[string]client.Action)

		//Add in check if resourceAction should be should be visible
		for resourceAction, resourceActionValue := range resourceSchema.ResourceActions {
			if !isBlacklistAction(resourceSchema.Id, resourceAction) {
				resourceActionMap[resourceAction] = resourceActionValue
			}
		}

		//Update the resource actions to the new resource action map
		resourceSchema.ResourceActions = resourceActionMap

		if !isBlacklistCollection(resourceSchema.Id) {
			for key := range resourceSchema.Resource.Links {
				if key == "collection" {
					//Add a link to show the resource for the visible pages
					resourceSchema.Resource.Links["showResource"] = "true"
				}
			}
		}

		schemaMap[resourceSchema.Id] = resourceSchema
	}

	generateCollectionResourcePages()

	for _, schema := range schemaMap {
		//Add in check to show if collection should be visible and if actions should be shown
		showActions := false

		if _, ok := schema.Resource.Links["showResource"]; ok {
			showActions = true
		}

		if err = generateIndividualDocs(schema, showActions); err != nil {
			return err
		}
	}

	return nil
}

func readInputFiles() error {

	//Read API Description Files for All Resources
	composeBytes, err := ioutil.ReadFile(apiInputDir + "/schema-check/api_description.yml")
	if err != nil {
		return err
	}
	//resourceDescriptionsMap = make(map[string]string)

	if err = yaml.Unmarshal(composeBytes, &descriptionsMap); err != nil {
		return err
	}

	//Read API Description for the Manual Descriptions
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/api_description_override.yml")
	if err != nil {
		return err
	}

	descriptionsOverrideMap := make(map[string]string)

	if err = yaml.Unmarshal(composeBytes, descriptionsOverrideMap); err != nil {
		return err
	}

	for key, desc := range descriptionsOverrideMap {
		if desc != "" {
			descriptionsMap[key] = desc

		}
	}

	//Read API Description Files for Collection Only Resources
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/collection_api_description.yml")
	if err != nil {
		return err
	}
	collectionDescriptionsMap := make(map[string]string)

	if err = yaml.Unmarshal(composeBytes, collectionDescriptionsMap); err != nil {
		return err
	}

	for key, desc := range collectionDescriptionsMap {
		if desc != "" {
			descriptionsMap[key] = desc
		}
	}
	//read yaml file to load the common fields
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/common_fields.yml")
	if err != nil {
		return err
	}

	if err = yaml.Unmarshal(composeBytes, &commonFieldsMap); err != nil {
		return err
	}

	//read yaml file to load the generic description fields
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/generic_descriptions.yml")
	if err != nil {
		return err
	}

	return yaml.Unmarshal(composeBytes, &genericDescriptionsMap)
}

func generateCollectionResourcePages() error {
	output, err := os.Create(path.Join(apiOutputDir, "api-resources", "index.md"))

	if err != nil {
		return err
	}

	defer output.Close()

	data := map[string]interface{}{
		"schemaMap": schemaMap,
		"version":   version,
		"language":  language,
		"layout":    layout,
	}

	funcMap := template.FuncMap{
		"getResourceDescription": getResourceDescription,
		"capitalize":             strings.Title,
	}

	typeTemplate, err := template.New("apiHomePage.template").Funcs(funcMap).ParseFiles("./templates/apiHomePage.template")
	if err != nil {
		return err
	}

	if err = typeTemplate.Execute(output, data); err != nil {
		return err
	}

	output, err = os.Create(path.Join(apiOutputDir, "rancher-api-sidebar.html"))

	if err != nil {
		return err
	}

	defer output.Close()

	typeTemplate, err = template.New("apiNavBar.template").Funcs(funcMap).ParseFiles("./templates/apiNavBar.template")
	if err != nil {
		return err
	}

	return typeTemplate.Execute(output, data)
}

func generateIndividualDocs(schema client.Schema, showActions bool) error {
	if err := setupDirectory(apiOutputDir + "/api-resources/" + schema.Id); err != nil {
		return err
	}

	output, err := os.Create(path.Join(apiOutputDir, "api-resources", schema.Id, "index.md"))

	if err != nil {
		return err
	}

	defer output.Close()

	data := map[string]interface{}{
		"schemaId":            schema.Id,
		"resourceDescription": getResourceDescription(schema.Id),
		"fieldMap":            getFieldMap(schema),
		"operationMap":        getActionMap(schema, true),
		"actionMap":           getActionMap(schema, false),
		"pluralName":          schema.PluralName,
		"version":             version,
		"language":            language,
		"layout":              layout,
	}

	funcMap := template.FuncMap{
		"getResourceDescription": getResourceDescription,
		"capitalize":             strings.Title,
	}

	var templateName string

	if showActions {
		templateName = "apiResource.template"
	} else {
		templateName = "apiActionInput.template"
	}

	typeTemplate, err := template.New(templateName).Funcs(funcMap).ParseFiles("./templates/" + templateName)
	if err != nil {
		return err
	}

	return typeTemplate.Execute(output, data)
}

func getResourceDescription(resourceID string) string {
	var desc string

	if updatedDesc, inDescMap := descriptionsMap[resourceID+"-description"]; inDescMap {
		if updatedDesc != "" {
			return updatedDesc
		}
	}
	return desc
}

func getFieldMap(schema client.Schema) map[string]APIField {
	fieldMap := make(map[string]APIField)

	for fieldName, field := range schema.ResourceFields {
		// Skip any fields that are in the common field list
		if commonFieldsMap[fieldName] {
			continue
		}

		apiField := APIField{}
		apiField.Field = field
		apiField.Description = getFieldDescription(schema.Id, fieldName, field)

		if referenceRegexp.MatchString(field.Type) {
			//put the link to the referenced field in the form
			//[type]({{site.baseurl}}/rancher/{{page.version}}/{{page.lang}}/api/api-resources/type/)
			apiField.TypeURL = getRefTypeURL(field.Type)
		} else if strings.HasSuffix(field.Type, "]") {
			//Update other types that have references to other resources
			apiField.TypeURL = getTypeURL(field.Type)
		} else if _, isResourceType := schemaMap[field.Type]; isResourceType {
			apiField.TypeURL = "[" + field.Type + "]({{site.baseurl}}/rancher/{{page.version}}/{{page.lang}}/api/api-resources/" + field.Type + "/)"
		}

		if field.Default == nil {
			apiField.Default = ""
		}

		fieldMap[fieldName] = apiField
	}

	return fieldMap
}

func getFieldDescription(resourceID string, fieldID string, field client.Field) string {
	var desc string
	//desc := "This is the " + fieldID + " field"

	//If it's a generic Description, translate the <resource> and <options>
	if genDescription, isGenericDescription := genericDescriptionsMap[fieldID]; isGenericDescription {
		desc = descRegexp.ReplaceAllString(genDescription, resourceID)
		desc = optionsRegexp.ReplaceAllString(desc, "["+strings.Join(field.Options, ", ")+"]")
		return desc
	}

	if updatedDesc, inDescMap := descriptionsMap[resourceID+"-resourceField-"+fieldID]; inDescMap {
		if updatedDesc != "" {
			return updatedDesc
		}
	}

	return desc
}

func getRefTypeURL(input string) string {
	return referenceRegexp.ReplaceAllString(input, "[$1]({{site.baseurl}}/rancher/{{page.version}}/{{page.lang}}/api/api-resources/$1/)")
}

func getTypeURL(typeInput string) string {
	var stringSliceByOpenBracket []string
	stringSliceByOpenBracket = strings.SplitAfter(typeInput, "[")

	var resourceName string

	for _, value := range stringSliceByOpenBracket {
		if strings.Contains(value, "]") {
			resourceName = strings.Replace(value, "]", "", -1)
		}
	}

	if _, isResourceType := schemaMap[resourceName]; isResourceType {
		urlResourceName := "[" + resourceName + "]({{site.baseurl}}/rancher/{{page.version}}/{{page.lang}}/api/api-resources/" + resourceName + "/)"
		return strings.Replace(typeInput, resourceName, urlResourceName, -1)
	}
	return typeInput
}

func getActionMap(schema client.Schema, operationsActions bool) map[string]APIAction {
	actionMap := make(map[string]APIAction)

	if operationsActions {
		//Check for create by looking for POST in collectionMethods
		for _, method := range schema.CollectionMethods {
			if method == postAPI {
				//add create
				apiAction := APIAction{}
				apiAction.Description = getActionDescription(schema.Id, "create")
				apiAction.Method = postAPI
				apiAction.ActionURL = "/v1/" + schema.PluralName
				resourceFields := make(map[string]client.Field)

				for fieldName, field := range schema.ResourceFields {
					if field.Create {
						resourceFields[fieldName] = field
					}
				}

				apiAction.Input.InputJSON = generateJSONFromFields(resourceFields)
				actionMap["Create"] = apiAction
			}
		}

		for _, method := range schema.ResourceMethods {
			if method == "PUT" {
				//add update
				apiAction := APIAction{}
				apiAction.Description = getActionDescription(schema.Id, "update")
				apiAction.Method = "PUT"
				apiAction.ActionURL = "/v1/" + schema.PluralName + "/${ID}"
				resourceFields := make(map[string]client.Field)

				for fieldName, field := range schema.ResourceFields {
					if field.Update {
						resourceFields[fieldName] = field
					}
				}

				apiAction.Input.InputJSON = generateJSONFromFields(resourceFields)
				actionMap["Update"] = apiAction
			} else if method == "DELETE" {
				//add delete
				apiAction := APIAction{}
				apiAction.Description = getActionDescription(schema.Id, "delete")
				apiAction.Method = "DELETE"
				apiAction.ActionURL = "/v1/" + schema.PluralName + "/${ID}"
				actionMap["Delete"] = apiAction
			}
		}

	} else {

		for actionName, action := range schema.ResourceActions {
			//Check if general action or resource specific action is blacklisted
			if isBlacklistAction(schema.Id, actionName) {
				continue
			}

			apiAction := APIAction{}
			apiAction.Description = getActionDescription(schema.Id, actionName)
			apiAction.Input = getActionInput(action.Input)
			apiAction.Output = action.Output
			apiAction.Method = postAPI
			apiAction.ActionURL = "/v1/" + schema.PluralName + "/${ID}?action=" + actionName

			actionMap[actionName] = apiAction
		}
	}

	return actionMap
}

func getActionDescription(resourceID string, fieldID string) string {
	var desc string
	//desc := "This is the " + fieldID + " action"

	if updatedDesc, inDescMap := descriptionsMap[resourceID+"-resourceAction-"+fieldID]; inDescMap {
		if updatedDesc != "" {
			return updatedDesc
		}
	}

	return desc
}

func getActionInput(schemaID string) ActionInput {
	actionInput := ActionInput{}
	actionInput.Name = schemaID
	//actionInput.FieldMap = getFieldMap(schemaMap[schemaID])
	actionInput.InputJSON = generateJSONFromFields(schemaMap[schemaID].ResourceFields)

	return actionInput
}

func generateJSONFromFields(resourceFields map[string]client.Field) string {
	j, err := json.MarshalIndent(generateFieldTypeMap(resourceFields), "", "\t")

	if err != nil {
		return err.Error()
	}
	return strings.Replace(string(j), "&#34;", "", -1)

}

func generateFieldTypeMap(resourceFields map[string]client.Field) map[string]interface{} {
	fieldTypeJSONMap := make(map[string]interface{})
	for fieldName, field := range resourceFields {
		fieldTypeJSONMap[fieldName] = generateTypeValue(field)
	}
	return fieldTypeJSONMap
}

func generateTypeValue(field client.Field) interface{} {
	//get default value if available
	if field.Default != nil {
		return field.Default
	}

	//basic types
	switch field.Type {
	case "string":
		return "string"
	case "int":
		return 0
	case "boolean":
		return true
	case "array[string]":
		return [...]string{"string1", "string2", "...stringN"}
	case "map[string]":
		return map[string]string{"key1": "value1", "key2": "value2", "keyN": "valueN"}
	case "password":
		return field.Type
	}

	//another resourceType
	subSchema, ok := schemaMap[field.Type]
	if ok {
		return generateFieldTypeMap(subSchema.ResourceFields)
	}

	return field.Type
}
