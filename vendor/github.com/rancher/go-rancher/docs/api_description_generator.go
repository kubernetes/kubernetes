package main

import (
	"os"
	"path"
	"regexp"
	"strings"
	"text/template"
)

var (
	descRegexp    *regexp.Regexp = regexp.MustCompile(`<resource>`)
	optionsRegexp *regexp.Regexp = regexp.MustCompile(`<list of options>`)
)

func generateDescriptionFile(emptyDesc bool, collectionOnly bool) error {
	schemas, err := readCattleSchema()

	if err != nil {
		return err
	}

	if err = readBlacklistFiles(); err != nil {
		return err
	}

	genericDescMap := make(map[string]string)
	err = readGenDescFile(genericDescMap)
	if err != nil {
		return err
	}

	for _, resourceSchema := range schemas.Data {
		//Check if it's an invalid Resource Type
		if blacklistTypes[resourceSchema.Id] {
			continue
		}

		//Only print out collection links for the collection yml
		if collectionOnly {
			// If it's not a collection or it's a blacklist Collection, skip
			if _, isCollection := resourceSchema.Links["collection"]; !isCollection || isBlacklistCollection(resourceSchema.Id) {
				continue
			}
		} else {
			//Only add in actions and fields for non-collection only files
			for actionName := range resourceSchema.ResourceActions {
				if !isBlacklistAction(resourceSchema.Id, actionName) {
					if emptyDesc {
						resourceDescriptionsMap[resourceSchema.Id+"-resourceActions-"+actionName] = ""
						//resourceDescriptionsMap[resourceSchema.Id+"-"+actionName] = ""
					} else {
						resourceDescriptionsMap[resourceSchema.Id+"-resourceActions-"+actionName] = "To " + actionName + " the " + resourceSchema.Id
					}
				}
			}

			for fieldName, field := range resourceSchema.ResourceFields {
				if emptyDesc {
					resourceDescriptionsMap[resourceSchema.Id+"-resourceFields-"+fieldName] = ""
					//resourceDescriptionsMap[resourceSchema.Id+"-"+fieldName] = ""
				} else {
					//check if a generic desc exists
					var description string
					if genericDesc, ok := genericDescMap[fieldName]; ok {
						description = descRegexp.ReplaceAllString(genericDesc, resourceSchema.Id)
						description = optionsRegexp.ReplaceAllString(description, "["+strings.Join(field.Options, ", ")+"]")
					} /*else {
						//description = "The " + fieldName + " for the " + schema.Id
					}*/
					resourceDescriptionsMap[resourceSchema.Id+"-resourceFields-"+fieldName] = description
				}
			}
		}
		resourceDescriptionsMap[resourceSchema.Id+"-description"] = ""
	}

	if err = setupDirectory(apiOutputDir); err != nil {
		return err
	}

	var filePrefix string
	if collectionOnly {
		filePrefix = "blank_collection_"
	} else if emptyDesc {
		filePrefix = "blank_"
	}

	output, err := os.Create(path.Join(apiInputDir, "/schema-check/"+filePrefix+"api_description.yml"))
	if err != nil {
		return err
	}

	defer output.Close()

	data := map[string]interface{}{
		"descriptionMap": resourceDescriptionsMap,
	}

	typeTemplate, err := template.New("apiDescription.template").ParseFiles("./templates/apiDescription.template")
	if err != nil {
		return err
	}

	return typeTemplate.Execute(output, data)
}
