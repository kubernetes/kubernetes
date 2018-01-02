package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	yaml "gopkg.in/yaml.v2"

	"github.com/rancher/go-rancher/client"
)

const (
	apiOutputDir = "./output"
	apiInputDir  = "./input"
	postAPI      = "POST"
)

var (
	blacklistTypes               map[string]bool
	blacklistCollectionResources map[string]bool
	blacklistActions             map[string]bool
	blacklistResourceActions     map[string]bool
	resourceDescriptionsMap      map[string]string
)

func init() {
	blacklistTypes = make(map[string]bool)
	blacklistTypes["schema"] = true
	blacklistTypes["resource"] = true
	blacklistTypes["collection"] = true
	blacklistCollectionResources = make(map[string]bool)
	resourceDescriptionsMap = make(map[string]string)

}

func readCattleSchema() (client.Schemas, error) {

	var schemas client.Schemas

	schemaBytes, err := ioutil.ReadFile(apiInputDir + "/schemas.json")
	if err != nil {
		return schemas, err
	}

	if err = json.Unmarshal(schemaBytes, &schemas); err != nil {
		return schemas, err
	}

	return schemas, nil
}

func readBlacklistFiles() error {

	//Create list of blacklist collections (collections that will not show up on navbar or have actions)
	composeBytes, err := ioutil.ReadFile(apiInputDir + "/schema-check/blacklist_collections.yml")
	if err != nil {
		return err
	}
	if err = yaml.Unmarshal(composeBytes, &blacklistCollectionResources); err != nil {
		return err
	}

	//Create list of blacklist actions (list of ALL actions to be hidden for ALL resources)
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/schema-check//blacklist_actions.yml")
	if err != nil {
		return err
	}
	if err = yaml.Unmarshal(composeBytes, &blacklistActions); err != nil {
		return err
	}

	//Create list of blacklist actions specific to a resource (list of actions to be hidden for certain resources)
	composeBytes, err = ioutil.ReadFile(apiInputDir + "/schema-check//blacklist_resource_actions.yml")
	if err != nil {
		return err
	}
	if err = yaml.Unmarshal(composeBytes, &blacklistResourceActions); err != nil {
		return err
	}

	return nil
}

func isBlacklistCollection(resourceName string) bool {
	if blacklistCollectionResources[resourceName] {
		return true
	}
	return false
}
func isBlacklistAction(resourceName string, actionName string) bool {
	if blacklistActions[actionName] || blacklistResourceActions[resourceName+"-"+actionName] {
		return true
	}
	return false
}

func readGenDescFile(structure map[string]string) error {
	//read yaml file to load the common desc
	composeBytes, err := ioutil.ReadFile("./input/generic_descriptions.yml")
	if err != nil {
		return err
	}

	return yaml.Unmarshal(composeBytes, &structure)
}

func setupDirectory(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return os.MkdirAll(dir, 0755)
	}

	return nil
}
