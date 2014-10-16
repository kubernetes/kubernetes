/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubectl

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type ModifyAction string

const (
	ModifyCreate = ModifyAction("create")
	ModifyUpdate = ModifyAction("update")
	ModifyDelete = ModifyAction("delete")
)

func Modify(w io.Writer, c *client.RESTClient, action ModifyAction, data []byte) error {
	if action != ModifyCreate && action != ModifyUpdate && action != ModifyDelete {
		return fmt.Errorf("Action not recognized")
	}

	// TODO Support multiple API versions.
	version, kind, err := versionAndKind(data)
	if err != nil {
		return err
	}

	if version != apiVersionToUse {
		return fmt.Errorf("Only supporting API version '%s' for now (version '%s' specified)", apiVersionToUse, version)
	}

	obj, err := dataToObject(data)
	if err != nil {
		if err.Error() == "No type '' for version ''" {
			return fmt.Errorf("Object could not be decoded. Make sure it has the Kind field defined.")
		}
		return err
	}

	resource, err := resolveKindToResource(kind)
	if err != nil {
		return err
	}

	var id string
	switch action {
	case "create":
		id, err = doCreate(c, resource, data)
	case "update":
		id, err = doUpdate(c, resource, obj)
	case "delete":
		id, err = doDelete(c, resource, obj)
	}

	if err != nil {
		return err
	}

	fmt.Fprintf(w, "%s\n", id)
	return nil
}

// Creates the object then returns the ID of the newly created object.
func doCreate(c *client.RESTClient, resource string, data []byte) (string, error) {
	obj, err := c.Post().Path(resource).Body(data).Do().Get()
	if err != nil {
		return "", err
	}
	return getIDFromObj(obj)
}

// Creates the object then returns the ID of the newly created object.
func doUpdate(c *client.RESTClient, resource string, obj runtime.Object) (string, error) {
	// Figure out the ID of the object to update by introspecting into the
	// object.
	id, err := getIDFromObj(obj)
	if err != nil {
		return "", fmt.Errorf("ID not retrievable from object for update: %v", err)
	}

	// Get the object from the server to find out its current resource
	// version to prevent race conditions in updating the object.
	serverObj, err := c.Get().Path(resource).Path(id).Do().Get()
	if err != nil {
		return "", fmt.Errorf("Item ID %s does not exist for update: %v", id, err)
	}
	version, err := getResourceVersionFromObj(serverObj)
	if err != nil {
		return "", err
	}

	// Update the object we are trying to send to the server with the
	// correct resource version.
	typeMeta, err := runtime.FindTypeMeta(obj)
	if err != nil {
		return "", err
	}
	typeMeta.SetResourceVersion(version)

	// Convert object with updated resourceVersion to data for PUT.
	data, err := c.Codec.Encode(obj)
	if err != nil {
		return "", err
	}

	// Do the update.
	err = c.Put().Path(resource).Path(id).Body(data).Do().Error()
	fmt.Printf("r: %q, i: %q, d: %s", resource, id, data)
	if err != nil {
		return "", err
	}

	return id, nil
}

func doDelete(c *client.RESTClient, resource string, obj runtime.Object) (string, error) {
	id, err := getIDFromObj(obj)
	if err != nil {
		return "", fmt.Errorf("ID not retrievable from object for update: %v", err)
	}

	err = c.Delete().Path(resource).Path(id).Do().Error()
	if err != nil {
		return "", err
	}

	return id, nil
}

func getIDFromObj(obj runtime.Object) (string, error) {
	typeMeta, err := runtime.FindTypeMeta(obj)
	if err != nil {
		return "", err
	}
	return typeMeta.ID(), nil
}

func getResourceVersionFromObj(obj runtime.Object) (string, error) {
	typeMeta, err := runtime.FindTypeMeta(obj)
	if err != nil {
		return "", err
	}
	return typeMeta.ResourceVersion(), nil
}
