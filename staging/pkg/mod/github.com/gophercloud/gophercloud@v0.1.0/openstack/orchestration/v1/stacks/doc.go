/*
Package stacks provides operation for working with Heat stacks. A stack is a
group of resources (servers, load balancers, databases, and so forth)
combined to fulfill a useful purpose. Based on a template, Heat orchestration
engine creates an instantiated set of resources (a stack) to run the
application framework or component specified (in the template). A stack is a
running instance of a template. The result of creating a stack is a deployment
of the application framework or component.

Prepare required import packages

import (
  "fmt"
  "github.com/gophercloud/gophercloud"
  "github.com/gophercloud/gophercloud/openstack"
  "github.com/gophercloud/gophercloud/openstack/orchestration/v1/stacks"
)

Example of Preparing Orchestration client:

    client, err := openstack.NewOrchestrationV1(provider,  gophercloud.EndpointOpts{Region: "RegionOne"})

Example of List Stack:
    all_stack_pages, err := stacks.List(client, nil).AllPages()
    if err != nil {
        panic(err)
    }

    all_stacks, err := stacks.ExtractStacks(all_stack_pages)
    if err != nil {
        panic(err)
    }

    for _, stack := range all_stacks {
        fmt.Printf("%+v\n", stack)
    }


Example to Create an Stack

    // Create Template
    t := make(map[string]interface{})
    f, err := ioutil.ReadFile("template.yaml")
    if err != nil {
        panic(err)
    }
    err = yaml.Unmarshal(f, t)
    if err != nil {
        panic(err)
    }

    template := &stacks.Template{}
    template.TE = stacks.TE{
        Bin: f,
    }
    // Create Environment if needed
    t_env := make(map[string]interface{})
    f_env, err := ioutil.ReadFile("env.yaml")
    if err != nil {
        panic(err)
    }
    err = yaml.Unmarshal(f_env, t_env)
    if err != nil {
        panic(err)
    }

    env := &stacks.Environment{}
    env.TE = stacks.TE{
        Bin: f_env,
    }

    // Remember, the priority of parameters you given through
    // Parameters is higher than the parameters you provided in EnvironmentOpts.
    params := make(map[string]string)
    params["number_of_nodes"] = 1
    tags := []string{"example-stack"}
    createOpts := &stacks.CreateOpts{
        // The name of the stack. It must start with an alphabetic character.
        Name:       "testing_group",
        // A structure that contains either the template file or url. Call the
        // associated methods to extract the information relevant to send in a create request.
        TemplateOpts: template,
        // A structure that contains details for the environment of the stack.
        EnvironmentOpts: env,
        // User-defined parameters to pass to the template.
        Parameters: params,
        // A list of tags to assosciate with the Stack
        Tags: tags,
    }

    r := stacks.Create(client, createOpts)
    //dcreated_stack := stacks.CreatedStack()
    if r.Err != nil {
        panic(r.Err)
    }
    created_stack, err := r.Extract()
    if err != nil {
        panic(err)
    }
    fmt.Printf("Created Stack: %v", created_stack.ID)

Example for Get Stack

    get_result := stacks.Get(client, stackName, created_stack.ID)
    if get_result.Err != nil {
        panic(get_result.Err)
    }
    stack, err := get_result.Extract()
    if err != nil {
        panic(err)
    }
    fmt.Println("Get Stack: Name: ", stack.Name, ", ID: ", stack.ID, ", Status: ", stack.Status)

Example for Find Stack

	find_result  := stacks.Find(client, stackIdentity)
	if find_result.Err != nil {
		panic(find_result.Err)
	}
	stack, err := find_result.Extract()
	if err != nil {
		panic(err)
	}
	fmt.Println("Find Stack: Name: ", stack.Name, ", ID: ", stack.ID, ", Status: ", stack.Status)

Example for Delete Stack

    del_r := stacks.Delete(client, stackName, created_stack.ID)
    if del_r.Err != nil {
        panic(del_r.Err)
    }
    fmt.Println("Deleted Stack: ", stackName)

Summary of  Behavior Between Stack Update and UpdatePatch Methods :

Function | Test Case | Result

Update()	| Template AND Parameters WITH Conflict | Parameter takes priority, parameters are set in raw_template.environment overlay
Update()	| Template ONLY | Template updates, raw_template.environment overlay is removed
Update()	| Parameters ONLY | No update, template is required

UpdatePatch() 	| Template AND Parameters WITH Conflict | Parameter takes priority, parameters are set in raw_template.environment overlay
UpdatePatch() 	| Template ONLY | Template updates, but raw_template.environment overlay is not removed, existing parameter values will remain
UpdatePatch() 	| Parameters ONLY | Parameters (raw_template.environment) is updated, excluded values are unchanged

The PUT Update() function will remove parameters from the raw_template.environment overlay
if they are excluded from the operation, whereas PATCH Update() will never be destructive to the
raw_template.environment overlay.  It is not possible to expose the raw_template values with a
patch update once they have been added to the environment overlay with the PATCH verb, but
newly added values that do not have a corresponding key in the overlay will display the
raw_template value.

Example to Update a Stack Using the Update (PUT) Method

	t := make(map[string]interface{})
	f, err := ioutil.ReadFile("template.yaml")
	if err != nil {
		panic(err)
	}
	err = yaml.Unmarshal(f, t)
	if err != nil {
		panic(err)
	}

	template := stacks.Template{}
	template.TE = stacks.TE{
		Bin: f,
	}

	var params = make(map[string]interface{})
	params["number_of_nodes"] = 2

	stackName := "my_stack"
	stackId := "d68cc349-ccc5-4b44-a17d-07f068c01e5a"

	stackOpts := &stacks.UpdateOpts{
		Parameters: params,
		TemplateOpts: &template,
	}

	res := stacks.Update(orchestrationClient, stackName, stackId, stackOpts)
	if res.Err != nil {
		panic(res.Err)
	}

Example to Update a Stack Using the UpdatePatch (PATCH) Method

	var params = make(map[string]interface{})
	params["number_of_nodes"] = 2

	stackName := "my_stack"
	stackId := "d68cc349-ccc5-4b44-a17d-07f068c01e5a"

	stackOpts := &stacks.UpdateOpts{
		Parameters: params,
	}

	res := stacks.UpdatePatch(orchestrationClient, stackName, stackId, stackOpts)
	if res.Err != nil {
		panic(res.Err)
	}

Example YAML Template Containing a Heat::ResourceGroup With Three Nodes

	heat_template_version: 2016-04-08

	parameters:
		number_of_nodes:
			type: number
			default: 3
			description: the number of nodes
		node_flavor:
			type: string
			default: m1.small
			description: node flavor
		node_image:
			type: string
			default: centos7.5-latest
			description: node os image
		node_network:
			type: string
			default: my-node-network
			description: node network name

	resources:
		resource_group:
			type: OS::Heat::ResourceGroup
			properties:
			count: { get_param: number_of_nodes }
			resource_def:
				type: OS::Nova::Server
				properties:
					name: my_nova_server_%index%
					image: { get_param: node_image }
					flavor: { get_param: node_flavor }
					networks:
						- network: {get_param: node_network}
*/
package stacks
