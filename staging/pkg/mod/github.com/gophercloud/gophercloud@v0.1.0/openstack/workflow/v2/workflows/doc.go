/*
Package workflows provides interaction with the workflows API in the OpenStack Mistral service.

Workflow represents a process that can be described in a various number of ways and that can do some job interesting to the end user.
Each workflow consists of tasks (at least one) describing what exact steps should be made during workflow execution.

Workflow definition is written in Mistral Workflow Language v2. You can find all specification here: https://docs.openstack.org/mistral/latest/user/wf_lang_v2.html

List workflows

	listOpts := workflows.ListOpts{
		Namespace: "some-namespace",
	}

	allPages, err := workflows.List(mistralClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allWorkflows, err := workflows.ExtractWorkflows(allPages)
	if err != nil {
		panic(err)
	}

	for _, workflow := range allWorkflows {
		fmt.Printf("%+v\n", workflow)
	}

Get a workflow

	workflow, err := workflows.Get(mistralClient, "604a3a1e-94e3-4066-a34a-aa56873ef236").Extract()
	if err != nil {
		t.Fatalf("Unable to get workflow %s: %v", id, err)
	}

	fmt.Printf("%+v\n", workflow)

Create a workflow

	workflowDefinition := `---
      version: '2.0'

      workflow_echo:
        description: Simple workflow example
        type: direct
        input:
          - msg

        tasks:
          test:
            action: std.echo output="<% $.msg %>"`

	createOpts := &workflows.CreateOpts{
		Definition: strings.NewReader(workflowDefinition),
		Scope: "private",
		Namespace: "some-namespace",
	}

	workflow, err := workflows.Create(mistralClient, opts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", workflow)

Delete a workflow

	res := workflows.Delete(fake.ServiceClient(), "604a3a1e-94e3-4066-a34a-aa56873ef236")
	if res.Err != nil {
		panic(res.Err)
	}
*/
package workflows
