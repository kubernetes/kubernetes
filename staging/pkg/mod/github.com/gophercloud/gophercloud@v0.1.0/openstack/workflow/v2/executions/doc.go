/*
Package executions provides interaction with the execution API in the OpenStack Mistral service.

An execution is a one-shot execution of a specific workflow. Each execution contains all information about workflow itself, about execution process, state, input and output data.

An execution represents also the execution of a cron trigger. Each run of a cron trigger will generate an execution.

List executions

To filter executions from a list request, you can use advanced filters with special FilterType to check for equality, non equality, values greater or lower, etc.
Default Filter checks equality, but you can override it with provided filter type.

	// List all executions from a given workflow list with a creation date upper than 2018-01-01 00:00:00
	listOpts := executions.ListOpts{
		WorkflowName: &executions.ListFilter{
			Value: "Workflow1,Workflow2",
			Filter: executions.FilterIN,
		},
		CreatedAt: &executions.ListDateFilter{
			Value: time.Date(2018, time.January, 1, 0, 0, 0, 0, time.UTC),
			Filter: executions.FilterGTE,
		},
	}

	allPages, err := executions.List(mistralClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allExecutions, err := executions.ExtractExecutions(allPages)
	if err != nil {
		panic(err)
	}

	for _, ex := range allExecutions {
		fmt.Printf("%+v\n", ex)
	}

Create an execution

	createOpts := &executions.CreateOpts{
		WorkflowID:  "6656c143-a009-4bcb-9814-cc100a20bbfa",
		Input: map[string]interface{}{
			"msg": "Hello",
		},
		Description: "this is a description",
	}

	execution, err := executions.Create(mistralClient, opts).Extract()
	if err != nil {
		panic(err)
	}

Get an execution

	execution, err := executions.Get(mistralClient, "50bb59f1-eb77-4017-a77f-6d575b002667").Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf(%+v\n", execution)

Delete an execution

	res := executions.Delete(mistralClient, "50bb59f1-eb77-4017-a77f-6d575b002667")
	if res.Err != nil {
		panic(res.Err)
	}

*/
package executions
