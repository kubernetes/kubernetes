/*
Package crontriggers provides interaction with the cron triggers API in the OpenStack Mistral service.

Cron trigger is an object that allows to run Mistral workflows according to a time pattern (Unix crontab patterns format).
Once a trigger is created it will run a specified workflow according to its properties: pattern, first_execution_time and remaining_executions.

List cron triggers

To filter cron triggers from a list request, you can use advanced filters with special FilterType to check for equality, non equality, values greater or lower, etc.
Default Filter checks equality, but you can override it with provided filter type.

	listOpts := crontriggers.ListOpts{
		WorkflowName: &executions.ListFilter{
			Value: "Workflow1,Workflow2",
			Filter: executions.FilterIN,
		},
		CreatedAt: &executions.ListDateFilter{
			Value: time.Date(2018, time.January, 1, 0, 0, 0, 0, time.UTC),
			Filter: executions.FilterGTE,
		},
	}

	allPages, err := crontriggers.List(mistralClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allCrontriggers, err := crontriggers.ExtractCronTriggers(allPages)
	if err != nil {
		panic(err)
	}

	for _, ct := range allCrontriggers {
		fmt.Printf("%+v\n", ct)
	}

Create a cron trigger. This example will start the workflow "echo" each day at 8am, and it will end after 10 executions.

	createOpts := &crontriggers.CreateOpts{
		Name:                "daily",
		Pattern:             "0 8 * * *",
		WorkflowName:        "echo",
		RemainingExecutions: 10,
		WorkflowParams: map[string]interface{}{
			"msg": "hello",
		},
		WorkflowInput: map[string]interface{}{
			"msg": "world",
		},
	}
	crontrigger, err := crontriggers.Create(mistralClient, opts).Extract()
	if err != nil {
		panic(err)
	}

Get a cron trigger

	crontrigger, err := crontriggers.Get(mistralClient, "0520ffd8-f7f1-4f2e-845b-55d953a1cf46").Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf(%+v\n", crontrigger)

Delete a cron trigger

	res := crontriggers.Delete(mistralClient, "0520ffd8-f7f1-4f2e-845b-55d953a1cf46")
	if res.Err != nil {
		panic(res.Err)
	}

*/
package crontriggers
