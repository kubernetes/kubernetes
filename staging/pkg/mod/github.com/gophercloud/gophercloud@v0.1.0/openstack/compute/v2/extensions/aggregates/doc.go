/*
Package aggregates manages information about the host aggregates in the
OpenStack cloud.

Example of Create Aggregate

	opts := aggregates.CreateOpts{
		Name:             "name",
		AvailabilityZone: "london",
	}

	aggregate, err := aggregates.Create(computeClient, opts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

Example of Show Aggregate Details

	aggregateID := 42
	aggregate, err := aggregates.Get(computeClient, aggregateID).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

Example of Delete Aggregate

	aggregateID := 32
	err := aggregates.Delete(computeClient, aggregateID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example of Update Aggregate

	aggregateID := 42
	opts := aggregates.UpdateOpts{
		Name:             "new_name",
		AvailabilityZone: "nova2",
	}

	aggregate, err := aggregates.Update(computeClient, aggregateID, opts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

Example of Retrieving list of all aggregates

	allPages, err := aggregates.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	allAggregates, err := aggregates.ExtractAggregates(allPages)
	if err != nil {
		panic(err)
	}

	for _, aggregate := range allAggregates {
		fmt.Printf("%+v\n", aggregate)
	}

Example of Add Host

	aggregateID := 22
	opts := aggregates.AddHostOpts{
		Host: "newhost-cmp1",
	}

	aggregate, err := aggregates.AddHost(computeClient, aggregateID, opts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

Example of Remove Host

	aggregateID := 22
	opts := aggregates.RemoveHostOpts{
		Host: "newhost-cmp1",
	}

	aggregate, err := aggregates.RemoveHost(computeClient, aggregateID, opts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

Example of Create or Update Metadata

	aggregateID := 22
	opts := aggregates.SetMetadata{
		Metadata: map[string]string{"key": "value"},
	}

	aggregate, err := aggregates.SetMetadata(computeClient, aggregateID, opts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", aggregate)

*/
package aggregates
