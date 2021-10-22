/*
Package nodes provides information and interaction with the nodes through
the OpenStack Clustering service.

Example to Create a Node

	opts := nodes.CreateOpts{
		ClusterID: "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
		Metadata:  map[string]interface{}{},
		Name:      "node-e395be1e-002",
		ProfileID: "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
		Role:      "",
	}

	node, err := nodes.Create(serviceClient, opts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("node", node)

Example to List Nodes

	listOpts := nodes.ListOpts{
		Name: "testnode",
	}

	allPages, err := nodes.List(serviceClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allNodes, err := nodes.ExtractNodes(allPages)
	if err != nil {
		panic(err)
	}

	for _, node := range allNodes {
		fmt.Printf("%+v\n", node)
	}

Example to Update a Node

	opts := nodes.UpdateOpts{
		Name: "new-node-name",
	}

	nodeID := "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1"
	node, err := nodes.Update(serviceClient, nodeID, opts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", node)

Example to Delete a Node

	nodeID := "6dc6d336e3fc4c0a951b5698cd1236ee"
	err := nodes.Delete(serviceClient, nodeID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Get a Node

	nodeID := "node123"
	node, err := nodes.Get(serviceClient, nodeID).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", node)

Example to Perform an Operation on a Node

	serviceClient.Microversion = "1.4"
	nodeID := "node123"
	operationOpts := nodes.OperationOpts{
		Operation: nodes.RebootOperation,
		Params:    nodes.OperationParams{"type": "SOFT"},
	}
	actionID, err := nodes.Ops(serviceClient, nodeID, operationOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Recover a Node

	nodeID := "b7b870e3-d3c5-4a93-b9d7-846c53b2c2da"
	check := true
	recoverOpts := nodes.RecoverOpts{
		Operation:     nodes.RebuildRecovery,
		Check:         &check,
	}
	actionID, err := nodes.Recover(computeClient, nodeID, recoverOpts).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Println("action=", actionID)

Example to Check a Node

	nodeID := "b7b870e3-d3c5-4a93-b9d7-846c53b2c2da"
	actionID, err := nodes.Check(serviceClient, nodeID).Extract()
	if err != nil {
		panic(err)
	}
	fmt.Println("action=", actionID)

*/
package nodes
