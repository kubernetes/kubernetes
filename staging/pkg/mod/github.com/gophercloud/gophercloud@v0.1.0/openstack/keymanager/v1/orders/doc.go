/*
Package orders manages and retrieves orders in the OpenStack Key Manager
Service.

Example to List Orders

	allPages, err := orders.List(client, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allOrders, err := orders.ExtractOrders(allPages)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", allOrders)

Example to Create a Order

	createOpts := orders.CreateOpts{
		Type: orders.KeyOrder,
		Meta: orders.MetaOpts{
			Name:      "order-name",
			Algorithm: "aes",
			BitLength: 256,
			Mode:      "cbc",
		},
	}

	order, err := orders.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", order)

Example to Delete a Order

	err := orders.Delete(client, orderID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package orders
