/*
Package siteconnections allows management and retrieval of IPSec site connections in the
OpenStack Networking Service.


Example to create an IPSec site connection

createOpts := siteconnections.CreateOpts{
		Name:           "Connection1",
		PSK:            "secret",
		Initiator:      siteconnections.InitiatorBiDirectional,
		AdminStateUp:   gophercloud.Enabled,
		IPSecPolicyID:  "4ab0a72e-64ef-4809-be43-c3f7e0e5239b",
		PeerEPGroupID:  "5f5801b1-b383-4cf0-bf61-9e85d4044b2d",
		IKEPolicyID:    "47a880f9-1da9-468c-b289-219c9eca78f0",
		VPNServiceID:   "692c1ec8-a7cd-44d9-972b-8ed3fe4cc476",
		LocalEPGroupID: "498bb96a-1517-47ea-b1eb-c4a53db46a16",
		PeerAddress:    "172.24.4.233",
		PeerID:         "172.24.4.233",
		MTU:            1500,
	}
	connection, err := siteconnections.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Show the details of a specific IPSec site connection by ID

	conn, err := siteconnections.Get(client, "f2b08c1e-aa81-4668-8ae1-1401bcb0576c").Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a site connection

	connID := "38aee955-6283-4279-b091-8b9c828000ec"
	err := siteconnections.Delete(networkClient, connID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to List site connections

	allPages, err := siteconnections.List(client, nil).AllPages()
	if err != nil {
		panic(err)
	}

	allConnections, err := siteconnections.ExtractConnections(allPages)
	if err != nil {
		panic(err)
	}

Example to Update an IPSec site connection

	description := "updated connection"
	name := "updatedname"
	updateOpts := siteconnections.UpdateOpts{
		Name:        &name,
		Description: &description,
	}
	updatedConnection, err := siteconnections.Update(client, "5c561d9d-eaea-45f6-ae3e-08d1a7080828", updateOpts).Extract()
	if err != nil {
		panic(err)
	}

*/
package siteconnections
