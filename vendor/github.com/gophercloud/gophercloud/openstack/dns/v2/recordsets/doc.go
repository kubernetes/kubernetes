/*
Package recordsets provides information and interaction with the zone API
resource for the OpenStack DNS service.

Example to List RecordSets by Zone

	listOpts := recordsets.ListOpts{
		Type: "A",
	}

	zoneID := "fff121f5-c506-410a-a69e-2d73ef9cbdbd"

	allPages, err := recordsets.ListByZone(dnsClient, zoneID, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allRRs, err := recordsets.ExtractRecordSets(allPages()
	if err != nil {
		panic(err)
	}

	for _, rr := range allRRs {
		fmt.Printf("%+v\n", rr)
	}

Example to Create a RecordSet

	createOpts := recordsets.CreateOpts{
		Name:        "example.com.",
		Type:        "A",
		TTL:         3600,
		Description: "This is a recordset.",
		Records:     []string{"10.1.0.2"},
	}

	zoneID := "fff121f5-c506-410a-a69e-2d73ef9cbdbd"

	rr, err := recordsets.Create(dnsClient, zoneID, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a RecordSet

	zoneID := "fff121f5-c506-410a-a69e-2d73ef9cbdbd"
	recordsetID := "d96ed01a-b439-4eb8-9b90-7a9f71017f7b"

	err := recordsets.Delete(dnsClient, zoneID, recordsetID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package recordsets
