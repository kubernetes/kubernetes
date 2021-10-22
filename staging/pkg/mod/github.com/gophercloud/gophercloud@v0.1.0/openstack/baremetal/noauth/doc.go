/*
Package noauth provides support for noauth bare metal endpoints.

Example of obtaining and using a client:

	client, err := noauth.NewBareMetalNoAuth(noauth.EndpointOpts{
		IronicEndpoint: "http://localhost:6385/v1/",
	})
	if err != nil {
		panic(err)
	}

	client.Microversion = "1.50"

	nodes.ListDetail(client, nodes.ListOpts{})
*/
package noauth
