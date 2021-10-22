/*
Package noauth provides support for noauth bare metal introspection endpoints.

Example of obtaining and using a client:

	client, err := noauth.NewBareMetalIntrospectionNoAuth(noauth.EndpointOpts{
		IronicInspectorEndpoint: "http://localhost:5050/v1/",
	})
	if err != nil {
		panic(err)
	}

	introspection.GetIntrospectionStatus(client, "a62b8495-52e2-407b-b3cb-62775d04c2b8")
*/
package noauth
