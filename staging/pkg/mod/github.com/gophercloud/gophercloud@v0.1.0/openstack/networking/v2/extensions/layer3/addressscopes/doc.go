/*
Package addressscopes provides the ability to retrieve and manage Address scopes through the Neutron API.

Example of Listing Address scopes

    listOpts := addressscopes.ListOpts{
        IPVersion: 6,
    }

    allPages, err := addressscopes.List(networkClient, listOpts).AllPages()
    if err != nil {
        panic(err)
    }

    allAddressScopes, err := addressscopes.ExtractAddressScopes(allPages)
    if err != nil {
        panic(err)
    }

    for _, addressScope := range allAddressScopes {
        fmt.Printf("%+v\n", addressScope)
    }

Example to Get an Address scope

    addressScopeID = "9cc35860-522a-4d35-974d-51d4b011801e"
    addressScope, err := addressscopes.Get(networkClient, addressScopeID).Extract()
    if err != nil {
        panic(err)
    }

Example to Create a new Address scope

    addressScopeOpts := addressscopes.CreateOpts{
        Name: "my_address_scope",
        IPVersion: 6,
    }
    addressScope, err := addressscopes.Create(networkClient, addressScopeOpts).Extract()
    if err != nil {
        panic(err)
    }

Example to Update an Address scope

    addressScopeID = "9cc35860-522a-4d35-974d-51d4b011801e"
    newName := "awesome_name"
    updateOpts := addressscopes.UpdateOpts{
        Name: &newName,
    }

    addressScope, err := addressscopes.Update(networkClient, addressScopeID, updateOpts).Extract()
    if err != nil {
        panic(err)
    }

Example to Delete an Address scope

    addressScopeID = "9cc35860-522a-4d35-974d-51d4b011801e"
    err := addressscopes.Delete(networkClient, addressScopeID).ExtractErr()
    if err != nil {
        panic(err)
    }
*/
package addressscopes
