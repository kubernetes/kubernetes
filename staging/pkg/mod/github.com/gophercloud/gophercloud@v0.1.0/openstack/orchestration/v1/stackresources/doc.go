/*
Package stackresources provides operations for working with stack resources.
A resource is a template artifact that represents some component of your
desired architecture (a Cloud Server, a group of scaled Cloud Servers, a load
balancer, some configuration management system, and so forth).

Example of get resource information in stack

    rsrc_result := stackresources.Get(client, stack.Name, stack.ID, rsrc.Name)
    if rsrc_result.Err != nil {
        panic(rsrc_result.Err)
    }
    rsrc, err := rsrc_result.Extract()
    if err != nil {
        panic(err)
    }

Example for list stack resources

    all_stack_rsrc_pages, err := stackresources.List(client, stack.Name, stack.ID, nil).AllPages()
    if err != nil {
        panic(err)
    }

    all_stack_rsrcs, err := stackresources.ExtractResources(all_stack_rsrc_pages)
    if err != nil {
        panic(err)
    }

    fmt.Println("Resource List:")
    for _, rsrc := range all_stack_rsrcs {
        // Get information of a resource in stack
        rsrc_result := stackresources.Get(client, stack.Name, stack.ID, rsrc.Name)
        if rsrc_result.Err != nil {
            panic(rsrc_result.Err)
        }
        rsrc, err := rsrc_result.Extract()
        if err != nil {
            panic(err)
        }
        fmt.Println("Resource Name: ", rsrc.Name, ", Physical ID: ", rsrc.PhysicalID, ", Status: ", rsrc.Status)
    }


Example for get resource type schema

    schema_result := stackresources.Schema(client, "OS::Heat::Stack")
    if schema_result.Err != nil {
        panic(schema_result.Err)
    }
    schema, err := schema_result.Extract()
    if err != nil {
        panic(err)
    }
    fmt.Println("Schema for resource type OS::Heat::Stack")
    fmt.Println(schema.SupportStatus)

Example for get resource type Template

    tmp_result := stackresources.Template(client, "OS::Heat::Stack")
    if tmp_result.Err != nil {
        panic(tmp_result.Err)
    }
    tmp, err := tmp_result.Extract()
    if err != nil {
        panic(err)
    }
    fmt.Println("Template for resource type OS::Heat::Stack")
    fmt.Println(string(tmp))
*/
package stackresources
