/*
Package stacktemplates provides operations for working with Heat templates.
A Cloud Orchestration template is a portable file, written in a user-readable
language, that describes how a set of resources should be assembled and what
software should be installed in order to produce a working stack. The template
specifies what resources should be used, what attributes can be set, and other
parameters that are critical to the successful, repeatable automation of a
specific application stack.

Example to get stack template

    temp, err := stacktemplates.Get(client, stack.Name, stack.ID).Extract()
    if err != nil {
        panic(err)
    }
    fmt.Println("Get Stack Template for Stack ", stack.Name)
    fmt.Println(string(temp))

Example to validate stack template

    f2, err := ioutil.ReadFile("template.err.yaml")
    if err != nil {
        panic(err)
    }
    fmt.Println(string(f2))
    validateOpts := &stacktemplates.ValidateOpts{
        Template: string(f2),
    }
    validate_result, err := stacktemplates.Validate(client, validateOpts).Extract()
    if err != nil {
        // If validate failed, you will get error message here
        fmt.Println("Validate failed: ", err.Error())
    } else {
        fmt.Println(validate_result.Parameters)
    }

*/
package stacktemplates
