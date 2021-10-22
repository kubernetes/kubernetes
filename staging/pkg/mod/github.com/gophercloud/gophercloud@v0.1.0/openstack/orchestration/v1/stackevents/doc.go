/*
Package stackevents provides operations for finding, listing, and retrieving
stack events. Stack events are events that take place on stacks such as
updating and abandoning.

Example for list events for a stack

    pages, err := stackevents.List(client, stack.Name, stack.ID, nil).AllPages()
    if err != nil {
        panic(err)
    }
    events, err := stackevents.ExtractEvents(pages)
    if err != nil {
        panic(err)
    }
    fmt.Println("Get Event List")
    fmt.Println(events)
*/
package stackevents
