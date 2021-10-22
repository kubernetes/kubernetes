/*
Package agents provides the ability to retrieve and manage Agents through the Neutron API.

Example of Listing Agents

    listOpts := agents.ListOpts{
        AgentType: "Open vSwitch agent",
    }

    allPages, err := agents.List(networkClient, listOpts).AllPages()
    if err != nil {
        panic(err)
    }

    allAgents, err := agents.ExtractAgents(allPages)
    if err != nil {
        panic(err)
    }

    for _, agent := range allAgents {
        fmt.Printf("%+v\n", agent)
    }

Example to Get an Agent

    agentID = "76af7b1f-d61b-4526-94f7-d2e14e2698df"
    agent, err := agents.Get(networkClient, agentID).Extract()
    if err != nil {
        panic(err)
    }
*/
package agents
