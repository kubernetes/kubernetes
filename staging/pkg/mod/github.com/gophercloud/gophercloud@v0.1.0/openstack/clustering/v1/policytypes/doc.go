/*
Package policytypes lists all policy types and shows details for a policy type
from the OpenStack Clustering Service.

Example to List Policy Types

    allPages, err := policytypes.List(clusteringClient).AllPages()
    if err != nil {
        panic(err)
    }

    allPolicyTypes, err := actions.ExtractPolicyTypes(allPages)
    if err != nil {
        panic(err)
    }

    for _, policyType := range allPolicyTypes {
        fmt.Printf("%+v\n", policyType)
    }

Example to Get a Policy Type

    policyTypeName := "senlin.policy.affinity-1.0"
    policyTypeDetail, err := policyTypes.Get(clusteringClient, policyTypeName).Extract()
    if err != nil {
        panic(err)
    }

    fmt.Printf("%+v\n", policyTypeDetail)
*/
package policytypes
