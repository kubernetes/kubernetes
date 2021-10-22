// +build example

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go/aws/endpoints"
)

// Demostrates how the SDK's endpoints can be enumerated over to discover
// regions, services, and endpoints defined by the SDK's Regions and Endpoints
// metadata.
//
// Usage:
//  -p=id partition id, e.g: aws
//  -r=id region id, e.g: us-west-2
//  -s=id service id, e.g: s3
//
//  -partitions Lists all partitions.
//  -regions Lists all regions in a partition. Requires partition ID.
//           If service ID is also provided will show endpoints for a service.
//  -services Lists all services in a partition. Requires partition ID.
//            If region ID is also provided, will show services available in that region.
//
// Example:
//   go run enumEndpoints.go -p aws -services -r us-west-2
//
// Output:
//   Services with endpoint us-west-2 in aws:
//   ...
func main() {
	var partitionID, regionID, serviceID string
	flag.StringVar(&partitionID, "p", "", "Partition ID")
	flag.StringVar(&regionID, "r", "", "Region ID")
	flag.StringVar(&serviceID, "s", "", "Service ID")

	var cmdPartitions, cmdRegions, cmdServices bool
	flag.BoolVar(&cmdPartitions, "partitions", false, "Lists partitions.")
	flag.BoolVar(&cmdRegions, "regions", false, "Lists regions of a partition. Requires partition ID to be provided. Will filter by a service if '-s' is set.")
	flag.BoolVar(&cmdServices, "services", false, "Lists services for a partition. Requires partition ID to be provided. Will filter by a region if '-r' is set.")
	flag.Parse()

	partitions := endpoints.DefaultResolver().(endpoints.EnumPartitions).Partitions()

	if cmdPartitions {
		printPartitions(partitions)
	}

	if !(cmdRegions || cmdServices) {
		return
	}

	p, ok := findPartition(partitions, partitionID)
	if !ok {
		fmt.Fprintf(os.Stderr, "Partition %q not found", partitionID)
		os.Exit(1)
	}

	if cmdRegions {
		printRegions(p, serviceID)
	}

	if cmdServices {
		printServices(p, regionID)
	}
}

func printPartitions(ps []endpoints.Partition) {
	fmt.Println("Partitions:")
	for _, p := range ps {
		fmt.Println(p.ID())
	}
}

func printRegions(p endpoints.Partition, serviceID string) {
	if len(serviceID) != 0 {
		s, ok := p.Services()[serviceID]
		if !ok {
			fmt.Fprintf(os.Stderr, "service %q does not exist in partition %q", serviceID, p.ID())
			os.Exit(1)
		}
		es := s.Endpoints()
		fmt.Printf("Endpoints for %s in %s:\n", serviceID, p.ID())
		for _, e := range es {
			r, _ := e.ResolveEndpoint()
			fmt.Printf("%s: %s\n", e.ID(), r.URL)
		}

	} else {
		rs := p.Regions()
		fmt.Printf("Regions in %s:\n", p.ID())
		for _, r := range rs {
			fmt.Println(r.ID())
		}
	}
}

func printServices(p endpoints.Partition, endpointID string) {
	ss := p.Services()

	if len(endpointID) > 0 {
		fmt.Printf("Services with endpoint %s in %s:\n", endpointID, p.ID())
	} else {
		fmt.Printf("Services in %s:\n", p.ID())
	}

	for id, s := range ss {
		if _, ok := s.Endpoints()[endpointID]; !ok && len(endpointID) > 0 {
			continue
		}
		fmt.Println(id)
	}
}

func findPartition(ps []endpoints.Partition, partitionID string) (endpoints.Partition, bool) {
	for _, p := range ps {
		if p.ID() == partitionID {
			return p, true
		}
	}

	return endpoints.Partition{}, false
}
