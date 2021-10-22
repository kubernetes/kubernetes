// +build example

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
)

// Prints a list of instances for each region. If no regions are provided
// all regions will be searched. The state is required.
//
// Will use the AWS SDK for Go's default credential chain and region. You can
// specify the region with the AWS_REGION environment variable.
//
// Usage: instancesByRegion -state <value> [-state val...] [-region region...]
func main() {
	states, regions := parseArguments()

	if len(states) == 0 {
		fmt.Fprintf(os.Stderr, "error: %v\n", usage())
		os.Exit(1)
	}
	instanceCriteria := " "
	for _, state := range states {
		instanceCriteria += "[" + state + "]"
	}

	if len(regions) == 0 {
		var err error
		regions, err = fetchRegion()
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}

	for _, region := range regions {
		sess := session.Must(session.NewSession(&aws.Config{
			Region: aws.String(region),
		}))

		ec2Svc := ec2.New(sess)
		params := &ec2.DescribeInstancesInput{
			Filters: []*ec2.Filter{
				{
					Name:   aws.String("instance-state-name"),
					Values: aws.StringSlice(states),
				},
			},
		}

		result, err := ec2Svc.DescribeInstances(params)
		if err != nil {
			fmt.Println("Error", err)
		} else {
			fmt.Printf("\n\n\nFetching instance details for region: %s with criteria: %s**\n ", region, instanceCriteria)
			if len(result.Reservations) == 0 {
				fmt.Printf("There is no instance for the region: %s with the matching criteria:%s  \n", region, instanceCriteria)
			}
			for _, reservation := range result.Reservations {

				fmt.Println("printing instance details.....")
				for _, instance := range reservation.Instances {
					fmt.Println("instance id " + *instance.InstanceId)
					fmt.Println("current State " + *instance.State.Name)
				}
			}
			fmt.Printf("done for region %s **** \n", region)
		}
	}
}

func fetchRegion() ([]string, error) {
	awsSession := session.Must(session.NewSession(&aws.Config{}))

	svc := ec2.New(awsSession)
	awsRegions, err := svc.DescribeRegions(&ec2.DescribeRegionsInput{})
	if err != nil {
		return nil, err
	}

	regions := make([]string, 0, len(awsRegions.Regions))
	for _, region := range awsRegions.Regions {
		regions = append(regions, *region.RegionName)
	}

	return regions, nil
}

type flagArgs []string

func (a flagArgs) String() string {
	return strings.Join(a.Args(), ",")
}

func (a *flagArgs) Set(value string) error {
	*a = append(*a, value)
	return nil
}
func (a flagArgs) Args() []string {
	return []string(a)
}

func parseArguments() (states []string, regions []string) {
	var stateArgs, regionArgs flagArgs

	flag.Var(&stateArgs, "state", "state list")
	flag.Var(&regionArgs, "region", "region list")
	flag.Parse()

	if flag.NFlag() != 0 {
		states = append([]string{}, stateArgs.Args()...)
		regions = append([]string{}, regionArgs.Args()...)
	}

	return states, regions
}

func usage() string {
	return `

Missing mandatory flag 'state'. Please use like below  Example:

To fetch the stopped instance of all region use below:
	./filter_ec2_by_region -state running -state stopped

To fetch the stopped and running instance  for  region us-west-1 and eu-west-1 use below:
	./filter_ec2_by_region -state running -state stopped -region us-west-1 -region=eu-west-1
`
}
