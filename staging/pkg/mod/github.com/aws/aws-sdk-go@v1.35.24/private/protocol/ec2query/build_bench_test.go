// +build bench

package ec2query_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/protocol/ec2query"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func BenchmarkEC2QueryBuild_Complex_ec2AuthorizeSecurityGroupEgress(b *testing.B) {
	params := &ec2.AuthorizeSecurityGroupEgressInput{
		GroupId:  aws.String("String"), // Required
		CidrIp:   aws.String("String"),
		DryRun:   aws.Bool(true),
		FromPort: aws.Int64(1),
		IpPermissions: []*ec2.IpPermission{
			{ // Required
				FromPort:   aws.Int64(1),
				IpProtocol: aws.String("String"),
				IpRanges: []*ec2.IpRange{
					{ // Required
						CidrIp: aws.String("String"),
					},
					// More values...
				},
				PrefixListIds: []*ec2.PrefixListId{
					{ // Required
						PrefixListId: aws.String("String"),
					},
					// More values...
				},
				ToPort: aws.Int64(1),
				UserIdGroupPairs: []*ec2.UserIdGroupPair{
					{ // Required
						GroupId:   aws.String("String"),
						GroupName: aws.String("String"),
						UserId:    aws.String("String"),
					},
					// More values...
				},
			},
			// More values...
		},
		IpProtocol:                 aws.String("String"),
		SourceSecurityGroupName:    aws.String("String"),
		SourceSecurityGroupOwnerId: aws.String("String"),
		ToPort:                     aws.Int64(1),
	}

	benchEC2QueryBuild(b, "AuthorizeSecurityGroupEgress", params)
}

func BenchmarkEC2QueryBuild_Simple_ec2AttachNetworkInterface(b *testing.B) {
	params := &ec2.AttachNetworkInterfaceInput{
		DeviceIndex:        aws.Int64(1),         // Required
		InstanceId:         aws.String("String"), // Required
		NetworkInterfaceId: aws.String("String"), // Required
		DryRun:             aws.Bool(true),
	}

	benchEC2QueryBuild(b, "AttachNetworkInterface", params)
}

func benchEC2QueryBuild(b *testing.B, opName string, params interface{}) {
	svc := awstesting.NewClient()
	svc.ServiceName = "ec2"
	svc.APIVersion = "2015-04-15"

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(&request.Operation{
			Name:       opName,
			HTTPMethod: "POST",
			HTTPPath:   "/",
		}, params, nil)
		ec2query.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}
