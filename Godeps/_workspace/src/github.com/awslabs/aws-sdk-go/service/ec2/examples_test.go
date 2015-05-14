package ec2_test

import (
	"bytes"
	"fmt"
	"time"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/aws/awsutil"
	"github.com/awslabs/aws-sdk-go/service/ec2"
)

var _ time.Duration
var _ bytes.Buffer

func ExampleEC2_AcceptVPCPeeringConnection() {
	svc := ec2.New(nil)

	params := &ec2.AcceptVPCPeeringConnectionInput{
		DryRun:                 aws.Boolean(true),
		VPCPeeringConnectionID: aws.String("String"),
	}
	resp, err := svc.AcceptVPCPeeringConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AllocateAddress() {
	svc := ec2.New(nil)

	params := &ec2.AllocateAddressInput{
		Domain: aws.String("DomainType"),
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.AllocateAddress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AssignPrivateIPAddresses() {
	svc := ec2.New(nil)

	params := &ec2.AssignPrivateIPAddressesInput{
		NetworkInterfaceID: aws.String("String"), // Required
		AllowReassignment:  aws.Boolean(true),
		PrivateIPAddresses: []*string{
			aws.String("String"), // Required
			// More values...
		},
		SecondaryPrivateIPAddressCount: aws.Long(1),
	}
	resp, err := svc.AssignPrivateIPAddresses(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AssociateAddress() {
	svc := ec2.New(nil)

	params := &ec2.AssociateAddressInput{
		AllocationID:       aws.String("String"),
		AllowReassociation: aws.Boolean(true),
		DryRun:             aws.Boolean(true),
		InstanceID:         aws.String("String"),
		NetworkInterfaceID: aws.String("String"),
		PrivateIPAddress:   aws.String("String"),
		PublicIP:           aws.String("String"),
	}
	resp, err := svc.AssociateAddress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AssociateDHCPOptions() {
	svc := ec2.New(nil)

	params := &ec2.AssociateDHCPOptionsInput{
		DHCPOptionsID: aws.String("String"), // Required
		VPCID:         aws.String("String"), // Required
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.AssociateDHCPOptions(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AssociateRouteTable() {
	svc := ec2.New(nil)

	params := &ec2.AssociateRouteTableInput{
		RouteTableID: aws.String("String"), // Required
		SubnetID:     aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.AssociateRouteTable(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AttachClassicLinkVPC() {
	svc := ec2.New(nil)

	params := &ec2.AttachClassicLinkVPCInput{
		Groups: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		InstanceID: aws.String("String"), // Required
		VPCID:      aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.AttachClassicLinkVPC(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AttachInternetGateway() {
	svc := ec2.New(nil)

	params := &ec2.AttachInternetGatewayInput{
		InternetGatewayID: aws.String("String"), // Required
		VPCID:             aws.String("String"), // Required
		DryRun:            aws.Boolean(true),
	}
	resp, err := svc.AttachInternetGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AttachNetworkInterface() {
	svc := ec2.New(nil)

	params := &ec2.AttachNetworkInterfaceInput{
		DeviceIndex:        aws.Long(1),          // Required
		InstanceID:         aws.String("String"), // Required
		NetworkInterfaceID: aws.String("String"), // Required
		DryRun:             aws.Boolean(true),
	}
	resp, err := svc.AttachNetworkInterface(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AttachVPNGateway() {
	svc := ec2.New(nil)

	params := &ec2.AttachVPNGatewayInput{
		VPCID:        aws.String("String"), // Required
		VPNGatewayID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.AttachVPNGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AttachVolume() {
	svc := ec2.New(nil)

	params := &ec2.AttachVolumeInput{
		Device:     aws.String("String"), // Required
		InstanceID: aws.String("String"), // Required
		VolumeID:   aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.AttachVolume(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AuthorizeSecurityGroupEgress() {
	svc := ec2.New(nil)

	params := &ec2.AuthorizeSecurityGroupEgressInput{
		GroupID:  aws.String("String"), // Required
		CIDRIP:   aws.String("String"),
		DryRun:   aws.Boolean(true),
		FromPort: aws.Long(1),
		IPPermissions: []*ec2.IPPermission{
			&ec2.IPPermission{ // Required
				FromPort:   aws.Long(1),
				IPProtocol: aws.String("String"),
				IPRanges: []*ec2.IPRange{
					&ec2.IPRange{ // Required
						CIDRIP: aws.String("String"),
					},
					// More values...
				},
				ToPort: aws.Long(1),
				UserIDGroupPairs: []*ec2.UserIDGroupPair{
					&ec2.UserIDGroupPair{ // Required
						GroupID:   aws.String("String"),
						GroupName: aws.String("String"),
						UserID:    aws.String("String"),
					},
					// More values...
				},
			},
			// More values...
		},
		IPProtocol:                 aws.String("String"),
		SourceSecurityGroupName:    aws.String("String"),
		SourceSecurityGroupOwnerID: aws.String("String"),
		ToPort: aws.Long(1),
	}
	resp, err := svc.AuthorizeSecurityGroupEgress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_AuthorizeSecurityGroupIngress() {
	svc := ec2.New(nil)

	params := &ec2.AuthorizeSecurityGroupIngressInput{
		CIDRIP:    aws.String("String"),
		DryRun:    aws.Boolean(true),
		FromPort:  aws.Long(1),
		GroupID:   aws.String("String"),
		GroupName: aws.String("String"),
		IPPermissions: []*ec2.IPPermission{
			&ec2.IPPermission{ // Required
				FromPort:   aws.Long(1),
				IPProtocol: aws.String("String"),
				IPRanges: []*ec2.IPRange{
					&ec2.IPRange{ // Required
						CIDRIP: aws.String("String"),
					},
					// More values...
				},
				ToPort: aws.Long(1),
				UserIDGroupPairs: []*ec2.UserIDGroupPair{
					&ec2.UserIDGroupPair{ // Required
						GroupID:   aws.String("String"),
						GroupName: aws.String("String"),
						UserID:    aws.String("String"),
					},
					// More values...
				},
			},
			// More values...
		},
		IPProtocol:                 aws.String("String"),
		SourceSecurityGroupName:    aws.String("String"),
		SourceSecurityGroupOwnerID: aws.String("String"),
		ToPort: aws.Long(1),
	}
	resp, err := svc.AuthorizeSecurityGroupIngress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_BundleInstance() {
	svc := ec2.New(nil)

	params := &ec2.BundleInstanceInput{
		InstanceID: aws.String("String"), // Required
		Storage: &ec2.Storage{ // Required
			S3: &ec2.S3Storage{
				AWSAccessKeyID:        aws.String("String"),
				Bucket:                aws.String("String"),
				Prefix:                aws.String("String"),
				UploadPolicy:          []byte("PAYLOAD"),
				UploadPolicySignature: aws.String("String"),
			},
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.BundleInstance(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelBundleTask() {
	svc := ec2.New(nil)

	params := &ec2.CancelBundleTaskInput{
		BundleID: aws.String("String"), // Required
		DryRun:   aws.Boolean(true),
	}
	resp, err := svc.CancelBundleTask(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelConversionTask() {
	svc := ec2.New(nil)

	params := &ec2.CancelConversionTaskInput{
		ConversionTaskID: aws.String("String"), // Required
		DryRun:           aws.Boolean(true),
		ReasonMessage:    aws.String("String"),
	}
	resp, err := svc.CancelConversionTask(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelExportTask() {
	svc := ec2.New(nil)

	params := &ec2.CancelExportTaskInput{
		ExportTaskID: aws.String("String"), // Required
	}
	resp, err := svc.CancelExportTask(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelImportTask() {
	svc := ec2.New(nil)

	params := &ec2.CancelImportTaskInput{
		CancelReason: aws.String("String"),
		DryRun:       aws.Boolean(true),
		ImportTaskID: aws.String("String"),
	}
	resp, err := svc.CancelImportTask(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelReservedInstancesListing() {
	svc := ec2.New(nil)

	params := &ec2.CancelReservedInstancesListingInput{
		ReservedInstancesListingID: aws.String("String"), // Required
	}
	resp, err := svc.CancelReservedInstancesListing(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CancelSpotInstanceRequests() {
	svc := ec2.New(nil)

	params := &ec2.CancelSpotInstanceRequestsInput{
		SpotInstanceRequestIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CancelSpotInstanceRequests(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ConfirmProductInstance() {
	svc := ec2.New(nil)

	params := &ec2.ConfirmProductInstanceInput{
		InstanceID:  aws.String("String"), // Required
		ProductCode: aws.String("String"), // Required
		DryRun:      aws.Boolean(true),
	}
	resp, err := svc.ConfirmProductInstance(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CopyImage() {
	svc := ec2.New(nil)

	params := &ec2.CopyImageInput{
		Name:          aws.String("String"), // Required
		SourceImageID: aws.String("String"), // Required
		SourceRegion:  aws.String("String"), // Required
		ClientToken:   aws.String("String"),
		Description:   aws.String("String"),
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.CopyImage(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CopySnapshot() {
	svc := ec2.New(nil)

	params := &ec2.CopySnapshotInput{
		SourceRegion:      aws.String("String"), // Required
		SourceSnapshotID:  aws.String("String"), // Required
		Description:       aws.String("String"),
		DestinationRegion: aws.String("String"),
		DryRun:            aws.Boolean(true),
		PresignedURL:      aws.String("String"),
	}
	resp, err := svc.CopySnapshot(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateCustomerGateway() {
	svc := ec2.New(nil)

	params := &ec2.CreateCustomerGatewayInput{
		BGPASN:   aws.Long(1),               // Required
		PublicIP: aws.String("String"),      // Required
		Type:     aws.String("GatewayType"), // Required
		DryRun:   aws.Boolean(true),
	}
	resp, err := svc.CreateCustomerGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateDHCPOptions() {
	svc := ec2.New(nil)

	params := &ec2.CreateDHCPOptionsInput{
		DHCPConfigurations: []*ec2.NewDHCPConfiguration{ // Required
			&ec2.NewDHCPConfiguration{ // Required
				Key: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CreateDHCPOptions(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateImage() {
	svc := ec2.New(nil)

	params := &ec2.CreateImageInput{
		InstanceID: aws.String("String"), // Required
		Name:       aws.String("String"), // Required
		BlockDeviceMappings: []*ec2.BlockDeviceMapping{
			&ec2.BlockDeviceMapping{ // Required
				DeviceName: aws.String("String"),
				EBS: &ec2.EBSBlockDevice{
					DeleteOnTermination: aws.Boolean(true),
					Encrypted:           aws.Boolean(true),
					IOPS:                aws.Long(1),
					SnapshotID:          aws.String("String"),
					VolumeSize:          aws.Long(1),
					VolumeType:          aws.String("VolumeType"),
				},
				NoDevice:    aws.String("String"),
				VirtualName: aws.String("String"),
			},
			// More values...
		},
		Description: aws.String("String"),
		DryRun:      aws.Boolean(true),
		NoReboot:    aws.Boolean(true),
	}
	resp, err := svc.CreateImage(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateInstanceExportTask() {
	svc := ec2.New(nil)

	params := &ec2.CreateInstanceExportTaskInput{
		InstanceID:  aws.String("String"), // Required
		Description: aws.String("String"),
		ExportToS3Task: &ec2.ExportToS3TaskSpecification{
			ContainerFormat: aws.String("ContainerFormat"),
			DiskImageFormat: aws.String("DiskImageFormat"),
			S3Bucket:        aws.String("String"),
			S3Prefix:        aws.String("String"),
		},
		TargetEnvironment: aws.String("ExportEnvironment"),
	}
	resp, err := svc.CreateInstanceExportTask(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateInternetGateway() {
	svc := ec2.New(nil)

	params := &ec2.CreateInternetGatewayInput{
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CreateInternetGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateKeyPair() {
	svc := ec2.New(nil)

	params := &ec2.CreateKeyPairInput{
		KeyName: aws.String("String"), // Required
		DryRun:  aws.Boolean(true),
	}
	resp, err := svc.CreateKeyPair(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateNetworkACL() {
	svc := ec2.New(nil)

	params := &ec2.CreateNetworkACLInput{
		VPCID:  aws.String("String"), // Required
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CreateNetworkACL(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateNetworkACLEntry() {
	svc := ec2.New(nil)

	params := &ec2.CreateNetworkACLEntryInput{
		CIDRBlock:    aws.String("String"),     // Required
		Egress:       aws.Boolean(true),        // Required
		NetworkACLID: aws.String("String"),     // Required
		Protocol:     aws.String("String"),     // Required
		RuleAction:   aws.String("RuleAction"), // Required
		RuleNumber:   aws.Long(1),              // Required
		DryRun:       aws.Boolean(true),
		ICMPTypeCode: &ec2.ICMPTypeCode{
			Code: aws.Long(1),
			Type: aws.Long(1),
		},
		PortRange: &ec2.PortRange{
			From: aws.Long(1),
			To:   aws.Long(1),
		},
	}
	resp, err := svc.CreateNetworkACLEntry(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateNetworkInterface() {
	svc := ec2.New(nil)

	params := &ec2.CreateNetworkInterfaceInput{
		SubnetID:    aws.String("String"), // Required
		Description: aws.String("String"),
		DryRun:      aws.Boolean(true),
		Groups: []*string{
			aws.String("String"), // Required
			// More values...
		},
		PrivateIPAddress: aws.String("String"),
		PrivateIPAddresses: []*ec2.PrivateIPAddressSpecification{
			&ec2.PrivateIPAddressSpecification{ // Required
				PrivateIPAddress: aws.String("String"), // Required
				Primary:          aws.Boolean(true),
			},
			// More values...
		},
		SecondaryPrivateIPAddressCount: aws.Long(1),
	}
	resp, err := svc.CreateNetworkInterface(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreatePlacementGroup() {
	svc := ec2.New(nil)

	params := &ec2.CreatePlacementGroupInput{
		GroupName: aws.String("String"),            // Required
		Strategy:  aws.String("PlacementStrategy"), // Required
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.CreatePlacementGroup(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateReservedInstancesListing() {
	svc := ec2.New(nil)

	params := &ec2.CreateReservedInstancesListingInput{
		ClientToken:   aws.String("String"), // Required
		InstanceCount: aws.Long(1),          // Required
		PriceSchedules: []*ec2.PriceScheduleSpecification{ // Required
			&ec2.PriceScheduleSpecification{ // Required
				CurrencyCode: aws.String("CurrencyCodeValues"),
				Price:        aws.Double(1.0),
				Term:         aws.Long(1),
			},
			// More values...
		},
		ReservedInstancesID: aws.String("String"), // Required
	}
	resp, err := svc.CreateReservedInstancesListing(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateRoute() {
	svc := ec2.New(nil)

	params := &ec2.CreateRouteInput{
		DestinationCIDRBlock:   aws.String("String"), // Required
		RouteTableID:           aws.String("String"), // Required
		DryRun:                 aws.Boolean(true),
		GatewayID:              aws.String("String"),
		InstanceID:             aws.String("String"),
		NetworkInterfaceID:     aws.String("String"),
		VPCPeeringConnectionID: aws.String("String"),
	}
	resp, err := svc.CreateRoute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateRouteTable() {
	svc := ec2.New(nil)

	params := &ec2.CreateRouteTableInput{
		VPCID:  aws.String("String"), // Required
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CreateRouteTable(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateSecurityGroup() {
	svc := ec2.New(nil)

	params := &ec2.CreateSecurityGroupInput{
		Description: aws.String("String"), // Required
		GroupName:   aws.String("String"), // Required
		DryRun:      aws.Boolean(true),
		VPCID:       aws.String("String"),
	}
	resp, err := svc.CreateSecurityGroup(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateSnapshot() {
	svc := ec2.New(nil)

	params := &ec2.CreateSnapshotInput{
		VolumeID:    aws.String("String"), // Required
		Description: aws.String("String"),
		DryRun:      aws.Boolean(true),
	}
	resp, err := svc.CreateSnapshot(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateSpotDatafeedSubscription() {
	svc := ec2.New(nil)

	params := &ec2.CreateSpotDatafeedSubscriptionInput{
		Bucket: aws.String("String"), // Required
		DryRun: aws.Boolean(true),
		Prefix: aws.String("String"),
	}
	resp, err := svc.CreateSpotDatafeedSubscription(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateSubnet() {
	svc := ec2.New(nil)

	params := &ec2.CreateSubnetInput{
		CIDRBlock:        aws.String("String"), // Required
		VPCID:            aws.String("String"), // Required
		AvailabilityZone: aws.String("String"),
		DryRun:           aws.Boolean(true),
	}
	resp, err := svc.CreateSubnet(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateTags() {
	svc := ec2.New(nil)

	params := &ec2.CreateTagsInput{
		Resources: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		Tags: []*ec2.Tag{ // Required
			&ec2.Tag{ // Required
				Key:   aws.String("String"),
				Value: aws.String("String"),
			},
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.CreateTags(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVPC() {
	svc := ec2.New(nil)

	params := &ec2.CreateVPCInput{
		CIDRBlock:       aws.String("String"), // Required
		DryRun:          aws.Boolean(true),
		InstanceTenancy: aws.String("Tenancy"),
	}
	resp, err := svc.CreateVPC(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVPCPeeringConnection() {
	svc := ec2.New(nil)

	params := &ec2.CreateVPCPeeringConnectionInput{
		DryRun:      aws.Boolean(true),
		PeerOwnerID: aws.String("String"),
		PeerVPCID:   aws.String("String"),
		VPCID:       aws.String("String"),
	}
	resp, err := svc.CreateVPCPeeringConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVPNConnection() {
	svc := ec2.New(nil)

	params := &ec2.CreateVPNConnectionInput{
		CustomerGatewayID: aws.String("String"), // Required
		Type:              aws.String("String"), // Required
		VPNGatewayID:      aws.String("String"), // Required
		DryRun:            aws.Boolean(true),
		Options: &ec2.VPNConnectionOptionsSpecification{
			StaticRoutesOnly: aws.Boolean(true),
		},
	}
	resp, err := svc.CreateVPNConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVPNConnectionRoute() {
	svc := ec2.New(nil)

	params := &ec2.CreateVPNConnectionRouteInput{
		DestinationCIDRBlock: aws.String("String"), // Required
		VPNConnectionID:      aws.String("String"), // Required
	}
	resp, err := svc.CreateVPNConnectionRoute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVPNGateway() {
	svc := ec2.New(nil)

	params := &ec2.CreateVPNGatewayInput{
		Type:             aws.String("GatewayType"), // Required
		AvailabilityZone: aws.String("String"),
		DryRun:           aws.Boolean(true),
	}
	resp, err := svc.CreateVPNGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_CreateVolume() {
	svc := ec2.New(nil)

	params := &ec2.CreateVolumeInput{
		AvailabilityZone: aws.String("String"), // Required
		DryRun:           aws.Boolean(true),
		Encrypted:        aws.Boolean(true),
		IOPS:             aws.Long(1),
		KMSKeyID:         aws.String("String"),
		Size:             aws.Long(1),
		SnapshotID:       aws.String("String"),
		VolumeType:       aws.String("VolumeType"),
	}
	resp, err := svc.CreateVolume(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteCustomerGateway() {
	svc := ec2.New(nil)

	params := &ec2.DeleteCustomerGatewayInput{
		CustomerGatewayID: aws.String("String"), // Required
		DryRun:            aws.Boolean(true),
	}
	resp, err := svc.DeleteCustomerGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteDHCPOptions() {
	svc := ec2.New(nil)

	params := &ec2.DeleteDHCPOptionsInput{
		DHCPOptionsID: aws.String("String"), // Required
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.DeleteDHCPOptions(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteInternetGateway() {
	svc := ec2.New(nil)

	params := &ec2.DeleteInternetGatewayInput{
		InternetGatewayID: aws.String("String"), // Required
		DryRun:            aws.Boolean(true),
	}
	resp, err := svc.DeleteInternetGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteKeyPair() {
	svc := ec2.New(nil)

	params := &ec2.DeleteKeyPairInput{
		KeyName: aws.String("String"), // Required
		DryRun:  aws.Boolean(true),
	}
	resp, err := svc.DeleteKeyPair(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteNetworkACL() {
	svc := ec2.New(nil)

	params := &ec2.DeleteNetworkACLInput{
		NetworkACLID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.DeleteNetworkACL(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteNetworkACLEntry() {
	svc := ec2.New(nil)

	params := &ec2.DeleteNetworkACLEntryInput{
		Egress:       aws.Boolean(true),    // Required
		NetworkACLID: aws.String("String"), // Required
		RuleNumber:   aws.Long(1),          // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.DeleteNetworkACLEntry(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteNetworkInterface() {
	svc := ec2.New(nil)

	params := &ec2.DeleteNetworkInterfaceInput{
		NetworkInterfaceID: aws.String("String"), // Required
		DryRun:             aws.Boolean(true),
	}
	resp, err := svc.DeleteNetworkInterface(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeletePlacementGroup() {
	svc := ec2.New(nil)

	params := &ec2.DeletePlacementGroupInput{
		GroupName: aws.String("String"), // Required
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.DeletePlacementGroup(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteRoute() {
	svc := ec2.New(nil)

	params := &ec2.DeleteRouteInput{
		DestinationCIDRBlock: aws.String("String"), // Required
		RouteTableID:         aws.String("String"), // Required
		DryRun:               aws.Boolean(true),
	}
	resp, err := svc.DeleteRoute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteRouteTable() {
	svc := ec2.New(nil)

	params := &ec2.DeleteRouteTableInput{
		RouteTableID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.DeleteRouteTable(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteSecurityGroup() {
	svc := ec2.New(nil)

	params := &ec2.DeleteSecurityGroupInput{
		DryRun:    aws.Boolean(true),
		GroupID:   aws.String("String"),
		GroupName: aws.String("String"),
	}
	resp, err := svc.DeleteSecurityGroup(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteSnapshot() {
	svc := ec2.New(nil)

	params := &ec2.DeleteSnapshotInput{
		SnapshotID: aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.DeleteSnapshot(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteSpotDatafeedSubscription() {
	svc := ec2.New(nil)

	params := &ec2.DeleteSpotDatafeedSubscriptionInput{
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.DeleteSpotDatafeedSubscription(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteSubnet() {
	svc := ec2.New(nil)

	params := &ec2.DeleteSubnetInput{
		SubnetID: aws.String("String"), // Required
		DryRun:   aws.Boolean(true),
	}
	resp, err := svc.DeleteSubnet(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteTags() {
	svc := ec2.New(nil)

	params := &ec2.DeleteTagsInput{
		Resources: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Tags: []*ec2.Tag{
			&ec2.Tag{ // Required
				Key:   aws.String("String"),
				Value: aws.String("String"),
			},
			// More values...
		},
	}
	resp, err := svc.DeleteTags(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVPC() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVPCInput{
		VPCID:  aws.String("String"), // Required
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.DeleteVPC(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVPCPeeringConnection() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVPCPeeringConnectionInput{
		VPCPeeringConnectionID: aws.String("String"), // Required
		DryRun:                 aws.Boolean(true),
	}
	resp, err := svc.DeleteVPCPeeringConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVPNConnection() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVPNConnectionInput{
		VPNConnectionID: aws.String("String"), // Required
		DryRun:          aws.Boolean(true),
	}
	resp, err := svc.DeleteVPNConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVPNConnectionRoute() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVPNConnectionRouteInput{
		DestinationCIDRBlock: aws.String("String"), // Required
		VPNConnectionID:      aws.String("String"), // Required
	}
	resp, err := svc.DeleteVPNConnectionRoute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVPNGateway() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVPNGatewayInput{
		VPNGatewayID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.DeleteVPNGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeleteVolume() {
	svc := ec2.New(nil)

	params := &ec2.DeleteVolumeInput{
		VolumeID: aws.String("String"), // Required
		DryRun:   aws.Boolean(true),
	}
	resp, err := svc.DeleteVolume(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DeregisterImage() {
	svc := ec2.New(nil)

	params := &ec2.DeregisterImageInput{
		ImageID: aws.String("String"), // Required
		DryRun:  aws.Boolean(true),
	}
	resp, err := svc.DeregisterImage(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeAccountAttributes() {
	svc := ec2.New(nil)

	params := &ec2.DescribeAccountAttributesInput{
		AttributeNames: []*string{
			aws.String("AccountAttributeName"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.DescribeAccountAttributes(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeAddresses() {
	svc := ec2.New(nil)

	params := &ec2.DescribeAddressesInput{
		AllocationIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		PublicIPs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeAddresses(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeAvailabilityZones() {
	svc := ec2.New(nil)

	params := &ec2.DescribeAvailabilityZonesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		ZoneNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeAvailabilityZones(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeBundleTasks() {
	svc := ec2.New(nil)

	params := &ec2.DescribeBundleTasksInput{
		BundleIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
	}
	resp, err := svc.DescribeBundleTasks(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeClassicLinkInstances() {
	svc := ec2.New(nil)

	params := &ec2.DescribeClassicLinkInstancesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		InstanceIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeClassicLinkInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeConversionTasks() {
	svc := ec2.New(nil)

	params := &ec2.DescribeConversionTasksInput{
		ConversionTaskIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
	}
	resp, err := svc.DescribeConversionTasks(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeCustomerGateways() {
	svc := ec2.New(nil)

	params := &ec2.DescribeCustomerGatewaysInput{
		CustomerGatewayIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
	}
	resp, err := svc.DescribeCustomerGateways(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeDHCPOptions() {
	svc := ec2.New(nil)

	params := &ec2.DescribeDHCPOptionsInput{
		DHCPOptionsIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
	}
	resp, err := svc.DescribeDHCPOptions(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeExportTasks() {
	svc := ec2.New(nil)

	params := &ec2.DescribeExportTasksInput{
		ExportTaskIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeExportTasks(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeImageAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeImageAttributeInput{
		Attribute: aws.String("ImageAttributeName"), // Required
		ImageID:   aws.String("String"),             // Required
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.DescribeImageAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeImages() {
	svc := ec2.New(nil)

	params := &ec2.DescribeImagesInput{
		DryRun: aws.Boolean(true),
		ExecutableUsers: []*string{
			aws.String("String"), // Required
			// More values...
		},
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		ImageIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		Owners: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeImages(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeImportImageTasks() {
	svc := ec2.New(nil)

	params := &ec2.DescribeImportImageTasksInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		ImportTaskIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeImportImageTasks(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeImportSnapshotTasks() {
	svc := ec2.New(nil)

	params := &ec2.DescribeImportSnapshotTasksInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		ImportTaskIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeImportSnapshotTasks(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeInstanceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeInstanceAttributeInput{
		Attribute:  aws.String("InstanceAttributeName"), // Required
		InstanceID: aws.String("String"),                // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.DescribeInstanceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeInstanceStatus() {
	svc := ec2.New(nil)

	params := &ec2.DescribeInstanceStatusInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		IncludeAllInstances: aws.Boolean(true),
		InstanceIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeInstanceStatus(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeInstances() {
	svc := ec2.New(nil)

	params := &ec2.DescribeInstancesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		InstanceIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeInternetGateways() {
	svc := ec2.New(nil)

	params := &ec2.DescribeInternetGatewaysInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		InternetGatewayIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeInternetGateways(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeKeyPairs() {
	svc := ec2.New(nil)

	params := &ec2.DescribeKeyPairsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		KeyNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeKeyPairs(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeNetworkACLs() {
	svc := ec2.New(nil)

	params := &ec2.DescribeNetworkACLsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		NetworkACLIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeNetworkACLs(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeNetworkInterfaceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeNetworkInterfaceAttributeInput{
		NetworkInterfaceID: aws.String("String"), // Required
		Attribute:          aws.String("NetworkInterfaceAttribute"),
		DryRun:             aws.Boolean(true),
	}
	resp, err := svc.DescribeNetworkInterfaceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeNetworkInterfaces() {
	svc := ec2.New(nil)

	params := &ec2.DescribeNetworkInterfacesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		NetworkInterfaceIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeNetworkInterfaces(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribePlacementGroups() {
	svc := ec2.New(nil)

	params := &ec2.DescribePlacementGroupsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		GroupNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribePlacementGroups(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeRegions() {
	svc := ec2.New(nil)

	params := &ec2.DescribeRegionsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		RegionNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeRegions(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeReservedInstances() {
	svc := ec2.New(nil)

	params := &ec2.DescribeReservedInstancesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		OfferingType: aws.String("OfferingTypeValues"),
		ReservedInstancesIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeReservedInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeReservedInstancesListings() {
	svc := ec2.New(nil)

	params := &ec2.DescribeReservedInstancesListingsInput{
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		ReservedInstancesID:        aws.String("String"),
		ReservedInstancesListingID: aws.String("String"),
	}
	resp, err := svc.DescribeReservedInstancesListings(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeReservedInstancesModifications() {
	svc := ec2.New(nil)

	params := &ec2.DescribeReservedInstancesModificationsInput{
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		NextToken: aws.String("String"),
		ReservedInstancesModificationIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeReservedInstancesModifications(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeReservedInstancesOfferings() {
	svc := ec2.New(nil)

	params := &ec2.DescribeReservedInstancesOfferingsInput{
		AvailabilityZone: aws.String("String"),
		DryRun:           aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		IncludeMarketplace: aws.Boolean(true),
		InstanceTenancy:    aws.String("Tenancy"),
		InstanceType:       aws.String("InstanceType"),
		MaxDuration:        aws.Long(1),
		MaxInstanceCount:   aws.Long(1),
		MaxResults:         aws.Long(1),
		MinDuration:        aws.Long(1),
		NextToken:          aws.String("String"),
		OfferingType:       aws.String("OfferingTypeValues"),
		ProductDescription: aws.String("RIProductDescription"),
		ReservedInstancesOfferingIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeReservedInstancesOfferings(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeRouteTables() {
	svc := ec2.New(nil)

	params := &ec2.DescribeRouteTablesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		RouteTableIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeRouteTables(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSecurityGroups() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSecurityGroupsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		GroupIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		GroupNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeSecurityGroups(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSnapshotAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSnapshotAttributeInput{
		Attribute:  aws.String("SnapshotAttributeName"), // Required
		SnapshotID: aws.String("String"),                // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.DescribeSnapshotAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSnapshots() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSnapshotsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
		OwnerIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		RestorableByUserIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		SnapshotIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeSnapshots(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSpotDatafeedSubscription() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSpotDatafeedSubscriptionInput{
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.DescribeSpotDatafeedSubscription(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSpotInstanceRequests() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSpotInstanceRequestsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		SpotInstanceRequestIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeSpotInstanceRequests(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSpotPriceHistory() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSpotPriceHistoryInput{
		AvailabilityZone: aws.String("String"),
		DryRun:           aws.Boolean(true),
		EndTime:          aws.Time(time.Now()),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		InstanceTypes: []*string{
			aws.String("InstanceType"), // Required
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
		ProductDescriptions: []*string{
			aws.String("String"), // Required
			// More values...
		},
		StartTime: aws.Time(time.Now()),
	}
	resp, err := svc.DescribeSpotPriceHistory(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeSubnets() {
	svc := ec2.New(nil)

	params := &ec2.DescribeSubnetsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		SubnetIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeSubnets(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeTags() {
	svc := ec2.New(nil)

	params := &ec2.DescribeTagsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
	}
	resp, err := svc.DescribeTags(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPCAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPCAttributeInput{
		VPCID:     aws.String("String"), // Required
		Attribute: aws.String("VpcAttributeName"),
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.DescribeVPCAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPCClassicLink() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPCClassicLinkInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		VPCIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVPCClassicLink(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPCPeeringConnections() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPCPeeringConnectionsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		VPCPeeringConnectionIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVPCPeeringConnections(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPCs() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPCsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		VPCIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVPCs(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPNConnections() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPNConnectionsInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		VPNConnectionIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVPNConnections(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVPNGateways() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVPNGatewaysInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		VPNGatewayIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVPNGateways(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVolumeAttribute() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVolumeAttributeInput{
		VolumeID:  aws.String("String"), // Required
		Attribute: aws.String("VolumeAttributeName"),
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.DescribeVolumeAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVolumeStatus() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVolumeStatusInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
		VolumeIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVolumeStatus(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DescribeVolumes() {
	svc := ec2.New(nil)

	params := &ec2.DescribeVolumesInput{
		DryRun: aws.Boolean(true),
		Filters: []*ec2.Filter{
			&ec2.Filter{ // Required
				Name: aws.String("String"),
				Values: []*string{
					aws.String("String"), // Required
					// More values...
				},
			},
			// More values...
		},
		MaxResults: aws.Long(1),
		NextToken:  aws.String("String"),
		VolumeIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.DescribeVolumes(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DetachClassicLinkVPC() {
	svc := ec2.New(nil)

	params := &ec2.DetachClassicLinkVPCInput{
		InstanceID: aws.String("String"), // Required
		VPCID:      aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.DetachClassicLinkVPC(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DetachInternetGateway() {
	svc := ec2.New(nil)

	params := &ec2.DetachInternetGatewayInput{
		InternetGatewayID: aws.String("String"), // Required
		VPCID:             aws.String("String"), // Required
		DryRun:            aws.Boolean(true),
	}
	resp, err := svc.DetachInternetGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DetachNetworkInterface() {
	svc := ec2.New(nil)

	params := &ec2.DetachNetworkInterfaceInput{
		AttachmentID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
		Force:        aws.Boolean(true),
	}
	resp, err := svc.DetachNetworkInterface(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DetachVPNGateway() {
	svc := ec2.New(nil)

	params := &ec2.DetachVPNGatewayInput{
		VPCID:        aws.String("String"), // Required
		VPNGatewayID: aws.String("String"), // Required
		DryRun:       aws.Boolean(true),
	}
	resp, err := svc.DetachVPNGateway(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DetachVolume() {
	svc := ec2.New(nil)

	params := &ec2.DetachVolumeInput{
		VolumeID:   aws.String("String"), // Required
		Device:     aws.String("String"),
		DryRun:     aws.Boolean(true),
		Force:      aws.Boolean(true),
		InstanceID: aws.String("String"),
	}
	resp, err := svc.DetachVolume(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DisableVGWRoutePropagation() {
	svc := ec2.New(nil)

	params := &ec2.DisableVGWRoutePropagationInput{
		GatewayID:    aws.String("String"), // Required
		RouteTableID: aws.String("String"), // Required
	}
	resp, err := svc.DisableVGWRoutePropagation(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DisableVPCClassicLink() {
	svc := ec2.New(nil)

	params := &ec2.DisableVPCClassicLinkInput{
		VPCID:  aws.String("String"), // Required
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.DisableVPCClassicLink(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DisassociateAddress() {
	svc := ec2.New(nil)

	params := &ec2.DisassociateAddressInput{
		AssociationID: aws.String("String"),
		DryRun:        aws.Boolean(true),
		PublicIP:      aws.String("String"),
	}
	resp, err := svc.DisassociateAddress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_DisassociateRouteTable() {
	svc := ec2.New(nil)

	params := &ec2.DisassociateRouteTableInput{
		AssociationID: aws.String("String"), // Required
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.DisassociateRouteTable(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_EnableVGWRoutePropagation() {
	svc := ec2.New(nil)

	params := &ec2.EnableVGWRoutePropagationInput{
		GatewayID:    aws.String("String"), // Required
		RouteTableID: aws.String("String"), // Required
	}
	resp, err := svc.EnableVGWRoutePropagation(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_EnableVPCClassicLink() {
	svc := ec2.New(nil)

	params := &ec2.EnableVPCClassicLinkInput{
		VPCID:  aws.String("String"), // Required
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.EnableVPCClassicLink(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_EnableVolumeIO() {
	svc := ec2.New(nil)

	params := &ec2.EnableVolumeIOInput{
		VolumeID: aws.String("String"), // Required
		DryRun:   aws.Boolean(true),
	}
	resp, err := svc.EnableVolumeIO(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_GetConsoleOutput() {
	svc := ec2.New(nil)

	params := &ec2.GetConsoleOutputInput{
		InstanceID: aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.GetConsoleOutput(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_GetPasswordData() {
	svc := ec2.New(nil)

	params := &ec2.GetPasswordDataInput{
		InstanceID: aws.String("String"), // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.GetPasswordData(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ImportImage() {
	svc := ec2.New(nil)

	params := &ec2.ImportImageInput{
		Architecture: aws.String("String"),
		ClientData: &ec2.ClientData{
			Comment:     aws.String("String"),
			UploadEnd:   aws.Time(time.Now()),
			UploadSize:  aws.Double(1.0),
			UploadStart: aws.Time(time.Now()),
		},
		ClientToken: aws.String("String"),
		Description: aws.String("String"),
		DiskContainers: []*ec2.ImageDiskContainer{
			&ec2.ImageDiskContainer{ // Required
				Description: aws.String("String"),
				DeviceName:  aws.String("String"),
				Format:      aws.String("String"),
				SnapshotID:  aws.String("String"),
				URL:         aws.String("String"),
				UserBucket: &ec2.UserBucket{
					S3Bucket: aws.String("String"),
					S3Key:    aws.String("String"),
				},
			},
			// More values...
		},
		DryRun:      aws.Boolean(true),
		Hypervisor:  aws.String("String"),
		LicenseType: aws.String("String"),
		Platform:    aws.String("String"),
		RoleName:    aws.String("String"),
	}
	resp, err := svc.ImportImage(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ImportInstance() {
	svc := ec2.New(nil)

	params := &ec2.ImportInstanceInput{
		Platform:    aws.String("PlatformValues"), // Required
		Description: aws.String("String"),
		DiskImages: []*ec2.DiskImage{
			&ec2.DiskImage{ // Required
				Description: aws.String("String"),
				Image: &ec2.DiskImageDetail{
					Bytes:             aws.Long(1),                   // Required
					Format:            aws.String("DiskImageFormat"), // Required
					ImportManifestURL: aws.String("String"),          // Required
				},
				Volume: &ec2.VolumeDetail{
					Size: aws.Long(1), // Required
				},
			},
			// More values...
		},
		DryRun: aws.Boolean(true),
		LaunchSpecification: &ec2.ImportInstanceLaunchSpecification{
			AdditionalInfo: aws.String("String"),
			Architecture:   aws.String("ArchitectureValues"),
			GroupIDs: []*string{
				aws.String("String"), // Required
				// More values...
			},
			GroupNames: []*string{
				aws.String("String"), // Required
				// More values...
			},
			InstanceInitiatedShutdownBehavior: aws.String("ShutdownBehavior"),
			InstanceType:                      aws.String("InstanceType"),
			Monitoring:                        aws.Boolean(true),
			Placement: &ec2.Placement{
				AvailabilityZone: aws.String("String"),
				GroupName:        aws.String("String"),
				Tenancy:          aws.String("Tenancy"),
			},
			PrivateIPAddress: aws.String("String"),
			SubnetID:         aws.String("String"),
			UserData: &ec2.UserData{
				Data: aws.String("String"),
			},
		},
	}
	resp, err := svc.ImportInstance(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ImportKeyPair() {
	svc := ec2.New(nil)

	params := &ec2.ImportKeyPairInput{
		KeyName:           aws.String("String"), // Required
		PublicKeyMaterial: []byte("PAYLOAD"),    // Required
		DryRun:            aws.Boolean(true),
	}
	resp, err := svc.ImportKeyPair(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ImportSnapshot() {
	svc := ec2.New(nil)

	params := &ec2.ImportSnapshotInput{
		ClientData: &ec2.ClientData{
			Comment:     aws.String("String"),
			UploadEnd:   aws.Time(time.Now()),
			UploadSize:  aws.Double(1.0),
			UploadStart: aws.Time(time.Now()),
		},
		ClientToken: aws.String("String"),
		Description: aws.String("String"),
		DiskContainer: &ec2.SnapshotDiskContainer{
			Description: aws.String("String"),
			Format:      aws.String("String"),
			URL:         aws.String("String"),
			UserBucket: &ec2.UserBucket{
				S3Bucket: aws.String("String"),
				S3Key:    aws.String("String"),
			},
		},
		DryRun:   aws.Boolean(true),
		RoleName: aws.String("String"),
	}
	resp, err := svc.ImportSnapshot(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ImportVolume() {
	svc := ec2.New(nil)

	params := &ec2.ImportVolumeInput{
		AvailabilityZone: aws.String("String"), // Required
		Image: &ec2.DiskImageDetail{ // Required
			Bytes:             aws.Long(1),                   // Required
			Format:            aws.String("DiskImageFormat"), // Required
			ImportManifestURL: aws.String("String"),          // Required
		},
		Volume: &ec2.VolumeDetail{ // Required
			Size: aws.Long(1), // Required
		},
		Description: aws.String("String"),
		DryRun:      aws.Boolean(true),
	}
	resp, err := svc.ImportVolume(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyImageAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifyImageAttributeInput{
		ImageID:   aws.String("String"), // Required
		Attribute: aws.String("String"),
		Description: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		DryRun: aws.Boolean(true),
		LaunchPermission: &ec2.LaunchPermissionModifications{
			Add: []*ec2.LaunchPermission{
				&ec2.LaunchPermission{ // Required
					Group:  aws.String("PermissionGroup"),
					UserID: aws.String("String"),
				},
				// More values...
			},
			Remove: []*ec2.LaunchPermission{
				&ec2.LaunchPermission{ // Required
					Group:  aws.String("PermissionGroup"),
					UserID: aws.String("String"),
				},
				// More values...
			},
		},
		OperationType: aws.String("String"),
		ProductCodes: []*string{
			aws.String("String"), // Required
			// More values...
		},
		UserGroups: []*string{
			aws.String("String"), // Required
			// More values...
		},
		UserIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		Value: aws.String("String"),
	}
	resp, err := svc.ModifyImageAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyInstanceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifyInstanceAttributeInput{
		InstanceID: aws.String("String"), // Required
		Attribute:  aws.String("InstanceAttributeName"),
		BlockDeviceMappings: []*ec2.InstanceBlockDeviceMappingSpecification{
			&ec2.InstanceBlockDeviceMappingSpecification{ // Required
				DeviceName: aws.String("String"),
				EBS: &ec2.EBSInstanceBlockDeviceSpecification{
					DeleteOnTermination: aws.Boolean(true),
					VolumeID:            aws.String("String"),
				},
				NoDevice:    aws.String("String"),
				VirtualName: aws.String("String"),
			},
			// More values...
		},
		DisableAPITermination: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
		DryRun: aws.Boolean(true),
		EBSOptimized: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
		Groups: []*string{
			aws.String("String"), // Required
			// More values...
		},
		InstanceInitiatedShutdownBehavior: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		InstanceType: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		Kernel: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		RAMDisk: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		SRIOVNetSupport: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		SourceDestCheck: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
		UserData: &ec2.BlobAttributeValue{
			Value: []byte("PAYLOAD"),
		},
		Value: aws.String("String"),
	}
	resp, err := svc.ModifyInstanceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyNetworkInterfaceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifyNetworkInterfaceAttributeInput{
		NetworkInterfaceID: aws.String("String"), // Required
		Attachment: &ec2.NetworkInterfaceAttachmentChanges{
			AttachmentID:        aws.String("String"),
			DeleteOnTermination: aws.Boolean(true),
		},
		Description: &ec2.AttributeValue{
			Value: aws.String("String"),
		},
		DryRun: aws.Boolean(true),
		Groups: []*string{
			aws.String("String"), // Required
			// More values...
		},
		SourceDestCheck: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
	}
	resp, err := svc.ModifyNetworkInterfaceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyReservedInstances() {
	svc := ec2.New(nil)

	params := &ec2.ModifyReservedInstancesInput{
		ReservedInstancesIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		TargetConfigurations: []*ec2.ReservedInstancesConfiguration{ // Required
			&ec2.ReservedInstancesConfiguration{ // Required
				AvailabilityZone: aws.String("String"),
				InstanceCount:    aws.Long(1),
				InstanceType:     aws.String("InstanceType"),
				Platform:         aws.String("String"),
			},
			// More values...
		},
		ClientToken: aws.String("String"),
	}
	resp, err := svc.ModifyReservedInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifySnapshotAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifySnapshotAttributeInput{
		SnapshotID: aws.String("String"), // Required
		Attribute:  aws.String("SnapshotAttributeName"),
		CreateVolumePermission: &ec2.CreateVolumePermissionModifications{
			Add: []*ec2.CreateVolumePermission{
				&ec2.CreateVolumePermission{ // Required
					Group:  aws.String("PermissionGroup"),
					UserID: aws.String("String"),
				},
				// More values...
			},
			Remove: []*ec2.CreateVolumePermission{
				&ec2.CreateVolumePermission{ // Required
					Group:  aws.String("PermissionGroup"),
					UserID: aws.String("String"),
				},
				// More values...
			},
		},
		DryRun: aws.Boolean(true),
		GroupNames: []*string{
			aws.String("String"), // Required
			// More values...
		},
		OperationType: aws.String("String"),
		UserIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.ModifySnapshotAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifySubnetAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifySubnetAttributeInput{
		SubnetID: aws.String("String"), // Required
		MapPublicIPOnLaunch: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
	}
	resp, err := svc.ModifySubnetAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyVPCAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifyVPCAttributeInput{
		VPCID: aws.String("String"), // Required
		EnableDNSHostnames: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
		EnableDNSSupport: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
	}
	resp, err := svc.ModifyVPCAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ModifyVolumeAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ModifyVolumeAttributeInput{
		VolumeID: aws.String("String"), // Required
		AutoEnableIO: &ec2.AttributeBooleanValue{
			Value: aws.Boolean(true),
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.ModifyVolumeAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_MonitorInstances() {
	svc := ec2.New(nil)

	params := &ec2.MonitorInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.MonitorInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_PurchaseReservedInstancesOffering() {
	svc := ec2.New(nil)

	params := &ec2.PurchaseReservedInstancesOfferingInput{
		InstanceCount:               aws.Long(1),          // Required
		ReservedInstancesOfferingID: aws.String("String"), // Required
		DryRun: aws.Boolean(true),
		LimitPrice: &ec2.ReservedInstanceLimitPrice{
			Amount:       aws.Double(1.0),
			CurrencyCode: aws.String("CurrencyCodeValues"),
		},
	}
	resp, err := svc.PurchaseReservedInstancesOffering(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RebootInstances() {
	svc := ec2.New(nil)

	params := &ec2.RebootInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.RebootInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RegisterImage() {
	svc := ec2.New(nil)

	params := &ec2.RegisterImageInput{
		Name:         aws.String("String"), // Required
		Architecture: aws.String("ArchitectureValues"),
		BlockDeviceMappings: []*ec2.BlockDeviceMapping{
			&ec2.BlockDeviceMapping{ // Required
				DeviceName: aws.String("String"),
				EBS: &ec2.EBSBlockDevice{
					DeleteOnTermination: aws.Boolean(true),
					Encrypted:           aws.Boolean(true),
					IOPS:                aws.Long(1),
					SnapshotID:          aws.String("String"),
					VolumeSize:          aws.Long(1),
					VolumeType:          aws.String("VolumeType"),
				},
				NoDevice:    aws.String("String"),
				VirtualName: aws.String("String"),
			},
			// More values...
		},
		Description:        aws.String("String"),
		DryRun:             aws.Boolean(true),
		ImageLocation:      aws.String("String"),
		KernelID:           aws.String("String"),
		RAMDiskID:          aws.String("String"),
		RootDeviceName:     aws.String("String"),
		SRIOVNetSupport:    aws.String("String"),
		VirtualizationType: aws.String("String"),
	}
	resp, err := svc.RegisterImage(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RejectVPCPeeringConnection() {
	svc := ec2.New(nil)

	params := &ec2.RejectVPCPeeringConnectionInput{
		VPCPeeringConnectionID: aws.String("String"), // Required
		DryRun:                 aws.Boolean(true),
	}
	resp, err := svc.RejectVPCPeeringConnection(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReleaseAddress() {
	svc := ec2.New(nil)

	params := &ec2.ReleaseAddressInput{
		AllocationID: aws.String("String"),
		DryRun:       aws.Boolean(true),
		PublicIP:     aws.String("String"),
	}
	resp, err := svc.ReleaseAddress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReplaceNetworkACLAssociation() {
	svc := ec2.New(nil)

	params := &ec2.ReplaceNetworkACLAssociationInput{
		AssociationID: aws.String("String"), // Required
		NetworkACLID:  aws.String("String"), // Required
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.ReplaceNetworkACLAssociation(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReplaceNetworkACLEntry() {
	svc := ec2.New(nil)

	params := &ec2.ReplaceNetworkACLEntryInput{
		CIDRBlock:    aws.String("String"),     // Required
		Egress:       aws.Boolean(true),        // Required
		NetworkACLID: aws.String("String"),     // Required
		Protocol:     aws.String("String"),     // Required
		RuleAction:   aws.String("RuleAction"), // Required
		RuleNumber:   aws.Long(1),              // Required
		DryRun:       aws.Boolean(true),
		ICMPTypeCode: &ec2.ICMPTypeCode{
			Code: aws.Long(1),
			Type: aws.Long(1),
		},
		PortRange: &ec2.PortRange{
			From: aws.Long(1),
			To:   aws.Long(1),
		},
	}
	resp, err := svc.ReplaceNetworkACLEntry(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReplaceRoute() {
	svc := ec2.New(nil)

	params := &ec2.ReplaceRouteInput{
		DestinationCIDRBlock:   aws.String("String"), // Required
		RouteTableID:           aws.String("String"), // Required
		DryRun:                 aws.Boolean(true),
		GatewayID:              aws.String("String"),
		InstanceID:             aws.String("String"),
		NetworkInterfaceID:     aws.String("String"),
		VPCPeeringConnectionID: aws.String("String"),
	}
	resp, err := svc.ReplaceRoute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReplaceRouteTableAssociation() {
	svc := ec2.New(nil)

	params := &ec2.ReplaceRouteTableAssociationInput{
		AssociationID: aws.String("String"), // Required
		RouteTableID:  aws.String("String"), // Required
		DryRun:        aws.Boolean(true),
	}
	resp, err := svc.ReplaceRouteTableAssociation(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ReportInstanceStatus() {
	svc := ec2.New(nil)

	params := &ec2.ReportInstanceStatusInput{
		Instances: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		ReasonCodes: []*string{ // Required
			aws.String("ReportInstanceReasonCodes"), // Required
			// More values...
		},
		Status:      aws.String("ReportStatusType"), // Required
		Description: aws.String("String"),
		DryRun:      aws.Boolean(true),
		EndTime:     aws.Time(time.Now()),
		StartTime:   aws.Time(time.Now()),
	}
	resp, err := svc.ReportInstanceStatus(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RequestSpotInstances() {
	svc := ec2.New(nil)

	params := &ec2.RequestSpotInstancesInput{
		SpotPrice:             aws.String("String"), // Required
		AvailabilityZoneGroup: aws.String("String"),
		DryRun:                aws.Boolean(true),
		InstanceCount:         aws.Long(1),
		LaunchGroup:           aws.String("String"),
		LaunchSpecification: &ec2.RequestSpotLaunchSpecification{
			AddressingType: aws.String("String"),
			BlockDeviceMappings: []*ec2.BlockDeviceMapping{
				&ec2.BlockDeviceMapping{ // Required
					DeviceName: aws.String("String"),
					EBS: &ec2.EBSBlockDevice{
						DeleteOnTermination: aws.Boolean(true),
						Encrypted:           aws.Boolean(true),
						IOPS:                aws.Long(1),
						SnapshotID:          aws.String("String"),
						VolumeSize:          aws.Long(1),
						VolumeType:          aws.String("VolumeType"),
					},
					NoDevice:    aws.String("String"),
					VirtualName: aws.String("String"),
				},
				// More values...
			},
			EBSOptimized: aws.Boolean(true),
			IAMInstanceProfile: &ec2.IAMInstanceProfileSpecification{
				ARN:  aws.String("String"),
				Name: aws.String("String"),
			},
			ImageID:      aws.String("String"),
			InstanceType: aws.String("InstanceType"),
			KernelID:     aws.String("String"),
			KeyName:      aws.String("String"),
			Monitoring: &ec2.RunInstancesMonitoringEnabled{
				Enabled: aws.Boolean(true), // Required
			},
			NetworkInterfaces: []*ec2.InstanceNetworkInterfaceSpecification{
				&ec2.InstanceNetworkInterfaceSpecification{ // Required
					AssociatePublicIPAddress: aws.Boolean(true),
					DeleteOnTermination:      aws.Boolean(true),
					Description:              aws.String("String"),
					DeviceIndex:              aws.Long(1),
					Groups: []*string{
						aws.String("String"), // Required
						// More values...
					},
					NetworkInterfaceID: aws.String("String"),
					PrivateIPAddress:   aws.String("String"),
					PrivateIPAddresses: []*ec2.PrivateIPAddressSpecification{
						&ec2.PrivateIPAddressSpecification{ // Required
							PrivateIPAddress: aws.String("String"), // Required
							Primary:          aws.Boolean(true),
						},
						// More values...
					},
					SecondaryPrivateIPAddressCount: aws.Long(1),
					SubnetID:                       aws.String("String"),
				},
				// More values...
			},
			Placement: &ec2.SpotPlacement{
				AvailabilityZone: aws.String("String"),
				GroupName:        aws.String("String"),
			},
			RAMDiskID: aws.String("String"),
			SecurityGroupIDs: []*string{
				aws.String("String"), // Required
				// More values...
			},
			SecurityGroups: []*string{
				aws.String("String"), // Required
				// More values...
			},
			SubnetID: aws.String("String"),
			UserData: aws.String("String"),
		},
		Type:       aws.String("SpotInstanceType"),
		ValidFrom:  aws.Time(time.Now()),
		ValidUntil: aws.Time(time.Now()),
	}
	resp, err := svc.RequestSpotInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ResetImageAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ResetImageAttributeInput{
		Attribute: aws.String("ResetImageAttributeName"), // Required
		ImageID:   aws.String("String"),                  // Required
		DryRun:    aws.Boolean(true),
	}
	resp, err := svc.ResetImageAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ResetInstanceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ResetInstanceAttributeInput{
		Attribute:  aws.String("InstanceAttributeName"), // Required
		InstanceID: aws.String("String"),                // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.ResetInstanceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ResetNetworkInterfaceAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ResetNetworkInterfaceAttributeInput{
		NetworkInterfaceID: aws.String("String"), // Required
		DryRun:             aws.Boolean(true),
		SourceDestCheck:    aws.String("String"),
	}
	resp, err := svc.ResetNetworkInterfaceAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_ResetSnapshotAttribute() {
	svc := ec2.New(nil)

	params := &ec2.ResetSnapshotAttributeInput{
		Attribute:  aws.String("SnapshotAttributeName"), // Required
		SnapshotID: aws.String("String"),                // Required
		DryRun:     aws.Boolean(true),
	}
	resp, err := svc.ResetSnapshotAttribute(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RevokeSecurityGroupEgress() {
	svc := ec2.New(nil)

	params := &ec2.RevokeSecurityGroupEgressInput{
		GroupID:  aws.String("String"), // Required
		CIDRIP:   aws.String("String"),
		DryRun:   aws.Boolean(true),
		FromPort: aws.Long(1),
		IPPermissions: []*ec2.IPPermission{
			&ec2.IPPermission{ // Required
				FromPort:   aws.Long(1),
				IPProtocol: aws.String("String"),
				IPRanges: []*ec2.IPRange{
					&ec2.IPRange{ // Required
						CIDRIP: aws.String("String"),
					},
					// More values...
				},
				ToPort: aws.Long(1),
				UserIDGroupPairs: []*ec2.UserIDGroupPair{
					&ec2.UserIDGroupPair{ // Required
						GroupID:   aws.String("String"),
						GroupName: aws.String("String"),
						UserID:    aws.String("String"),
					},
					// More values...
				},
			},
			// More values...
		},
		IPProtocol:                 aws.String("String"),
		SourceSecurityGroupName:    aws.String("String"),
		SourceSecurityGroupOwnerID: aws.String("String"),
		ToPort: aws.Long(1),
	}
	resp, err := svc.RevokeSecurityGroupEgress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RevokeSecurityGroupIngress() {
	svc := ec2.New(nil)

	params := &ec2.RevokeSecurityGroupIngressInput{
		CIDRIP:    aws.String("String"),
		DryRun:    aws.Boolean(true),
		FromPort:  aws.Long(1),
		GroupID:   aws.String("String"),
		GroupName: aws.String("String"),
		IPPermissions: []*ec2.IPPermission{
			&ec2.IPPermission{ // Required
				FromPort:   aws.Long(1),
				IPProtocol: aws.String("String"),
				IPRanges: []*ec2.IPRange{
					&ec2.IPRange{ // Required
						CIDRIP: aws.String("String"),
					},
					// More values...
				},
				ToPort: aws.Long(1),
				UserIDGroupPairs: []*ec2.UserIDGroupPair{
					&ec2.UserIDGroupPair{ // Required
						GroupID:   aws.String("String"),
						GroupName: aws.String("String"),
						UserID:    aws.String("String"),
					},
					// More values...
				},
			},
			// More values...
		},
		IPProtocol:                 aws.String("String"),
		SourceSecurityGroupName:    aws.String("String"),
		SourceSecurityGroupOwnerID: aws.String("String"),
		ToPort: aws.Long(1),
	}
	resp, err := svc.RevokeSecurityGroupIngress(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_RunInstances() {
	svc := ec2.New(nil)

	params := &ec2.RunInstancesInput{
		ImageID:        aws.String("String"), // Required
		MaxCount:       aws.Long(1),          // Required
		MinCount:       aws.Long(1),          // Required
		AdditionalInfo: aws.String("String"),
		BlockDeviceMappings: []*ec2.BlockDeviceMapping{
			&ec2.BlockDeviceMapping{ // Required
				DeviceName: aws.String("String"),
				EBS: &ec2.EBSBlockDevice{
					DeleteOnTermination: aws.Boolean(true),
					Encrypted:           aws.Boolean(true),
					IOPS:                aws.Long(1),
					SnapshotID:          aws.String("String"),
					VolumeSize:          aws.Long(1),
					VolumeType:          aws.String("VolumeType"),
				},
				NoDevice:    aws.String("String"),
				VirtualName: aws.String("String"),
			},
			// More values...
		},
		ClientToken:           aws.String("String"),
		DisableAPITermination: aws.Boolean(true),
		DryRun:                aws.Boolean(true),
		EBSOptimized:          aws.Boolean(true),
		IAMInstanceProfile: &ec2.IAMInstanceProfileSpecification{
			ARN:  aws.String("String"),
			Name: aws.String("String"),
		},
		InstanceInitiatedShutdownBehavior: aws.String("ShutdownBehavior"),
		InstanceType:                      aws.String("InstanceType"),
		KernelID:                          aws.String("String"),
		KeyName:                           aws.String("String"),
		Monitoring: &ec2.RunInstancesMonitoringEnabled{
			Enabled: aws.Boolean(true), // Required
		},
		NetworkInterfaces: []*ec2.InstanceNetworkInterfaceSpecification{
			&ec2.InstanceNetworkInterfaceSpecification{ // Required
				AssociatePublicIPAddress: aws.Boolean(true),
				DeleteOnTermination:      aws.Boolean(true),
				Description:              aws.String("String"),
				DeviceIndex:              aws.Long(1),
				Groups: []*string{
					aws.String("String"), // Required
					// More values...
				},
				NetworkInterfaceID: aws.String("String"),
				PrivateIPAddress:   aws.String("String"),
				PrivateIPAddresses: []*ec2.PrivateIPAddressSpecification{
					&ec2.PrivateIPAddressSpecification{ // Required
						PrivateIPAddress: aws.String("String"), // Required
						Primary:          aws.Boolean(true),
					},
					// More values...
				},
				SecondaryPrivateIPAddressCount: aws.Long(1),
				SubnetID:                       aws.String("String"),
			},
			// More values...
		},
		Placement: &ec2.Placement{
			AvailabilityZone: aws.String("String"),
			GroupName:        aws.String("String"),
			Tenancy:          aws.String("Tenancy"),
		},
		PrivateIPAddress: aws.String("String"),
		RAMDiskID:        aws.String("String"),
		SecurityGroupIDs: []*string{
			aws.String("String"), // Required
			// More values...
		},
		SecurityGroups: []*string{
			aws.String("String"), // Required
			// More values...
		},
		SubnetID: aws.String("String"),
		UserData: aws.String("String"),
	}
	resp, err := svc.RunInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_StartInstances() {
	svc := ec2.New(nil)

	params := &ec2.StartInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		AdditionalInfo: aws.String("String"),
		DryRun:         aws.Boolean(true),
	}
	resp, err := svc.StartInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_StopInstances() {
	svc := ec2.New(nil)

	params := &ec2.StopInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
		Force:  aws.Boolean(true),
	}
	resp, err := svc.StopInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_TerminateInstances() {
	svc := ec2.New(nil)

	params := &ec2.TerminateInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.TerminateInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_UnassignPrivateIPAddresses() {
	svc := ec2.New(nil)

	params := &ec2.UnassignPrivateIPAddressesInput{
		NetworkInterfaceID: aws.String("String"), // Required
		PrivateIPAddresses: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
	}
	resp, err := svc.UnassignPrivateIPAddresses(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}

func ExampleEC2_UnmonitorInstances() {
	svc := ec2.New(nil)

	params := &ec2.UnmonitorInstancesInput{
		InstanceIDs: []*string{ // Required
			aws.String("String"), // Required
			// More values...
		},
		DryRun: aws.Boolean(true),
	}
	resp, err := svc.UnmonitorInstances(params)

	if awserr := aws.Error(err); awserr != nil {
		// A service error occurred.
		fmt.Println("Error:", awserr.Code, awserr.Message)
	} else if err != nil {
		// A non-service error occurred.
		panic(err)
	}

	// Pretty-print the response data.
	fmt.Println(awsutil.StringValue(resp))
}
