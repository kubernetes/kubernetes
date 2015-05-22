package ec2iface

import (
	"github.com/awslabs/aws-sdk-go/service/ec2"
)

type EC2API interface {
	AcceptVPCPeeringConnection(*ec2.AcceptVPCPeeringConnectionInput) (*ec2.AcceptVPCPeeringConnectionOutput, error)

	AllocateAddress(*ec2.AllocateAddressInput) (*ec2.AllocateAddressOutput, error)

	AssignPrivateIPAddresses(*ec2.AssignPrivateIPAddressesInput) (*ec2.AssignPrivateIPAddressesOutput, error)

	AssociateAddress(*ec2.AssociateAddressInput) (*ec2.AssociateAddressOutput, error)

	AssociateDHCPOptions(*ec2.AssociateDHCPOptionsInput) (*ec2.AssociateDHCPOptionsOutput, error)

	AssociateRouteTable(*ec2.AssociateRouteTableInput) (*ec2.AssociateRouteTableOutput, error)

	AttachClassicLinkVPC(*ec2.AttachClassicLinkVPCInput) (*ec2.AttachClassicLinkVPCOutput, error)

	AttachInternetGateway(*ec2.AttachInternetGatewayInput) (*ec2.AttachInternetGatewayOutput, error)

	AttachNetworkInterface(*ec2.AttachNetworkInterfaceInput) (*ec2.AttachNetworkInterfaceOutput, error)

	AttachVPNGateway(*ec2.AttachVPNGatewayInput) (*ec2.AttachVPNGatewayOutput, error)

	AttachVolume(*ec2.AttachVolumeInput) (*ec2.VolumeAttachment, error)

	AuthorizeSecurityGroupEgress(*ec2.AuthorizeSecurityGroupEgressInput) (*ec2.AuthorizeSecurityGroupEgressOutput, error)

	AuthorizeSecurityGroupIngress(*ec2.AuthorizeSecurityGroupIngressInput) (*ec2.AuthorizeSecurityGroupIngressOutput, error)

	BundleInstance(*ec2.BundleInstanceInput) (*ec2.BundleInstanceOutput, error)

	CancelBundleTask(*ec2.CancelBundleTaskInput) (*ec2.CancelBundleTaskOutput, error)

	CancelConversionTask(*ec2.CancelConversionTaskInput) (*ec2.CancelConversionTaskOutput, error)

	CancelExportTask(*ec2.CancelExportTaskInput) (*ec2.CancelExportTaskOutput, error)

	CancelImportTask(*ec2.CancelImportTaskInput) (*ec2.CancelImportTaskOutput, error)

	CancelReservedInstancesListing(*ec2.CancelReservedInstancesListingInput) (*ec2.CancelReservedInstancesListingOutput, error)

	CancelSpotInstanceRequests(*ec2.CancelSpotInstanceRequestsInput) (*ec2.CancelSpotInstanceRequestsOutput, error)

	ConfirmProductInstance(*ec2.ConfirmProductInstanceInput) (*ec2.ConfirmProductInstanceOutput, error)

	CopyImage(*ec2.CopyImageInput) (*ec2.CopyImageOutput, error)

	CopySnapshot(*ec2.CopySnapshotInput) (*ec2.CopySnapshotOutput, error)

	CreateCustomerGateway(*ec2.CreateCustomerGatewayInput) (*ec2.CreateCustomerGatewayOutput, error)

	CreateDHCPOptions(*ec2.CreateDHCPOptionsInput) (*ec2.CreateDHCPOptionsOutput, error)

	CreateImage(*ec2.CreateImageInput) (*ec2.CreateImageOutput, error)

	CreateInstanceExportTask(*ec2.CreateInstanceExportTaskInput) (*ec2.CreateInstanceExportTaskOutput, error)

	CreateInternetGateway(*ec2.CreateInternetGatewayInput) (*ec2.CreateInternetGatewayOutput, error)

	CreateKeyPair(*ec2.CreateKeyPairInput) (*ec2.CreateKeyPairOutput, error)

	CreateNetworkACL(*ec2.CreateNetworkACLInput) (*ec2.CreateNetworkACLOutput, error)

	CreateNetworkACLEntry(*ec2.CreateNetworkACLEntryInput) (*ec2.CreateNetworkACLEntryOutput, error)

	CreateNetworkInterface(*ec2.CreateNetworkInterfaceInput) (*ec2.CreateNetworkInterfaceOutput, error)

	CreatePlacementGroup(*ec2.CreatePlacementGroupInput) (*ec2.CreatePlacementGroupOutput, error)

	CreateReservedInstancesListing(*ec2.CreateReservedInstancesListingInput) (*ec2.CreateReservedInstancesListingOutput, error)

	CreateRoute(*ec2.CreateRouteInput) (*ec2.CreateRouteOutput, error)

	CreateRouteTable(*ec2.CreateRouteTableInput) (*ec2.CreateRouteTableOutput, error)

	CreateSecurityGroup(*ec2.CreateSecurityGroupInput) (*ec2.CreateSecurityGroupOutput, error)

	CreateSnapshot(*ec2.CreateSnapshotInput) (*ec2.Snapshot, error)

	CreateSpotDatafeedSubscription(*ec2.CreateSpotDatafeedSubscriptionInput) (*ec2.CreateSpotDatafeedSubscriptionOutput, error)

	CreateSubnet(*ec2.CreateSubnetInput) (*ec2.CreateSubnetOutput, error)

	CreateTags(*ec2.CreateTagsInput) (*ec2.CreateTagsOutput, error)

	CreateVPC(*ec2.CreateVPCInput) (*ec2.CreateVPCOutput, error)

	CreateVPCPeeringConnection(*ec2.CreateVPCPeeringConnectionInput) (*ec2.CreateVPCPeeringConnectionOutput, error)

	CreateVPNConnection(*ec2.CreateVPNConnectionInput) (*ec2.CreateVPNConnectionOutput, error)

	CreateVPNConnectionRoute(*ec2.CreateVPNConnectionRouteInput) (*ec2.CreateVPNConnectionRouteOutput, error)

	CreateVPNGateway(*ec2.CreateVPNGatewayInput) (*ec2.CreateVPNGatewayOutput, error)

	CreateVolume(*ec2.CreateVolumeInput) (*ec2.Volume, error)

	DeleteCustomerGateway(*ec2.DeleteCustomerGatewayInput) (*ec2.DeleteCustomerGatewayOutput, error)

	DeleteDHCPOptions(*ec2.DeleteDHCPOptionsInput) (*ec2.DeleteDHCPOptionsOutput, error)

	DeleteInternetGateway(*ec2.DeleteInternetGatewayInput) (*ec2.DeleteInternetGatewayOutput, error)

	DeleteKeyPair(*ec2.DeleteKeyPairInput) (*ec2.DeleteKeyPairOutput, error)

	DeleteNetworkACL(*ec2.DeleteNetworkACLInput) (*ec2.DeleteNetworkACLOutput, error)

	DeleteNetworkACLEntry(*ec2.DeleteNetworkACLEntryInput) (*ec2.DeleteNetworkACLEntryOutput, error)

	DeleteNetworkInterface(*ec2.DeleteNetworkInterfaceInput) (*ec2.DeleteNetworkInterfaceOutput, error)

	DeletePlacementGroup(*ec2.DeletePlacementGroupInput) (*ec2.DeletePlacementGroupOutput, error)

	DeleteRoute(*ec2.DeleteRouteInput) (*ec2.DeleteRouteOutput, error)

	DeleteRouteTable(*ec2.DeleteRouteTableInput) (*ec2.DeleteRouteTableOutput, error)

	DeleteSecurityGroup(*ec2.DeleteSecurityGroupInput) (*ec2.DeleteSecurityGroupOutput, error)

	DeleteSnapshot(*ec2.DeleteSnapshotInput) (*ec2.DeleteSnapshotOutput, error)

	DeleteSpotDatafeedSubscription(*ec2.DeleteSpotDatafeedSubscriptionInput) (*ec2.DeleteSpotDatafeedSubscriptionOutput, error)

	DeleteSubnet(*ec2.DeleteSubnetInput) (*ec2.DeleteSubnetOutput, error)

	DeleteTags(*ec2.DeleteTagsInput) (*ec2.DeleteTagsOutput, error)

	DeleteVPC(*ec2.DeleteVPCInput) (*ec2.DeleteVPCOutput, error)

	DeleteVPCPeeringConnection(*ec2.DeleteVPCPeeringConnectionInput) (*ec2.DeleteVPCPeeringConnectionOutput, error)

	DeleteVPNConnection(*ec2.DeleteVPNConnectionInput) (*ec2.DeleteVPNConnectionOutput, error)

	DeleteVPNConnectionRoute(*ec2.DeleteVPNConnectionRouteInput) (*ec2.DeleteVPNConnectionRouteOutput, error)

	DeleteVPNGateway(*ec2.DeleteVPNGatewayInput) (*ec2.DeleteVPNGatewayOutput, error)

	DeleteVolume(*ec2.DeleteVolumeInput) (*ec2.DeleteVolumeOutput, error)

	DeregisterImage(*ec2.DeregisterImageInput) (*ec2.DeregisterImageOutput, error)

	DescribeAccountAttributes(*ec2.DescribeAccountAttributesInput) (*ec2.DescribeAccountAttributesOutput, error)

	DescribeAddresses(*ec2.DescribeAddressesInput) (*ec2.DescribeAddressesOutput, error)

	DescribeAvailabilityZones(*ec2.DescribeAvailabilityZonesInput) (*ec2.DescribeAvailabilityZonesOutput, error)

	DescribeBundleTasks(*ec2.DescribeBundleTasksInput) (*ec2.DescribeBundleTasksOutput, error)

	DescribeClassicLinkInstances(*ec2.DescribeClassicLinkInstancesInput) (*ec2.DescribeClassicLinkInstancesOutput, error)

	DescribeConversionTasks(*ec2.DescribeConversionTasksInput) (*ec2.DescribeConversionTasksOutput, error)

	DescribeCustomerGateways(*ec2.DescribeCustomerGatewaysInput) (*ec2.DescribeCustomerGatewaysOutput, error)

	DescribeDHCPOptions(*ec2.DescribeDHCPOptionsInput) (*ec2.DescribeDHCPOptionsOutput, error)

	DescribeExportTasks(*ec2.DescribeExportTasksInput) (*ec2.DescribeExportTasksOutput, error)

	DescribeImageAttribute(*ec2.DescribeImageAttributeInput) (*ec2.DescribeImageAttributeOutput, error)

	DescribeImages(*ec2.DescribeImagesInput) (*ec2.DescribeImagesOutput, error)

	DescribeImportImageTasks(*ec2.DescribeImportImageTasksInput) (*ec2.DescribeImportImageTasksOutput, error)

	DescribeImportSnapshotTasks(*ec2.DescribeImportSnapshotTasksInput) (*ec2.DescribeImportSnapshotTasksOutput, error)

	DescribeInstanceAttribute(*ec2.DescribeInstanceAttributeInput) (*ec2.DescribeInstanceAttributeOutput, error)

	DescribeInstanceStatus(*ec2.DescribeInstanceStatusInput) (*ec2.DescribeInstanceStatusOutput, error)

	DescribeInstances(*ec2.DescribeInstancesInput) (*ec2.DescribeInstancesOutput, error)

	DescribeInternetGateways(*ec2.DescribeInternetGatewaysInput) (*ec2.DescribeInternetGatewaysOutput, error)

	DescribeKeyPairs(*ec2.DescribeKeyPairsInput) (*ec2.DescribeKeyPairsOutput, error)

	DescribeNetworkACLs(*ec2.DescribeNetworkACLsInput) (*ec2.DescribeNetworkACLsOutput, error)

	DescribeNetworkInterfaceAttribute(*ec2.DescribeNetworkInterfaceAttributeInput) (*ec2.DescribeNetworkInterfaceAttributeOutput, error)

	DescribeNetworkInterfaces(*ec2.DescribeNetworkInterfacesInput) (*ec2.DescribeNetworkInterfacesOutput, error)

	DescribePlacementGroups(*ec2.DescribePlacementGroupsInput) (*ec2.DescribePlacementGroupsOutput, error)

	DescribeRegions(*ec2.DescribeRegionsInput) (*ec2.DescribeRegionsOutput, error)

	DescribeReservedInstances(*ec2.DescribeReservedInstancesInput) (*ec2.DescribeReservedInstancesOutput, error)

	DescribeReservedInstancesListings(*ec2.DescribeReservedInstancesListingsInput) (*ec2.DescribeReservedInstancesListingsOutput, error)

	DescribeReservedInstancesModifications(*ec2.DescribeReservedInstancesModificationsInput) (*ec2.DescribeReservedInstancesModificationsOutput, error)

	DescribeReservedInstancesOfferings(*ec2.DescribeReservedInstancesOfferingsInput) (*ec2.DescribeReservedInstancesOfferingsOutput, error)

	DescribeRouteTables(*ec2.DescribeRouteTablesInput) (*ec2.DescribeRouteTablesOutput, error)

	DescribeSecurityGroups(*ec2.DescribeSecurityGroupsInput) (*ec2.DescribeSecurityGroupsOutput, error)

	DescribeSnapshotAttribute(*ec2.DescribeSnapshotAttributeInput) (*ec2.DescribeSnapshotAttributeOutput, error)

	DescribeSnapshots(*ec2.DescribeSnapshotsInput) (*ec2.DescribeSnapshotsOutput, error)

	DescribeSpotDatafeedSubscription(*ec2.DescribeSpotDatafeedSubscriptionInput) (*ec2.DescribeSpotDatafeedSubscriptionOutput, error)

	DescribeSpotInstanceRequests(*ec2.DescribeSpotInstanceRequestsInput) (*ec2.DescribeSpotInstanceRequestsOutput, error)

	DescribeSpotPriceHistory(*ec2.DescribeSpotPriceHistoryInput) (*ec2.DescribeSpotPriceHistoryOutput, error)

	DescribeSubnets(*ec2.DescribeSubnetsInput) (*ec2.DescribeSubnetsOutput, error)

	DescribeTags(*ec2.DescribeTagsInput) (*ec2.DescribeTagsOutput, error)

	DescribeVPCAttribute(*ec2.DescribeVPCAttributeInput) (*ec2.DescribeVPCAttributeOutput, error)

	DescribeVPCClassicLink(*ec2.DescribeVPCClassicLinkInput) (*ec2.DescribeVPCClassicLinkOutput, error)

	DescribeVPCPeeringConnections(*ec2.DescribeVPCPeeringConnectionsInput) (*ec2.DescribeVPCPeeringConnectionsOutput, error)

	DescribeVPCs(*ec2.DescribeVPCsInput) (*ec2.DescribeVPCsOutput, error)

	DescribeVPNConnections(*ec2.DescribeVPNConnectionsInput) (*ec2.DescribeVPNConnectionsOutput, error)

	DescribeVPNGateways(*ec2.DescribeVPNGatewaysInput) (*ec2.DescribeVPNGatewaysOutput, error)

	DescribeVolumeAttribute(*ec2.DescribeVolumeAttributeInput) (*ec2.DescribeVolumeAttributeOutput, error)

	DescribeVolumeStatus(*ec2.DescribeVolumeStatusInput) (*ec2.DescribeVolumeStatusOutput, error)

	DescribeVolumes(*ec2.DescribeVolumesInput) (*ec2.DescribeVolumesOutput, error)

	DetachClassicLinkVPC(*ec2.DetachClassicLinkVPCInput) (*ec2.DetachClassicLinkVPCOutput, error)

	DetachInternetGateway(*ec2.DetachInternetGatewayInput) (*ec2.DetachInternetGatewayOutput, error)

	DetachNetworkInterface(*ec2.DetachNetworkInterfaceInput) (*ec2.DetachNetworkInterfaceOutput, error)

	DetachVPNGateway(*ec2.DetachVPNGatewayInput) (*ec2.DetachVPNGatewayOutput, error)

	DetachVolume(*ec2.DetachVolumeInput) (*ec2.VolumeAttachment, error)

	DisableVGWRoutePropagation(*ec2.DisableVGWRoutePropagationInput) (*ec2.DisableVGWRoutePropagationOutput, error)

	DisableVPCClassicLink(*ec2.DisableVPCClassicLinkInput) (*ec2.DisableVPCClassicLinkOutput, error)

	DisassociateAddress(*ec2.DisassociateAddressInput) (*ec2.DisassociateAddressOutput, error)

	DisassociateRouteTable(*ec2.DisassociateRouteTableInput) (*ec2.DisassociateRouteTableOutput, error)

	EnableVGWRoutePropagation(*ec2.EnableVGWRoutePropagationInput) (*ec2.EnableVGWRoutePropagationOutput, error)

	EnableVPCClassicLink(*ec2.EnableVPCClassicLinkInput) (*ec2.EnableVPCClassicLinkOutput, error)

	EnableVolumeIO(*ec2.EnableVolumeIOInput) (*ec2.EnableVolumeIOOutput, error)

	GetConsoleOutput(*ec2.GetConsoleOutputInput) (*ec2.GetConsoleOutputOutput, error)

	GetPasswordData(*ec2.GetPasswordDataInput) (*ec2.GetPasswordDataOutput, error)

	ImportImage(*ec2.ImportImageInput) (*ec2.ImportImageOutput, error)

	ImportInstance(*ec2.ImportInstanceInput) (*ec2.ImportInstanceOutput, error)

	ImportKeyPair(*ec2.ImportKeyPairInput) (*ec2.ImportKeyPairOutput, error)

	ImportSnapshot(*ec2.ImportSnapshotInput) (*ec2.ImportSnapshotOutput, error)

	ImportVolume(*ec2.ImportVolumeInput) (*ec2.ImportVolumeOutput, error)

	ModifyImageAttribute(*ec2.ModifyImageAttributeInput) (*ec2.ModifyImageAttributeOutput, error)

	ModifyInstanceAttribute(*ec2.ModifyInstanceAttributeInput) (*ec2.ModifyInstanceAttributeOutput, error)

	ModifyNetworkInterfaceAttribute(*ec2.ModifyNetworkInterfaceAttributeInput) (*ec2.ModifyNetworkInterfaceAttributeOutput, error)

	ModifyReservedInstances(*ec2.ModifyReservedInstancesInput) (*ec2.ModifyReservedInstancesOutput, error)

	ModifySnapshotAttribute(*ec2.ModifySnapshotAttributeInput) (*ec2.ModifySnapshotAttributeOutput, error)

	ModifySubnetAttribute(*ec2.ModifySubnetAttributeInput) (*ec2.ModifySubnetAttributeOutput, error)

	ModifyVPCAttribute(*ec2.ModifyVPCAttributeInput) (*ec2.ModifyVPCAttributeOutput, error)

	ModifyVolumeAttribute(*ec2.ModifyVolumeAttributeInput) (*ec2.ModifyVolumeAttributeOutput, error)

	MonitorInstances(*ec2.MonitorInstancesInput) (*ec2.MonitorInstancesOutput, error)

	PurchaseReservedInstancesOffering(*ec2.PurchaseReservedInstancesOfferingInput) (*ec2.PurchaseReservedInstancesOfferingOutput, error)

	RebootInstances(*ec2.RebootInstancesInput) (*ec2.RebootInstancesOutput, error)

	RegisterImage(*ec2.RegisterImageInput) (*ec2.RegisterImageOutput, error)

	RejectVPCPeeringConnection(*ec2.RejectVPCPeeringConnectionInput) (*ec2.RejectVPCPeeringConnectionOutput, error)

	ReleaseAddress(*ec2.ReleaseAddressInput) (*ec2.ReleaseAddressOutput, error)

	ReplaceNetworkACLAssociation(*ec2.ReplaceNetworkACLAssociationInput) (*ec2.ReplaceNetworkACLAssociationOutput, error)

	ReplaceNetworkACLEntry(*ec2.ReplaceNetworkACLEntryInput) (*ec2.ReplaceNetworkACLEntryOutput, error)

	ReplaceRoute(*ec2.ReplaceRouteInput) (*ec2.ReplaceRouteOutput, error)

	ReplaceRouteTableAssociation(*ec2.ReplaceRouteTableAssociationInput) (*ec2.ReplaceRouteTableAssociationOutput, error)

	ReportInstanceStatus(*ec2.ReportInstanceStatusInput) (*ec2.ReportInstanceStatusOutput, error)

	RequestSpotInstances(*ec2.RequestSpotInstancesInput) (*ec2.RequestSpotInstancesOutput, error)

	ResetImageAttribute(*ec2.ResetImageAttributeInput) (*ec2.ResetImageAttributeOutput, error)

	ResetInstanceAttribute(*ec2.ResetInstanceAttributeInput) (*ec2.ResetInstanceAttributeOutput, error)

	ResetNetworkInterfaceAttribute(*ec2.ResetNetworkInterfaceAttributeInput) (*ec2.ResetNetworkInterfaceAttributeOutput, error)

	ResetSnapshotAttribute(*ec2.ResetSnapshotAttributeInput) (*ec2.ResetSnapshotAttributeOutput, error)

	RevokeSecurityGroupEgress(*ec2.RevokeSecurityGroupEgressInput) (*ec2.RevokeSecurityGroupEgressOutput, error)

	RevokeSecurityGroupIngress(*ec2.RevokeSecurityGroupIngressInput) (*ec2.RevokeSecurityGroupIngressOutput, error)

	RunInstances(*ec2.RunInstancesInput) (*ec2.Reservation, error)

	StartInstances(*ec2.StartInstancesInput) (*ec2.StartInstancesOutput, error)

	StopInstances(*ec2.StopInstancesInput) (*ec2.StopInstancesOutput, error)

	TerminateInstances(*ec2.TerminateInstancesInput) (*ec2.TerminateInstancesOutput, error)

	UnassignPrivateIPAddresses(*ec2.UnassignPrivateIPAddressesInput) (*ec2.UnassignPrivateIPAddressesOutput, error)

	UnmonitorInstances(*ec2.UnmonitorInstancesInput) (*ec2.UnmonitorInstancesOutput, error)
}
