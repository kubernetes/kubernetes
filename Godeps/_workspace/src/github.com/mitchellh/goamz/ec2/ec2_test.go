package ec2_test

import (
	"testing"

	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"
	"github.com/mitchellh/goamz/testutil"
	. "github.com/motain/gocheck"
)

func Test(t *testing.T) {
	TestingT(t)
}

var _ = Suite(&S{})

type S struct {
	ec2 *ec2.EC2
}

var testServer = testutil.NewHTTPServer()

func (s *S) SetUpSuite(c *C) {
	testServer.Start()
	auth := aws.Auth{"abc", "123", ""}
	s.ec2 = ec2.NewWithClient(
		auth,
		aws.Region{EC2Endpoint: testServer.URL},
		testutil.DefaultClient,
	)
}

func (s *S) TearDownTest(c *C) {
	testServer.Flush()
}

func (s *S) TestRunInstancesErrorDump(c *C) {
	testServer.Response(400, nil, ErrorDump)

	options := ec2.RunInstances{
		ImageId:      "ami-a6f504cf", // Ubuntu Maverick, i386, instance store
		InstanceType: "t1.micro",     // Doesn't work with micro, results in 400.
	}

	msg := `AMIs with an instance-store root device are not supported for the instance type 't1\.micro'\.`

	resp, err := s.ec2.RunInstances(&options)

	testServer.WaitRequest()

	c.Assert(resp, IsNil)
	c.Assert(err, ErrorMatches, msg+` \(UnsupportedOperation\)`)

	ec2err, ok := err.(*ec2.Error)
	c.Assert(ok, Equals, true)
	c.Assert(ec2err.StatusCode, Equals, 400)
	c.Assert(ec2err.Code, Equals, "UnsupportedOperation")
	c.Assert(ec2err.Message, Matches, msg)
	c.Assert(ec2err.RequestId, Equals, "0503f4e9-bbd6-483c-b54f-c4ae9f3b30f4")
}

func (s *S) TestRequestSpotInstancesErrorDump(c *C) {
	testServer.Response(400, nil, ErrorDump)

	options := ec2.RequestSpotInstances{
		SpotPrice:    "0.01",
		ImageId:      "ami-a6f504cf", // Ubuntu Maverick, i386, instance store
		InstanceType: "t1.micro",     // Doesn't work with micro, results in 400.
	}

	msg := `AMIs with an instance-store root device are not supported for the instance type 't1\.micro'\.`

	resp, err := s.ec2.RequestSpotInstances(&options)

	testServer.WaitRequest()

	c.Assert(resp, IsNil)
	c.Assert(err, ErrorMatches, msg+` \(UnsupportedOperation\)`)

	ec2err, ok := err.(*ec2.Error)
	c.Assert(ok, Equals, true)
	c.Assert(ec2err.StatusCode, Equals, 400)
	c.Assert(ec2err.Code, Equals, "UnsupportedOperation")
	c.Assert(ec2err.Message, Matches, msg)
	c.Assert(ec2err.RequestId, Equals, "0503f4e9-bbd6-483c-b54f-c4ae9f3b30f4")
}

func (s *S) TestRunInstancesErrorWithoutXML(c *C) {
	testServer.Responses(5, 500, nil, "")
	options := ec2.RunInstances{ImageId: "image-id"}

	resp, err := s.ec2.RunInstances(&options)

	testServer.WaitRequest()

	c.Assert(resp, IsNil)
	c.Assert(err, ErrorMatches, "500 Internal Server Error")

	ec2err, ok := err.(*ec2.Error)
	c.Assert(ok, Equals, true)
	c.Assert(ec2err.StatusCode, Equals, 500)
	c.Assert(ec2err.Code, Equals, "")
	c.Assert(ec2err.Message, Equals, "500 Internal Server Error")
	c.Assert(ec2err.RequestId, Equals, "")
}

func (s *S) TestRequestSpotInstancesErrorWithoutXML(c *C) {
	testServer.Responses(5, 500, nil, "")
	options := ec2.RequestSpotInstances{SpotPrice: "spot-price", ImageId: "image-id"}

	resp, err := s.ec2.RequestSpotInstances(&options)

	testServer.WaitRequest()

	c.Assert(resp, IsNil)
	c.Assert(err, ErrorMatches, "500 Internal Server Error")

	ec2err, ok := err.(*ec2.Error)
	c.Assert(ok, Equals, true)
	c.Assert(ec2err.StatusCode, Equals, 500)
	c.Assert(ec2err.Code, Equals, "")
	c.Assert(ec2err.Message, Equals, "500 Internal Server Error")
	c.Assert(ec2err.RequestId, Equals, "")
}

func (s *S) TestRunInstancesExample(c *C) {
	testServer.Response(200, nil, RunInstancesExample)

	options := ec2.RunInstances{
		KeyName:               "my-keys",
		ImageId:               "image-id",
		InstanceType:          "inst-type",
		SecurityGroups:        []ec2.SecurityGroup{{Name: "g1"}, {Id: "g2"}, {Name: "g3"}, {Id: "g4"}},
		UserData:              []byte("1234"),
		KernelId:              "kernel-id",
		RamdiskId:             "ramdisk-id",
		AvailZone:             "zone",
		PlacementGroupName:    "group",
		Monitoring:            true,
		SubnetId:              "subnet-id",
		DisableAPITermination: true,
		ShutdownBehavior:      "terminate",
		PrivateIPAddress:      "10.0.0.25",
		BlockDevices: []ec2.BlockDeviceMapping{
			{DeviceName: "/dev/sdb", VirtualName: "ephemeral0"},
			{DeviceName: "/dev/sdc", SnapshotId: "snap-a08912c9", DeleteOnTermination: true},
		},
	}
	resp, err := s.ec2.RunInstances(&options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"RunInstances"})
	c.Assert(req.Form["ImageId"], DeepEquals, []string{"image-id"})
	c.Assert(req.Form["MinCount"], DeepEquals, []string{"1"})
	c.Assert(req.Form["MaxCount"], DeepEquals, []string{"1"})
	c.Assert(req.Form["KeyName"], DeepEquals, []string{"my-keys"})
	c.Assert(req.Form["InstanceType"], DeepEquals, []string{"inst-type"})
	c.Assert(req.Form["SecurityGroup.1"], DeepEquals, []string{"g1"})
	c.Assert(req.Form["SecurityGroup.2"], DeepEquals, []string{"g3"})
	c.Assert(req.Form["SecurityGroupId.1"], DeepEquals, []string{"g2"})
	c.Assert(req.Form["SecurityGroupId.2"], DeepEquals, []string{"g4"})
	c.Assert(req.Form["UserData"], DeepEquals, []string{"MTIzNA=="})
	c.Assert(req.Form["KernelId"], DeepEquals, []string{"kernel-id"})
	c.Assert(req.Form["RamdiskId"], DeepEquals, []string{"ramdisk-id"})
	c.Assert(req.Form["Placement.AvailabilityZone"], DeepEquals, []string{"zone"})
	c.Assert(req.Form["Placement.GroupName"], DeepEquals, []string{"group"})
	c.Assert(req.Form["Monitoring.Enabled"], DeepEquals, []string{"true"})
	c.Assert(req.Form["SubnetId"], DeepEquals, []string{"subnet-id"})
	c.Assert(req.Form["DisableApiTermination"], DeepEquals, []string{"true"})
	c.Assert(req.Form["InstanceInitiatedShutdownBehavior"], DeepEquals, []string{"terminate"})
	c.Assert(req.Form["PrivateIpAddress"], DeepEquals, []string{"10.0.0.25"})
	c.Assert(req.Form["BlockDeviceMapping.1.DeviceName"], DeepEquals, []string{"/dev/sdb"})
	c.Assert(req.Form["BlockDeviceMapping.1.VirtualName"], DeepEquals, []string{"ephemeral0"})
	c.Assert(req.Form["BlockDeviceMapping.2.Ebs.SnapshotId"], DeepEquals, []string{"snap-a08912c9"})
	c.Assert(req.Form["BlockDeviceMapping.2.Ebs.DeleteOnTermination"], DeepEquals, []string{"true"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.ReservationId, Equals, "r-47a5402e")
	c.Assert(resp.OwnerId, Equals, "999988887777")
	c.Assert(resp.SecurityGroups, DeepEquals, []ec2.SecurityGroup{{Name: "default", Id: "sg-67ad940e"}})
	c.Assert(resp.Instances, HasLen, 3)

	i0 := resp.Instances[0]
	c.Assert(i0.InstanceId, Equals, "i-2ba64342")
	c.Assert(i0.InstanceType, Equals, "m1.small")
	c.Assert(i0.ImageId, Equals, "ami-60a54009")
	c.Assert(i0.Monitoring, Equals, "enabled")
	c.Assert(i0.KeyName, Equals, "example-key-name")
	c.Assert(i0.AMILaunchIndex, Equals, 0)
	c.Assert(i0.VirtType, Equals, "paravirtual")
	c.Assert(i0.Hypervisor, Equals, "xen")

	i1 := resp.Instances[1]
	c.Assert(i1.InstanceId, Equals, "i-2bc64242")
	c.Assert(i1.InstanceType, Equals, "m1.small")
	c.Assert(i1.ImageId, Equals, "ami-60a54009")
	c.Assert(i1.Monitoring, Equals, "enabled")
	c.Assert(i1.KeyName, Equals, "example-key-name")
	c.Assert(i1.AMILaunchIndex, Equals, 1)
	c.Assert(i1.VirtType, Equals, "paravirtual")
	c.Assert(i1.Hypervisor, Equals, "xen")

	i2 := resp.Instances[2]
	c.Assert(i2.InstanceId, Equals, "i-2be64332")
	c.Assert(i2.InstanceType, Equals, "m1.small")
	c.Assert(i2.ImageId, Equals, "ami-60a54009")
	c.Assert(i2.Monitoring, Equals, "enabled")
	c.Assert(i2.KeyName, Equals, "example-key-name")
	c.Assert(i2.AMILaunchIndex, Equals, 2)
	c.Assert(i2.VirtType, Equals, "paravirtual")
	c.Assert(i2.Hypervisor, Equals, "xen")
}

func (s *S) TestRequestSpotInstancesExample(c *C) {
	testServer.Response(200, nil, RequestSpotInstancesExample)

	options := ec2.RequestSpotInstances{
		SpotPrice:          "0.5",
		KeyName:            "my-keys",
		ImageId:            "image-id",
		InstanceType:       "inst-type",
		SecurityGroups:     []ec2.SecurityGroup{{Name: "g1"}, {Id: "g2"}, {Name: "g3"}, {Id: "g4"}},
		UserData:           []byte("1234"),
		KernelId:           "kernel-id",
		RamdiskId:          "ramdisk-id",
		AvailZone:          "zone",
		PlacementGroupName: "group",
		Monitoring:         true,
		SubnetId:           "subnet-id",
		PrivateIPAddress:   "10.0.0.25",
		BlockDevices: []ec2.BlockDeviceMapping{
			{DeviceName: "/dev/sdb", VirtualName: "ephemeral0"},
			{DeviceName: "/dev/sdc", SnapshotId: "snap-a08912c9", DeleteOnTermination: true},
		},
	}
	resp, err := s.ec2.RequestSpotInstances(&options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"RequestSpotInstances"})
	c.Assert(req.Form["SpotPrice"], DeepEquals, []string{"0.5"})
	c.Assert(req.Form["LaunchSpecification.ImageId"], DeepEquals, []string{"image-id"})
	c.Assert(req.Form["LaunchSpecification.KeyName"], DeepEquals, []string{"my-keys"})
	c.Assert(req.Form["LaunchSpecification.InstanceType"], DeepEquals, []string{"inst-type"})
	c.Assert(req.Form["LaunchSpecification.SecurityGroup.1"], DeepEquals, []string{"g1"})
	c.Assert(req.Form["LaunchSpecification.SecurityGroup.2"], DeepEquals, []string{"g3"})
	c.Assert(req.Form["LaunchSpecification.SecurityGroupId.1"], DeepEquals, []string{"g2"})
	c.Assert(req.Form["LaunchSpecification.SecurityGroupId.2"], DeepEquals, []string{"g4"})
	c.Assert(req.Form["LaunchSpecification.UserData"], DeepEquals, []string{"MTIzNA=="})
	c.Assert(req.Form["LaunchSpecification.KernelId"], DeepEquals, []string{"kernel-id"})
	c.Assert(req.Form["LaunchSpecification.RamdiskId"], DeepEquals, []string{"ramdisk-id"})
	c.Assert(req.Form["LaunchSpecification.Placement.AvailabilityZone"], DeepEquals, []string{"zone"})
	c.Assert(req.Form["LaunchSpecification.Placement.GroupName"], DeepEquals, []string{"group"})
	c.Assert(req.Form["LaunchSpecification.Monitoring.Enabled"], DeepEquals, []string{"true"})
	c.Assert(req.Form["LaunchSpecification.SubnetId"], DeepEquals, []string{"subnet-id"})
	c.Assert(req.Form["LaunchSpecification.PrivateIpAddress"], DeepEquals, []string{"10.0.0.25"})
	c.Assert(req.Form["LaunchSpecification.BlockDeviceMapping.1.DeviceName"], DeepEquals, []string{"/dev/sdb"})
	c.Assert(req.Form["LaunchSpecification.BlockDeviceMapping.1.VirtualName"], DeepEquals, []string{"ephemeral0"})
	c.Assert(req.Form["LaunchSpecification.BlockDeviceMapping.2.Ebs.SnapshotId"], DeepEquals, []string{"snap-a08912c9"})
	c.Assert(req.Form["LaunchSpecification.BlockDeviceMapping.2.Ebs.DeleteOnTermination"], DeepEquals, []string{"true"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.SpotRequestResults[0].SpotRequestId, Equals, "sir-1a2b3c4d")
	c.Assert(resp.SpotRequestResults[0].SpotPrice, Equals, "0.5")
	c.Assert(resp.SpotRequestResults[0].State, Equals, "open")
	c.Assert(resp.SpotRequestResults[0].SpotLaunchSpec.ImageId, Equals, "ami-1a2b3c4d")
	c.Assert(resp.SpotRequestResults[0].Status.Code, Equals, "pending-evaluation")
	c.Assert(resp.SpotRequestResults[0].Status.UpdateTime, Equals, "2008-05-07T12:51:50.000Z")
	c.Assert(resp.SpotRequestResults[0].Status.Message, Equals, "Your Spot request has been submitted for review, and is pending evaluation.")
}

func (s *S) TestCancelSpotRequestsExample(c *C) {
	testServer.Response(200, nil, CancelSpotRequestsExample)

	resp, err := s.ec2.CancelSpotRequests([]string{"s-1", "s-2"})

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CancelSpotInstanceRequests"})
	c.Assert(req.Form["SpotInstanceRequestId.1"], DeepEquals, []string{"s-1"})
	c.Assert(req.Form["SpotInstanceRequestId.2"], DeepEquals, []string{"s-2"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.CancelSpotRequestResults[0].SpotRequestId, Equals, "sir-1a2b3c4d")
	c.Assert(resp.CancelSpotRequestResults[0].State, Equals, "cancelled")
}

func (s *S) TestTerminateInstancesExample(c *C) {
	testServer.Response(200, nil, TerminateInstancesExample)

	resp, err := s.ec2.TerminateInstances([]string{"i-1", "i-2"})

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"TerminateInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-1"})
	c.Assert(req.Form["InstanceId.2"], DeepEquals, []string{"i-2"})
	c.Assert(req.Form["UserData"], IsNil)
	c.Assert(req.Form["KernelId"], IsNil)
	c.Assert(req.Form["RamdiskId"], IsNil)
	c.Assert(req.Form["Placement.AvailabilityZone"], IsNil)
	c.Assert(req.Form["Placement.GroupName"], IsNil)
	c.Assert(req.Form["Monitoring.Enabled"], IsNil)
	c.Assert(req.Form["SubnetId"], IsNil)
	c.Assert(req.Form["DisableApiTermination"], IsNil)
	c.Assert(req.Form["InstanceInitiatedShutdownBehavior"], IsNil)
	c.Assert(req.Form["PrivateIpAddress"], IsNil)

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.StateChanges, HasLen, 1)
	c.Assert(resp.StateChanges[0].InstanceId, Equals, "i-3ea74257")
	c.Assert(resp.StateChanges[0].CurrentState.Code, Equals, 32)
	c.Assert(resp.StateChanges[0].CurrentState.Name, Equals, "shutting-down")
	c.Assert(resp.StateChanges[0].PreviousState.Code, Equals, 16)
	c.Assert(resp.StateChanges[0].PreviousState.Name, Equals, "running")
}

func (s *S) TestDescribeSpotRequestsExample(c *C) {
	testServer.Response(200, nil, DescribeSpotRequestsExample)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.DescribeSpotRequests([]string{"s-1", "s-2"}, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeSpotInstanceRequests"})
	c.Assert(req.Form["SpotInstanceRequestId.1"], DeepEquals, []string{"s-1"})
	c.Assert(req.Form["SpotInstanceRequestId.2"], DeepEquals, []string{"s-2"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "b1719f2a-5334-4479-b2f1-26926EXAMPLE")
	c.Assert(resp.SpotRequestResults[0].SpotRequestId, Equals, "sir-1a2b3c4d")
	c.Assert(resp.SpotRequestResults[0].State, Equals, "active")
	c.Assert(resp.SpotRequestResults[0].SpotPrice, Equals, "0.5")
	c.Assert(resp.SpotRequestResults[0].SpotLaunchSpec.ImageId, Equals, "ami-1a2b3c4d")
	c.Assert(resp.SpotRequestResults[0].Status.Code, Equals, "fulfilled")
	c.Assert(resp.SpotRequestResults[0].Status.UpdateTime, Equals, "2008-05-07T12:51:50.000Z")
	c.Assert(resp.SpotRequestResults[0].Status.Message, Equals, "Your Spot request is fulfilled.")
}

func (s *S) TestDescribeInstancesExample1(c *C) {
	testServer.Response(200, nil, DescribeInstancesExample1)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.Instances([]string{"i-1", "i-2"}, nil)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-1"})
	c.Assert(req.Form["InstanceId.2"], DeepEquals, []string{"i-2"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "98e3c9a4-848c-4d6d-8e8a-b1bdEXAMPLE")
	c.Assert(resp.Reservations, HasLen, 2)

	r0 := resp.Reservations[0]
	c.Assert(r0.ReservationId, Equals, "r-b27e30d9")
	c.Assert(r0.OwnerId, Equals, "999988887777")
	c.Assert(r0.RequesterId, Equals, "854251627541")
	c.Assert(r0.SecurityGroups, DeepEquals, []ec2.SecurityGroup{{Name: "default", Id: "sg-67ad940e"}})
	c.Assert(r0.Instances, HasLen, 1)

	r0i := r0.Instances[0]
	c.Assert(r0i.InstanceId, Equals, "i-c5cd56af")
	c.Assert(r0i.PrivateDNSName, Equals, "domU-12-31-39-10-56-34.compute-1.internal")
	c.Assert(r0i.DNSName, Equals, "ec2-174-129-165-232.compute-1.amazonaws.com")
	c.Assert(r0i.AvailZone, Equals, "us-east-1b")
}

func (s *S) TestDescribeInstancesExample2(c *C) {
	testServer.Response(200, nil, DescribeInstancesExample2)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.Instances([]string{"i-1", "i-2"}, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-1"})
	c.Assert(req.Form["InstanceId.2"], DeepEquals, []string{"i-2"})
	c.Assert(req.Form["Filter.1.Name"], DeepEquals, []string{"key1"})
	c.Assert(req.Form["Filter.1.Value.1"], DeepEquals, []string{"value1"})
	c.Assert(req.Form["Filter.1.Value.2"], IsNil)
	c.Assert(req.Form["Filter.2.Name"], DeepEquals, []string{"key2"})
	c.Assert(req.Form["Filter.2.Value.1"], DeepEquals, []string{"value2"})
	c.Assert(req.Form["Filter.2.Value.2"], DeepEquals, []string{"value3"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.Reservations, HasLen, 1)

	r0 := resp.Reservations[0]
	r0i := r0.Instances[0]
	c.Assert(r0i.State.Code, Equals, 16)
	c.Assert(r0i.State.Name, Equals, "running")

	r0t0 := r0i.Tags[0]
	r0t1 := r0i.Tags[1]
	c.Assert(r0t0.Key, Equals, "webserver")
	c.Assert(r0t0.Value, Equals, "")
	c.Assert(r0t1.Key, Equals, "stack")
	c.Assert(r0t1.Value, Equals, "Production")
}

func (s *S) TestCreateImageExample(c *C) {
	testServer.Response(200, nil, CreateImageExample)

	options := &ec2.CreateImage{
		InstanceId:  "i-123456",
		Name:        "foo",
		Description: "Test CreateImage",
		NoReboot:    true,
		BlockDevices: []ec2.BlockDeviceMapping{
			{DeviceName: "/dev/sdb", VirtualName: "ephemeral0"},
			{DeviceName: "/dev/sdc", SnapshotId: "snap-a08912c9", DeleteOnTermination: true},
		},
	}

	resp, err := s.ec2.CreateImage(options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CreateImage"})
	c.Assert(req.Form["InstanceId"], DeepEquals, []string{options.InstanceId})
	c.Assert(req.Form["Name"], DeepEquals, []string{options.Name})
	c.Assert(req.Form["Description"], DeepEquals, []string{options.Description})
	c.Assert(req.Form["NoReboot"], DeepEquals, []string{"true"})
	c.Assert(req.Form["BlockDeviceMapping.1.DeviceName"], DeepEquals, []string{"/dev/sdb"})
	c.Assert(req.Form["BlockDeviceMapping.1.VirtualName"], DeepEquals, []string{"ephemeral0"})
	c.Assert(req.Form["BlockDeviceMapping.2.DeviceName"], DeepEquals, []string{"/dev/sdc"})
	c.Assert(req.Form["BlockDeviceMapping.2.Ebs.SnapshotId"], DeepEquals, []string{"snap-a08912c9"})
	c.Assert(req.Form["BlockDeviceMapping.2.Ebs.DeleteOnTermination"], DeepEquals, []string{"true"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.ImageId, Equals, "ami-4fa54026")
}

func (s *S) TestDescribeImagesExample(c *C) {
	testServer.Response(200, nil, DescribeImagesExample)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.Images([]string{"ami-1", "ami-2"}, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeImages"})
	c.Assert(req.Form["ImageId.1"], DeepEquals, []string{"ami-1"})
	c.Assert(req.Form["ImageId.2"], DeepEquals, []string{"ami-2"})
	c.Assert(req.Form["Filter.1.Name"], DeepEquals, []string{"key1"})
	c.Assert(req.Form["Filter.1.Value.1"], DeepEquals, []string{"value1"})
	c.Assert(req.Form["Filter.1.Value.2"], IsNil)
	c.Assert(req.Form["Filter.2.Name"], DeepEquals, []string{"key2"})
	c.Assert(req.Form["Filter.2.Value.1"], DeepEquals, []string{"value2"})
	c.Assert(req.Form["Filter.2.Value.2"], DeepEquals, []string{"value3"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "4a4a27a2-2e7c-475d-b35b-ca822EXAMPLE")
	c.Assert(resp.Images, HasLen, 1)

	i0 := resp.Images[0]
	c.Assert(i0.Id, Equals, "ami-a2469acf")
	c.Assert(i0.Type, Equals, "machine")
	c.Assert(i0.Name, Equals, "example-marketplace-amzn-ami.1")
	c.Assert(i0.Description, Equals, "Amazon Linux AMI i386 EBS")
	c.Assert(i0.Location, Equals, "aws-marketplace/example-marketplace-amzn-ami.1")
	c.Assert(i0.State, Equals, "available")
	c.Assert(i0.Public, Equals, true)
	c.Assert(i0.OwnerId, Equals, "123456789999")
	c.Assert(i0.OwnerAlias, Equals, "aws-marketplace")
	c.Assert(i0.Architecture, Equals, "i386")
	c.Assert(i0.KernelId, Equals, "aki-805ea7e9")
	c.Assert(i0.RootDeviceType, Equals, "ebs")
	c.Assert(i0.RootDeviceName, Equals, "/dev/sda1")
	c.Assert(i0.VirtualizationType, Equals, "paravirtual")
	c.Assert(i0.Hypervisor, Equals, "xen")

	c.Assert(i0.BlockDevices, HasLen, 1)
	c.Assert(i0.BlockDevices[0].DeviceName, Equals, "/dev/sda1")
	c.Assert(i0.BlockDevices[0].SnapshotId, Equals, "snap-787e9403")
	c.Assert(i0.BlockDevices[0].VolumeSize, Equals, int64(8))
	c.Assert(i0.BlockDevices[0].DeleteOnTermination, Equals, true)

	testServer.Response(200, nil, DescribeImagesExample)
	resp2, err := s.ec2.ImagesByOwners([]string{"ami-1", "ami-2"}, []string{"123456789999", "id2"}, filter)

	req2 := testServer.WaitRequest()
	c.Assert(req2.Form["Action"], DeepEquals, []string{"DescribeImages"})
	c.Assert(req2.Form["ImageId.1"], DeepEquals, []string{"ami-1"})
	c.Assert(req2.Form["ImageId.2"], DeepEquals, []string{"ami-2"})
	c.Assert(req2.Form["Owner.1"], DeepEquals, []string{"123456789999"})
	c.Assert(req2.Form["Owner.2"], DeepEquals, []string{"id2"})
	c.Assert(req2.Form["Filter.1.Name"], DeepEquals, []string{"key1"})
	c.Assert(req2.Form["Filter.1.Value.1"], DeepEquals, []string{"value1"})
	c.Assert(req2.Form["Filter.1.Value.2"], IsNil)
	c.Assert(req2.Form["Filter.2.Name"], DeepEquals, []string{"key2"})
	c.Assert(req2.Form["Filter.2.Value.1"], DeepEquals, []string{"value2"})
	c.Assert(req2.Form["Filter.2.Value.2"], DeepEquals, []string{"value3"})

	c.Assert(err, IsNil)
	c.Assert(resp2.RequestId, Equals, "4a4a27a2-2e7c-475d-b35b-ca822EXAMPLE")
	c.Assert(resp2.Images, HasLen, 1)

	i1 := resp2.Images[0]
	c.Assert(i1.Id, Equals, "ami-a2469acf")
	c.Assert(i1.Type, Equals, "machine")
	c.Assert(i1.Name, Equals, "example-marketplace-amzn-ami.1")
	c.Assert(i1.Description, Equals, "Amazon Linux AMI i386 EBS")
	c.Assert(i1.Location, Equals, "aws-marketplace/example-marketplace-amzn-ami.1")
	c.Assert(i1.State, Equals, "available")
	c.Assert(i1.Public, Equals, true)
	c.Assert(i1.OwnerId, Equals, "123456789999")
	c.Assert(i1.OwnerAlias, Equals, "aws-marketplace")
	c.Assert(i1.Architecture, Equals, "i386")
	c.Assert(i1.KernelId, Equals, "aki-805ea7e9")
	c.Assert(i1.RootDeviceType, Equals, "ebs")
	c.Assert(i1.RootDeviceName, Equals, "/dev/sda1")
	c.Assert(i1.VirtualizationType, Equals, "paravirtual")
	c.Assert(i1.Hypervisor, Equals, "xen")

	c.Assert(i1.BlockDevices, HasLen, 1)
	c.Assert(i1.BlockDevices[0].DeviceName, Equals, "/dev/sda1")
	c.Assert(i1.BlockDevices[0].SnapshotId, Equals, "snap-787e9403")
	c.Assert(i1.BlockDevices[0].VolumeSize, Equals, int64(8))
	c.Assert(i1.BlockDevices[0].DeleteOnTermination, Equals, true)
}

func (s *S) TestImageAttributeExample(c *C) {
	testServer.Response(200, nil, ImageAttributeExample)

	resp, err := s.ec2.ImageAttribute("ami-61a54008", "launchPermission")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeImageAttribute"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.ImageId, Equals, "ami-61a54008")
	c.Assert(resp.Group, Equals, "all")
	c.Assert(resp.UserIds[0], Equals, "495219933132")
}

func (s *S) TestCreateSnapshotExample(c *C) {
	testServer.Response(200, nil, CreateSnapshotExample)

	resp, err := s.ec2.CreateSnapshot("vol-4d826724", "Daily Backup")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CreateSnapshot"})
	c.Assert(req.Form["VolumeId"], DeepEquals, []string{"vol-4d826724"})
	c.Assert(req.Form["Description"], DeepEquals, []string{"Daily Backup"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.Snapshot.Id, Equals, "snap-78a54011")
	c.Assert(resp.Snapshot.VolumeId, Equals, "vol-4d826724")
	c.Assert(resp.Snapshot.Status, Equals, "pending")
	c.Assert(resp.Snapshot.StartTime, Equals, "2008-05-07T12:51:50.000Z")
	c.Assert(resp.Snapshot.Progress, Equals, "60%")
	c.Assert(resp.Snapshot.OwnerId, Equals, "111122223333")
	c.Assert(resp.Snapshot.VolumeSize, Equals, "10")
	c.Assert(resp.Snapshot.Description, Equals, "Daily Backup")
}

func (s *S) TestDeleteSnapshotsExample(c *C) {
	testServer.Response(200, nil, DeleteSnapshotExample)

	resp, err := s.ec2.DeleteSnapshots([]string{"snap-78a54011"})

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DeleteSnapshot"})
	c.Assert(req.Form["SnapshotId.1"], DeepEquals, []string{"snap-78a54011"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestDescribeSnapshotsExample(c *C) {
	testServer.Response(200, nil, DescribeSnapshotsExample)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.Snapshots([]string{"snap-1", "snap-2"}, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeSnapshots"})
	c.Assert(req.Form["SnapshotId.1"], DeepEquals, []string{"snap-1"})
	c.Assert(req.Form["SnapshotId.2"], DeepEquals, []string{"snap-2"})
	c.Assert(req.Form["Filter.1.Name"], DeepEquals, []string{"key1"})
	c.Assert(req.Form["Filter.1.Value.1"], DeepEquals, []string{"value1"})
	c.Assert(req.Form["Filter.1.Value.2"], IsNil)
	c.Assert(req.Form["Filter.2.Name"], DeepEquals, []string{"key2"})
	c.Assert(req.Form["Filter.2.Value.1"], DeepEquals, []string{"value2"})
	c.Assert(req.Form["Filter.2.Value.2"], DeepEquals, []string{"value3"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.Snapshots, HasLen, 1)

	s0 := resp.Snapshots[0]
	c.Assert(s0.Id, Equals, "snap-1a2b3c4d")
	c.Assert(s0.VolumeId, Equals, "vol-8875daef")
	c.Assert(s0.VolumeSize, Equals, "15")
	c.Assert(s0.Status, Equals, "pending")
	c.Assert(s0.StartTime, Equals, "2010-07-29T04:12:01.000Z")
	c.Assert(s0.Progress, Equals, "30%")
	c.Assert(s0.OwnerId, Equals, "111122223333")
	c.Assert(s0.Description, Equals, "Daily Backup")

	c.Assert(s0.Tags, HasLen, 1)
	c.Assert(s0.Tags[0].Key, Equals, "Purpose")
	c.Assert(s0.Tags[0].Value, Equals, "demo_db_14_backup")
}

func (s *S) TestModifyImageAttributeExample(c *C) {
	testServer.Response(200, nil, ModifyImageAttributeExample)

	options := ec2.ModifyImageAttribute{
		Description: "Test Description",
	}

	resp, err := s.ec2.ModifyImageAttribute("ami-4fa54026", &options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"ModifyImageAttribute"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestModifyImageAttributeExample_complex(c *C) {
	testServer.Response(200, nil, ModifyImageAttributeExample)

	options := ec2.ModifyImageAttribute{
		AddUsers:     []string{"u1", "u2"},
		RemoveUsers:  []string{"u3"},
		AddGroups:    []string{"g1", "g3"},
		RemoveGroups: []string{"g2"},
		Description:  "Test Description",
	}

	resp, err := s.ec2.ModifyImageAttribute("ami-4fa54026", &options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"ModifyImageAttribute"})
	c.Assert(req.Form["LaunchPermission.Add.1.UserId"], DeepEquals, []string{"u1"})
	c.Assert(req.Form["LaunchPermission.Add.2.UserId"], DeepEquals, []string{"u2"})
	c.Assert(req.Form["LaunchPermission.Remove.1.UserId"], DeepEquals, []string{"u3"})
	c.Assert(req.Form["LaunchPermission.Add.1.Group"], DeepEquals, []string{"g1"})
	c.Assert(req.Form["LaunchPermission.Add.2.Group"], DeepEquals, []string{"g3"})
	c.Assert(req.Form["LaunchPermission.Remove.1.Group"], DeepEquals, []string{"g2"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestCopyImageExample(c *C) {
	testServer.Response(200, nil, CopyImageExample)

	options := ec2.CopyImage{
		SourceRegion:  "us-west-2",
		SourceImageId: "ami-1a2b3c4d",
		Description:   "Test Description",
	}

	resp, err := s.ec2.CopyImage(&options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CopyImage"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "60bc441d-fa2c-494d-b155-5d6a3EXAMPLE")
}

func (s *S) TestCreateKeyPairExample(c *C) {
	testServer.Response(200, nil, CreateKeyPairExample)

	resp, err := s.ec2.CreateKeyPair("foo")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CreateKeyPair"})
	c.Assert(req.Form["KeyName"], DeepEquals, []string{"foo"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.KeyName, Equals, "foo")
	c.Assert(resp.KeyFingerprint, Equals, "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00")
}

func (s *S) TestDeleteKeyPairExample(c *C) {
	testServer.Response(200, nil, DeleteKeyPairExample)

	resp, err := s.ec2.DeleteKeyPair("foo")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DeleteKeyPair"})
	c.Assert(req.Form["KeyName"], DeepEquals, []string{"foo"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestCreateSecurityGroupExample(c *C) {
	testServer.Response(200, nil, CreateSecurityGroupExample)

	resp, err := s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: "websrv", Description: "Web Servers"})

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"CreateSecurityGroup"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(req.Form["GroupDescription"], DeepEquals, []string{"Web Servers"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.Name, Equals, "websrv")
	c.Assert(resp.Id, Equals, "sg-67ad940e")
}

func (s *S) TestDescribeSecurityGroupsExample(c *C) {
	testServer.Response(200, nil, DescribeSecurityGroupsExample)

	resp, err := s.ec2.SecurityGroups([]ec2.SecurityGroup{{Name: "WebServers"}, {Name: "RangedPortsBySource"}}, nil)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeSecurityGroups"})
	c.Assert(req.Form["GroupName.1"], DeepEquals, []string{"WebServers"})
	c.Assert(req.Form["GroupName.2"], DeepEquals, []string{"RangedPortsBySource"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.Groups, HasLen, 2)

	g0 := resp.Groups[0]
	c.Assert(g0.OwnerId, Equals, "999988887777")
	c.Assert(g0.Name, Equals, "WebServers")
	c.Assert(g0.Id, Equals, "sg-67ad940e")
	c.Assert(g0.Description, Equals, "Web Servers")
	c.Assert(g0.IPPerms, HasLen, 1)

	g0ipp := g0.IPPerms[0]
	c.Assert(g0ipp.Protocol, Equals, "tcp")
	c.Assert(g0ipp.FromPort, Equals, 80)
	c.Assert(g0ipp.ToPort, Equals, 80)
	c.Assert(g0ipp.SourceIPs, DeepEquals, []string{"0.0.0.0/0"})

	g1 := resp.Groups[1]
	c.Assert(g1.OwnerId, Equals, "999988887777")
	c.Assert(g1.Name, Equals, "RangedPortsBySource")
	c.Assert(g1.Id, Equals, "sg-76abc467")
	c.Assert(g1.Description, Equals, "Group A")
	c.Assert(g1.IPPerms, HasLen, 1)

	g1ipp := g1.IPPerms[0]
	c.Assert(g1ipp.Protocol, Equals, "tcp")
	c.Assert(g1ipp.FromPort, Equals, 6000)
	c.Assert(g1ipp.ToPort, Equals, 7000)
	c.Assert(g1ipp.SourceIPs, IsNil)
}

func (s *S) TestDescribeSecurityGroupsExampleWithFilter(c *C) {
	testServer.Response(200, nil, DescribeSecurityGroupsExample)

	filter := ec2.NewFilter()
	filter.Add("ip-permission.protocol", "tcp")
	filter.Add("ip-permission.from-port", "22")
	filter.Add("ip-permission.to-port", "22")
	filter.Add("ip-permission.group-name", "app_server_group", "database_group")

	_, err := s.ec2.SecurityGroups(nil, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeSecurityGroups"})
	c.Assert(req.Form["Filter.1.Name"], DeepEquals, []string{"ip-permission.from-port"})
	c.Assert(req.Form["Filter.1.Value.1"], DeepEquals, []string{"22"})
	c.Assert(req.Form["Filter.2.Name"], DeepEquals, []string{"ip-permission.group-name"})
	c.Assert(req.Form["Filter.2.Value.1"], DeepEquals, []string{"app_server_group"})
	c.Assert(req.Form["Filter.2.Value.2"], DeepEquals, []string{"database_group"})
	c.Assert(req.Form["Filter.3.Name"], DeepEquals, []string{"ip-permission.protocol"})
	c.Assert(req.Form["Filter.3.Value.1"], DeepEquals, []string{"tcp"})
	c.Assert(req.Form["Filter.4.Name"], DeepEquals, []string{"ip-permission.to-port"})
	c.Assert(req.Form["Filter.4.Value.1"], DeepEquals, []string{"22"})

	c.Assert(err, IsNil)
}

func (s *S) TestDescribeSecurityGroupsDumpWithGroup(c *C) {
	testServer.Response(200, nil, DescribeSecurityGroupsDump)

	resp, err := s.ec2.SecurityGroups(nil, nil)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeSecurityGroups"})
	c.Assert(err, IsNil)
	c.Check(resp.Groups, HasLen, 1)
	c.Check(resp.Groups[0].IPPerms, HasLen, 2)

	ipp0 := resp.Groups[0].IPPerms[0]
	c.Assert(ipp0.SourceIPs, IsNil)
	c.Check(ipp0.Protocol, Equals, "icmp")
	c.Assert(ipp0.SourceGroups, HasLen, 1)
	c.Check(ipp0.SourceGroups[0].OwnerId, Equals, "12345")
	c.Check(ipp0.SourceGroups[0].Name, Equals, "default")
	c.Check(ipp0.SourceGroups[0].Id, Equals, "sg-67ad940e")

	ipp1 := resp.Groups[0].IPPerms[1]
	c.Check(ipp1.Protocol, Equals, "tcp")
	c.Assert(ipp0.SourceIPs, IsNil)
	c.Assert(ipp0.SourceGroups, HasLen, 1)
	c.Check(ipp1.SourceGroups[0].Id, Equals, "sg-76abc467")
	c.Check(ipp1.SourceGroups[0].OwnerId, Equals, "12345")
	c.Check(ipp1.SourceGroups[0].Name, Equals, "other")
}

func (s *S) TestDeleteSecurityGroupExample(c *C) {
	testServer.Response(200, nil, DeleteSecurityGroupExample)

	resp, err := s.ec2.DeleteSecurityGroup(ec2.SecurityGroup{Name: "websrv"})
	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"DeleteSecurityGroup"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(req.Form["GroupId"], IsNil)
	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestDeleteSecurityGroupExampleWithId(c *C) {
	testServer.Response(200, nil, DeleteSecurityGroupExample)

	// ignore return and error - we're only want to check the parameter handling.
	s.ec2.DeleteSecurityGroup(ec2.SecurityGroup{Id: "sg-67ad940e", Name: "ignored"})
	req := testServer.WaitRequest()

	c.Assert(req.Form["GroupName"], IsNil)
	c.Assert(req.Form["GroupId"], DeepEquals, []string{"sg-67ad940e"})
}

func (s *S) TestAuthorizeSecurityGroupExample1(c *C) {
	testServer.Response(200, nil, AuthorizeSecurityGroupIngressExample)

	perms := []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  80,
		ToPort:    80,
		SourceIPs: []string{"205.192.0.0/16", "205.159.0.0/16"},
	}}
	resp, err := s.ec2.AuthorizeSecurityGroup(ec2.SecurityGroup{Name: "websrv"}, perms)

	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"AuthorizeSecurityGroupIngress"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(req.Form["IpPermissions.1.IpProtocol"], DeepEquals, []string{"tcp"})
	c.Assert(req.Form["IpPermissions.1.FromPort"], DeepEquals, []string{"80"})
	c.Assert(req.Form["IpPermissions.1.ToPort"], DeepEquals, []string{"80"})
	c.Assert(req.Form["IpPermissions.1.IpRanges.1.CidrIp"], DeepEquals, []string{"205.192.0.0/16"})
	c.Assert(req.Form["IpPermissions.1.IpRanges.2.CidrIp"], DeepEquals, []string{"205.159.0.0/16"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestAuthorizeSecurityGroupEgress(c *C) {
	testServer.Response(200, nil, AuthorizeSecurityGroupEgressExample)

	perms := []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  80,
		ToPort:    80,
		SourceIPs: []string{"205.192.0.0/16", "205.159.0.0/16"},
	}}
	resp, err := s.ec2.AuthorizeSecurityGroupEgress(ec2.SecurityGroup{Name: "websrv"}, perms)

	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"AuthorizeSecurityGroupEgress"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(req.Form["IpPermissions.1.IpProtocol"], DeepEquals, []string{"tcp"})
	c.Assert(req.Form["IpPermissions.1.FromPort"], DeepEquals, []string{"80"})
	c.Assert(req.Form["IpPermissions.1.ToPort"], DeepEquals, []string{"80"})
	c.Assert(req.Form["IpPermissions.1.IpRanges.1.CidrIp"], DeepEquals, []string{"205.192.0.0/16"})
	c.Assert(req.Form["IpPermissions.1.IpRanges.2.CidrIp"], DeepEquals, []string{"205.159.0.0/16"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestAuthorizeSecurityGroupExample1WithId(c *C) {
	testServer.Response(200, nil, AuthorizeSecurityGroupIngressExample)

	perms := []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  80,
		ToPort:    80,
		SourceIPs: []string{"205.192.0.0/16", "205.159.0.0/16"},
	}}
	// ignore return and error - we're only want to check the parameter handling.
	s.ec2.AuthorizeSecurityGroup(ec2.SecurityGroup{Id: "sg-67ad940e", Name: "ignored"}, perms)

	req := testServer.WaitRequest()

	c.Assert(req.Form["GroupName"], IsNil)
	c.Assert(req.Form["GroupId"], DeepEquals, []string{"sg-67ad940e"})
}

func (s *S) TestAuthorizeSecurityGroupExample2(c *C) {
	testServer.Response(200, nil, AuthorizeSecurityGroupIngressExample)

	perms := []ec2.IPPerm{{
		Protocol: "tcp",
		FromPort: 80,
		ToPort:   81,
		SourceGroups: []ec2.UserSecurityGroup{
			{OwnerId: "999988887777", Name: "OtherAccountGroup"},
			{Id: "sg-67ad940e"},
		},
	}}
	resp, err := s.ec2.AuthorizeSecurityGroup(ec2.SecurityGroup{Name: "websrv"}, perms)

	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"AuthorizeSecurityGroupIngress"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(req.Form["IpPermissions.1.IpProtocol"], DeepEquals, []string{"tcp"})
	c.Assert(req.Form["IpPermissions.1.FromPort"], DeepEquals, []string{"80"})
	c.Assert(req.Form["IpPermissions.1.ToPort"], DeepEquals, []string{"81"})
	c.Assert(req.Form["IpPermissions.1.Groups.1.UserId"], DeepEquals, []string{"999988887777"})
	c.Assert(req.Form["IpPermissions.1.Groups.1.GroupName"], DeepEquals, []string{"OtherAccountGroup"})
	c.Assert(req.Form["IpPermissions.1.Groups.2.UserId"], IsNil)
	c.Assert(req.Form["IpPermissions.1.Groups.2.GroupName"], IsNil)
	c.Assert(req.Form["IpPermissions.1.Groups.2.GroupId"], DeepEquals, []string{"sg-67ad940e"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestRevokeSecurityGroupExample(c *C) {
	// RevokeSecurityGroup is implemented by the same code as AuthorizeSecurityGroup
	// so there's no need to duplicate all the tests.
	testServer.Response(200, nil, RevokeSecurityGroupIngressExample)

	resp, err := s.ec2.RevokeSecurityGroup(ec2.SecurityGroup{Name: "websrv"}, nil)

	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"RevokeSecurityGroupIngress"})
	c.Assert(req.Form["GroupName"], DeepEquals, []string{"websrv"})
	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestCreateTags(c *C) {
	testServer.Response(200, nil, CreateTagsExample)

	resp, err := s.ec2.CreateTags([]string{"ami-1a2b3c4d", "i-7f4d3a2b"}, []ec2.Tag{{"webserver", ""}, {"stack", "Production"}})

	req := testServer.WaitRequest()
	c.Assert(req.Form["ResourceId.1"], DeepEquals, []string{"ami-1a2b3c4d"})
	c.Assert(req.Form["ResourceId.2"], DeepEquals, []string{"i-7f4d3a2b"})
	c.Assert(req.Form["Tag.1.Key"], DeepEquals, []string{"webserver"})
	c.Assert(req.Form["Tag.1.Value"], DeepEquals, []string{""})
	c.Assert(req.Form["Tag.2.Key"], DeepEquals, []string{"stack"})
	c.Assert(req.Form["Tag.2.Value"], DeepEquals, []string{"Production"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestStartInstances(c *C) {
	testServer.Response(200, nil, StartInstancesExample)

	resp, err := s.ec2.StartInstances("i-10a64379")
	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"StartInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-10a64379"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")

	s0 := resp.StateChanges[0]
	c.Assert(s0.InstanceId, Equals, "i-10a64379")
	c.Assert(s0.CurrentState.Code, Equals, 0)
	c.Assert(s0.CurrentState.Name, Equals, "pending")
	c.Assert(s0.PreviousState.Code, Equals, 80)
	c.Assert(s0.PreviousState.Name, Equals, "stopped")
}

func (s *S) TestStopInstances(c *C) {
	testServer.Response(200, nil, StopInstancesExample)

	resp, err := s.ec2.StopInstances("i-10a64379")
	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"StopInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-10a64379"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")

	s0 := resp.StateChanges[0]
	c.Assert(s0.InstanceId, Equals, "i-10a64379")
	c.Assert(s0.CurrentState.Code, Equals, 64)
	c.Assert(s0.CurrentState.Name, Equals, "stopping")
	c.Assert(s0.PreviousState.Code, Equals, 16)
	c.Assert(s0.PreviousState.Name, Equals, "running")
}

func (s *S) TestRebootInstances(c *C) {
	testServer.Response(200, nil, RebootInstancesExample)

	resp, err := s.ec2.RebootInstances("i-10a64379")
	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"RebootInstances"})
	c.Assert(req.Form["InstanceId.1"], DeepEquals, []string{"i-10a64379"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestSignatureWithEndpointPath(c *C) {
	ec2.FakeTime(true)
	defer ec2.FakeTime(false)

	testServer.Response(200, nil, RebootInstancesExample)

	// https://bugs.launchpad.net/goamz/+bug/1022749
	ec2 := ec2.NewWithClient(s.ec2.Auth, aws.Region{EC2Endpoint: testServer.URL + "/services/Cloud"}, testutil.DefaultClient)

	_, err := ec2.RebootInstances("i-10a64379")
	c.Assert(err, IsNil)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Signature"], DeepEquals, []string{"QmvgkYGn19WirCuCz/jRp3RmRgFwWR5WRkKZ5AZnyXQ="})
}

func (s *S) TestAllocateAddressExample(c *C) {
	testServer.Response(200, nil, AllocateAddressExample)

	options := &ec2.AllocateAddress{
		Domain: "vpc",
	}

	resp, err := s.ec2.AllocateAddress(options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"AllocateAddress"})
	c.Assert(req.Form["Domain"], DeepEquals, []string{"vpc"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.PublicIp, Equals, "198.51.100.1")
	c.Assert(resp.Domain, Equals, "vpc")
	c.Assert(resp.AllocationId, Equals, "eipalloc-5723d13e")
}

func (s *S) TestReleaseAddressExample(c *C) {
	testServer.Response(200, nil, ReleaseAddressExample)

	resp, err := s.ec2.ReleaseAddress("eipalloc-5723d13e")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"ReleaseAddress"})
	c.Assert(req.Form["AllocationId"], DeepEquals, []string{"eipalloc-5723d13e"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestAssociateAddressExample(c *C) {
	testServer.Response(200, nil, AssociateAddressExample)

	options := &ec2.AssociateAddress{
		InstanceId:         "i-4fd2431a",
		AllocationId:       "eipalloc-5723d13e",
		AllowReassociation: true,
	}

	resp, err := s.ec2.AssociateAddress(options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"AssociateAddress"})
	c.Assert(req.Form["InstanceId"], DeepEquals, []string{"i-4fd2431a"})
	c.Assert(req.Form["AllocationId"], DeepEquals, []string{"eipalloc-5723d13e"})
	c.Assert(req.Form["AllowReassociation"], DeepEquals, []string{"true"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
	c.Assert(resp.AssociationId, Equals, "eipassoc-fc5ca095")
}

func (s *S) TestDisassociateAddressExample(c *C) {
	testServer.Response(200, nil, DisassociateAddressExample)

	resp, err := s.ec2.DisassociateAddress("eipassoc-aa7486c3")

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DisassociateAddress"})
	c.Assert(req.Form["AssociationId"], DeepEquals, []string{"eipassoc-aa7486c3"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestModifyInstance(c *C) {
	testServer.Response(200, nil, ModifyInstanceExample)

	options := ec2.ModifyInstance{
		InstanceType:          "m1.small",
		DisableAPITermination: true,
		EbsOptimized:          true,
		SecurityGroups:        []ec2.SecurityGroup{{Id: "g1"}, {Id: "g2"}},
		ShutdownBehavior:      "terminate",
		KernelId:              "kernel-id",
		RamdiskId:             "ramdisk-id",
		SourceDestCheck:       true,
		SriovNetSupport:       true,
		UserData:              []byte("1234"),
		BlockDevices: []ec2.BlockDeviceMapping{
			{DeviceName: "/dev/sda1", SnapshotId: "snap-a08912c9", DeleteOnTermination: true},
		},
	}

	resp, err := s.ec2.ModifyInstance("i-2ba64342", &options)
	req := testServer.WaitRequest()

	c.Assert(req.Form["Action"], DeepEquals, []string{"ModifyInstanceAttribute"})
	c.Assert(req.Form["InstanceId"], DeepEquals, []string{"i-2ba64342"})
	c.Assert(req.Form["InstanceType.Value"], DeepEquals, []string{"m1.small"})
	c.Assert(req.Form["BlockDeviceMapping.1.DeviceName"], DeepEquals, []string{"/dev/sda1"})
	c.Assert(req.Form["BlockDeviceMapping.1.Ebs.SnapshotId"], DeepEquals, []string{"snap-a08912c9"})
	c.Assert(req.Form["BlockDeviceMapping.1.Ebs.DeleteOnTermination"], DeepEquals, []string{"true"})
	c.Assert(req.Form["DisableApiTermination.Value"], DeepEquals, []string{"true"})
	c.Assert(req.Form["EbsOptimized"], DeepEquals, []string{"true"})
	c.Assert(req.Form["GroupId.1"], DeepEquals, []string{"g1"})
	c.Assert(req.Form["GroupId.2"], DeepEquals, []string{"g2"})
	c.Assert(req.Form["InstanceInitiatedShutdownBehavior.Value"], DeepEquals, []string{"terminate"})
	c.Assert(req.Form["Kernel.Value"], DeepEquals, []string{"kernel-id"})
	c.Assert(req.Form["Ramdisk.Value"], DeepEquals, []string{"ramdisk-id"})
	c.Assert(req.Form["SourceDestCheck.Value"], DeepEquals, []string{"true"})
	c.Assert(req.Form["SriovNetSupport.Value"], DeepEquals, []string{"simple"})
	c.Assert(req.Form["UserData"], DeepEquals, []string{"MTIzNA=="})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}

func (s *S) TestCreateVpc(c *C) {
	testServer.Response(200, nil, CreateVpcExample)

	options := &ec2.CreateVpc{
		CidrBlock: "foo",
	}

	resp, err := s.ec2.CreateVpc(options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["CidrBlock"], DeepEquals, []string{"foo"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "7a62c49f-347e-4fc4-9331-6e8eEXAMPLE")
	c.Assert(resp.VPC.VpcId, Equals, "vpc-1a2b3c4d")
	c.Assert(resp.VPC.State, Equals, "pending")
	c.Assert(resp.VPC.CidrBlock, Equals, "10.0.0.0/16")
	c.Assert(resp.VPC.DHCPOptionsID, Equals, "dopt-1a2b3c4d2")
	c.Assert(resp.VPC.InstanceTenancy, Equals, "default")
}

func (s *S) TestDescribeVpcs(c *C) {
	testServer.Response(200, nil, DescribeVpcsExample)

	filter := ec2.NewFilter()
	filter.Add("key1", "value1")
	filter.Add("key2", "value2", "value3")

	resp, err := s.ec2.DescribeVpcs([]string{"id1", "id2"}, filter)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"DescribeVpcs"})
	c.Assert(req.Form["VpcId.1"], DeepEquals, []string{"id1"})
	c.Assert(req.Form["VpcId.2"], DeepEquals, []string{"id2"})
	c.Assert(req.Form["Filter.1.Name"], DeepEquals, []string{"key1"})
	c.Assert(req.Form["Filter.1.Value.1"], DeepEquals, []string{"value1"})
	c.Assert(req.Form["Filter.1.Value.2"], IsNil)
	c.Assert(req.Form["Filter.2.Name"], DeepEquals, []string{"key2"})
	c.Assert(req.Form["Filter.2.Value.1"], DeepEquals, []string{"value2"})
	c.Assert(req.Form["Filter.2.Value.2"], DeepEquals, []string{"value3"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "7a62c49f-347e-4fc4-9331-6e8eEXAMPLE")
	c.Assert(resp.VPCs, HasLen, 1)
}

func (s *S) TestCreateSubnet(c *C) {
	testServer.Response(200, nil, CreateSubnetExample)

	options := &ec2.CreateSubnet{
		AvailabilityZone: "baz",
		CidrBlock:        "foo",
		VpcId:            "bar",
	}

	resp, err := s.ec2.CreateSubnet(options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["VpcId"], DeepEquals, []string{"bar"})
	c.Assert(req.Form["CidrBlock"], DeepEquals, []string{"foo"})
	c.Assert(req.Form["AvailabilityZone"], DeepEquals, []string{"baz"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "7a62c49f-347e-4fc4-9331-6e8eEXAMPLE")
	c.Assert(resp.Subnet.SubnetId, Equals, "subnet-9d4a7b6c")
	c.Assert(resp.Subnet.State, Equals, "pending")
	c.Assert(resp.Subnet.VpcId, Equals, "vpc-1a2b3c4d")
	c.Assert(resp.Subnet.CidrBlock, Equals, "10.0.1.0/24")
	c.Assert(resp.Subnet.AvailableIpAddressCount, Equals, 251)
}

func (s *S) TestResetImageAttribute(c *C) {
	testServer.Response(200, nil, ResetImageAttributeExample)

	options := ec2.ResetImageAttribute{Attribute: "launchPermission"}
	resp, err := s.ec2.ResetImageAttribute("i-2ba64342", &options)

	req := testServer.WaitRequest()
	c.Assert(req.Form["Action"], DeepEquals, []string{"ResetImageAttribute"})

	c.Assert(err, IsNil)
	c.Assert(resp.RequestId, Equals, "59dbff89-35bd-4eac-99ed-be587EXAMPLE")
}
