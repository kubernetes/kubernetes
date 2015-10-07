package ec2_test

var ErrorDump = `
<?xml version="1.0" encoding="UTF-8"?>
<Response><Errors><Error><Code>UnsupportedOperation</Code>
<Message>AMIs with an instance-store root device are not supported for the instance type 't1.micro'.</Message>
</Error></Errors><RequestID>0503f4e9-bbd6-483c-b54f-c4ae9f3b30f4</RequestID></Response>
`

// http://goo.gl/Mcm3b
var RunInstancesExample = `
<RunInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <reservationId>r-47a5402e</reservationId>
  <ownerId>999988887777</ownerId>
  <groupSet>
      <item>
          <groupId>sg-67ad940e</groupId>
          <groupName>default</groupName>
      </item>
  </groupSet>
  <instancesSet>
    <item>
      <instanceId>i-2ba64342</instanceId>
      <imageId>ami-60a54009</imageId>
      <instanceState>
        <code>0</code>
        <name>pending</name>
      </instanceState>
      <privateDnsName></privateDnsName>
      <dnsName></dnsName>
      <keyName>example-key-name</keyName>
      <amiLaunchIndex>0</amiLaunchIndex>
      <instanceType>m1.small</instanceType>
      <launchTime>2007-08-07T11:51:50.000Z</launchTime>
      <placement>
        <availabilityZone>us-east-1b</availabilityZone>
      </placement>
      <monitoring>
        <state>enabled</state>
      </monitoring>
      <virtualizationType>paravirtual</virtualizationType>
      <clientToken/>
      <tagSet/>
      <hypervisor>xen</hypervisor>
    </item>
    <item>
      <instanceId>i-2bc64242</instanceId>
      <imageId>ami-60a54009</imageId>
      <instanceState>
        <code>0</code>
        <name>pending</name>
      </instanceState>
      <privateDnsName></privateDnsName>
      <dnsName></dnsName>
      <keyName>example-key-name</keyName>
      <amiLaunchIndex>1</amiLaunchIndex>
      <instanceType>m1.small</instanceType>
      <launchTime>2007-08-07T11:51:50.000Z</launchTime>
      <placement>
         <availabilityZone>us-east-1b</availabilityZone>
      </placement>
      <monitoring>
        <state>enabled</state>
      </monitoring>
      <virtualizationType>paravirtual</virtualizationType>
      <clientToken/>
      <tagSet/>
      <hypervisor>xen</hypervisor>
    </item>
    <item>
      <instanceId>i-2be64332</instanceId>
      <imageId>ami-60a54009</imageId>
      <instanceState>
        <code>0</code>
        <name>pending</name>
      </instanceState>
      <privateDnsName></privateDnsName>
      <dnsName></dnsName>
      <keyName>example-key-name</keyName>
      <amiLaunchIndex>2</amiLaunchIndex>
      <instanceType>m1.small</instanceType>
      <launchTime>2007-08-07T11:51:50.000Z</launchTime>
      <placement>
         <availabilityZone>us-east-1b</availabilityZone>
      </placement>
      <monitoring>
        <state>enabled</state>
      </monitoring>
      <virtualizationType>paravirtual</virtualizationType>
      <clientToken/>
      <tagSet/>
      <hypervisor>xen</hypervisor>
    </item>
  </instancesSet>
</RunInstancesResponse>
`

// http://goo.gl/GRZgCD
var RequestSpotInstancesExample = `
<RequestSpotInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2014-02-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <spotInstanceRequestSet>
    <item>
      <spotInstanceRequestId>sir-1a2b3c4d</spotInstanceRequestId>
      <spotPrice>0.5</spotPrice>
      <type>one-time</type>
      <state>open</state>
      <status>
        <code>pending-evaluation</code>
        <updateTime>2008-05-07T12:51:50.000Z</updateTime>
        <message>Your Spot request has been submitted for review, and is pending evaluation.</message>
      </status>
      <availabilityZoneGroup>MyAzGroup</availabilityZoneGroup>
      <launchSpecification>
        <imageId>ami-1a2b3c4d</imageId>
        <keyName>gsg-keypair</keyName>
        <groupSet>
          <item>
            <groupId>sg-1a2b3c4d</groupId>
            <groupName>websrv</groupName>
          </item>
        </groupSet>
        <instanceType>m1.small</instanceType>
        <blockDeviceMapping/>
        <monitoring>
          <enabled>false</enabled>
        </monitoring>
        <ebsOptimized>false</ebsOptimized>
      </launchSpecification>
      <createTime>YYYY-MM-DDTHH:MM:SS.000Z</createTime>
      <productDescription>Linux/UNIX</productDescription>
    </item>
 </spotInstanceRequestSet>
</RequestSpotInstancesResponse>
`

// http://goo.gl/KsKJJk
var DescribeSpotRequestsExample = `
<DescribeSpotInstanceRequestsResponse xmlns="http://ec2.amazonaws.com/doc/2014-02-01/">
  <requestId>b1719f2a-5334-4479-b2f1-26926EXAMPLE</requestId>
  <spotInstanceRequestSet>
    <item>
      <spotInstanceRequestId>sir-1a2b3c4d</spotInstanceRequestId>
      <spotPrice>0.5</spotPrice>
      <type>one-time</type>
      <state>active</state>
      <status>
        <code>fulfilled</code>
        <updateTime>2008-05-07T12:51:50.000Z</updateTime>
        <message>Your Spot request is fulfilled.</message>
      </status>
      <launchSpecification>
        <imageId>ami-1a2b3c4d</imageId>
        <keyName>gsg-keypair</keyName>
        <groupSet>
          <item>
            <groupId>sg-1a2b3c4d</groupId>
            <groupName>websrv</groupName>
          </item>
        </groupSet>
        <instanceType>m1.small</instanceType>
        <monitoring>
          <enabled>false</enabled>
        </monitoring>
        <ebsOptimized>false</ebsOptimized>
      </launchSpecification>
      <instanceId>i-1a2b3c4d</instanceId>
      <createTime>YYYY-MM-DDTHH:MM:SS.000Z</createTime>
      <productDescription>Linux/UNIX</productDescription>
      <launchedAvailabilityZone>us-east-1a</launchedAvailabilityZone>
    </item>
  </spotInstanceRequestSet>
</DescribeSpotInstanceRequestsResponse>
`

// http://goo.gl/DcfFgJ
var CancelSpotRequestsExample = `
<CancelSpotInstanceRequestsResponse xmlns="http://ec2.amazonaws.com/doc/2014-02-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <spotInstanceRequestSet>
    <item>
      <spotInstanceRequestId>sir-1a2b3c4d</spotInstanceRequestId>
      <state>cancelled</state>
    </item>
  </spotInstanceRequestSet>
</CancelSpotInstanceRequestsResponse>
`

// http://goo.gl/3BKHj
var TerminateInstancesExample = `
<TerminateInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <instancesSet>
    <item>
      <instanceId>i-3ea74257</instanceId>
      <currentState>
        <code>32</code>
        <name>shutting-down</name>
      </currentState>
      <previousState>
        <code>16</code>
        <name>running</name>
      </previousState>
    </item>
  </instancesSet>
</TerminateInstancesResponse>
`

// http://goo.gl/mLbmw
var DescribeInstancesExample1 = `
<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>98e3c9a4-848c-4d6d-8e8a-b1bdEXAMPLE</requestId>
  <reservationSet>
    <item>
      <reservationId>r-b27e30d9</reservationId>
      <ownerId>999988887777</ownerId>
      <groupSet>
        <item>
          <groupId>sg-67ad940e</groupId>
          <groupName>default</groupName>
        </item>
      </groupSet>
      <instancesSet>
        <item>
          <instanceId>i-c5cd56af</instanceId>
          <imageId>ami-1a2b3c4d</imageId>
          <instanceState>
            <code>16</code>
            <name>running</name>
          </instanceState>
          <privateDnsName>domU-12-31-39-10-56-34.compute-1.internal</privateDnsName>
          <dnsName>ec2-174-129-165-232.compute-1.amazonaws.com</dnsName>
          <reason/>
          <keyName>GSG_Keypair</keyName>
          <amiLaunchIndex>0</amiLaunchIndex>
          <productCodes/>
          <instanceType>m1.small</instanceType>
          <launchTime>2010-08-17T01:15:18.000Z</launchTime>
          <placement>
            <availabilityZone>us-east-1b</availabilityZone>
            <groupName/>
          </placement>
          <kernelId>aki-94c527fd</kernelId>
          <ramdiskId>ari-96c527ff</ramdiskId>
          <monitoring>
            <state>disabled</state>
          </monitoring>
          <privateIpAddress>10.198.85.190</privateIpAddress>
          <ipAddress>174.129.165.232</ipAddress>
          <architecture>i386</architecture>
          <rootDeviceType>ebs</rootDeviceType>
          <rootDeviceName>/dev/sda1</rootDeviceName>
          <blockDeviceMapping>
            <item>
              <deviceName>/dev/sda1</deviceName>
              <ebs>
                <volumeId>vol-a082c1c9</volumeId>
                <status>attached</status>
                <attachTime>2010-08-17T01:15:21.000Z</attachTime>
                <deleteOnTermination>false</deleteOnTermination>
              </ebs>
            </item>
          </blockDeviceMapping>
          <instanceLifecycle>spot</instanceLifecycle>
          <spotInstanceRequestId>sir-7a688402</spotInstanceRequestId>
          <virtualizationType>paravirtual</virtualizationType>
          <clientToken/>
          <tagSet/>
          <hypervisor>xen</hypervisor>
       </item>
      </instancesSet>
      <requesterId>854251627541</requesterId>
    </item>
    <item>
      <reservationId>r-b67e30dd</reservationId>
      <ownerId>999988887777</ownerId>
      <groupSet>
        <item>
          <groupId>sg-67ad940e</groupId>
          <groupName>default</groupName>
        </item>
      </groupSet>
      <instancesSet>
        <item>
          <instanceId>i-d9cd56b3</instanceId>
          <imageId>ami-1a2b3c4d</imageId>
          <instanceState>
            <code>16</code>
            <name>running</name>
          </instanceState>
          <privateDnsName>domU-12-31-39-10-54-E5.compute-1.internal</privateDnsName>
          <dnsName>ec2-184-73-58-78.compute-1.amazonaws.com</dnsName>
          <reason/>
          <keyName>GSG_Keypair</keyName>
          <amiLaunchIndex>0</amiLaunchIndex>
          <productCodes/>
          <instanceType>m1.large</instanceType>
          <launchTime>2010-08-17T01:15:19.000Z</launchTime>
          <placement>
            <availabilityZone>us-east-1b</availabilityZone>
            <groupName/>
          </placement>
          <kernelId>aki-94c527fd</kernelId>
          <ramdiskId>ari-96c527ff</ramdiskId>
          <monitoring>
            <state>disabled</state>
          </monitoring>
          <privateIpAddress>10.198.87.19</privateIpAddress>
          <ipAddress>184.73.58.78</ipAddress>
          <architecture>i386</architecture>
          <rootDeviceType>ebs</rootDeviceType>
          <rootDeviceName>/dev/sda1</rootDeviceName>
          <blockDeviceMapping>
            <item>
              <deviceName>/dev/sda1</deviceName>
              <ebs>
                <volumeId>vol-a282c1cb</volumeId>
                <status>attached</status>
                <attachTime>2010-08-17T01:15:23.000Z</attachTime>
                <deleteOnTermination>false</deleteOnTermination>
              </ebs>
            </item>
          </blockDeviceMapping>
          <instanceLifecycle>spot</instanceLifecycle>
          <spotInstanceRequestId>sir-55a3aa02</spotInstanceRequestId>
          <virtualizationType>paravirtual</virtualizationType>
          <clientToken/>
          <tagSet/>
          <hypervisor>xen</hypervisor>
       </item>
      </instancesSet>
      <requesterId>854251627541</requesterId>
    </item>
  </reservationSet>
</DescribeInstancesResponse>
`

// http://goo.gl/mLbmw
var DescribeInstancesExample2 = `
<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <reservationSet>
    <item>
      <reservationId>r-bc7e30d7</reservationId>
      <ownerId>999988887777</ownerId>
      <groupSet>
        <item>
          <groupId>sg-67ad940e</groupId>
          <groupName>default</groupName>
        </item>
      </groupSet>
      <instancesSet>
        <item>
          <instanceId>i-c7cd56ad</instanceId>
          <imageId>ami-b232d0db</imageId>
          <instanceState>
            <code>16</code>
            <name>running</name>
          </instanceState>
          <privateDnsName>domU-12-31-39-01-76-06.compute-1.internal</privateDnsName>
          <dnsName>ec2-72-44-52-124.compute-1.amazonaws.com</dnsName>
          <keyName>GSG_Keypair</keyName>
          <amiLaunchIndex>0</amiLaunchIndex>
          <productCodes/>
          <instanceType>m1.small</instanceType>
          <launchTime>2010-08-17T01:15:16.000Z</launchTime>
          <placement>
              <availabilityZone>us-east-1b</availabilityZone>
          </placement>
          <kernelId>aki-94c527fd</kernelId>
          <ramdiskId>ari-96c527ff</ramdiskId>
          <monitoring>
              <state>disabled</state>
          </monitoring>
          <privateIpAddress>10.255.121.240</privateIpAddress>
          <ipAddress>72.44.52.124</ipAddress>
          <architecture>i386</architecture>
          <rootDeviceType>ebs</rootDeviceType>
          <rootDeviceName>/dev/sda1</rootDeviceName>
          <blockDeviceMapping>
              <item>
                 <deviceName>/dev/sda1</deviceName>
                 <ebs>
                    <volumeId>vol-a482c1cd</volumeId>
                    <status>attached</status>
                    <attachTime>2010-08-17T01:15:26.000Z</attachTime>
                    <deleteOnTermination>true</deleteOnTermination>
                </ebs>
             </item>
          </blockDeviceMapping>
          <virtualizationType>paravirtual</virtualizationType>
          <clientToken/>
          <tagSet>
              <item>
                    <key>webserver</key>
                    <value></value>
             </item>
              <item>
                    <key>stack</key>
                    <value>Production</value>
             </item>
          </tagSet>
          <hypervisor>xen</hypervisor>
        </item>
      </instancesSet>
    </item>
  </reservationSet>
</DescribeInstancesResponse>
`

// http://goo.gl/cxU41
var CreateImageExample = `
<CreateImageResponse xmlns="http://ec2.amazonaws.com/doc/2013-02-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <imageId>ami-4fa54026</imageId>
</CreateImageResponse>
`

// http://goo.gl/V0U25
var DescribeImagesExample = `
<DescribeImagesResponse xmlns="http://ec2.amazonaws.com/doc/2012-08-15/">
         <requestId>4a4a27a2-2e7c-475d-b35b-ca822EXAMPLE</requestId>
    <imagesSet>
        <item>
            <imageId>ami-a2469acf</imageId>
            <imageLocation>aws-marketplace/example-marketplace-amzn-ami.1</imageLocation>
            <imageState>available</imageState>
            <imageOwnerId>123456789999</imageOwnerId>
            <isPublic>true</isPublic>
            <productCodes>
                <item>
                    <productCode>a1b2c3d4e5f6g7h8i9j10k11</productCode>
                    <type>marketplace</type>
                </item>
            </productCodes>
            <architecture>i386</architecture>
            <imageType>machine</imageType>
            <kernelId>aki-805ea7e9</kernelId>
            <imageOwnerAlias>aws-marketplace</imageOwnerAlias>
            <name>example-marketplace-amzn-ami.1</name>
            <description>Amazon Linux AMI i386 EBS</description>
            <rootDeviceType>ebs</rootDeviceType>
            <rootDeviceName>/dev/sda1</rootDeviceName>
            <blockDeviceMapping>
                <item>
                    <deviceName>/dev/sda1</deviceName>
                    <ebs>
                        <snapshotId>snap-787e9403</snapshotId>
                        <volumeSize>8</volumeSize>
                        <deleteOnTermination>true</deleteOnTermination>
                    </ebs>
                </item>
            </blockDeviceMapping>
            <virtualizationType>paravirtual</virtualizationType>
            <hypervisor>xen</hypervisor>
        </item>
    </imagesSet>
</DescribeImagesResponse>
`

// http://goo.gl/bHO3z
var ImageAttributeExample = `
<DescribeImageAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2013-07-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <imageId>ami-61a54008</imageId>
   <launchPermission>
      <item>
         <group>all</group>
      </item>
      <item>
         <userId>495219933132</userId>
      </item>
   </launchPermission>
</DescribeImageAttributeResponse>
`

// http://goo.gl/ttcda
var CreateSnapshotExample = `
<CreateSnapshotResponse xmlns="http://ec2.amazonaws.com/doc/2012-10-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <snapshotId>snap-78a54011</snapshotId>
  <volumeId>vol-4d826724</volumeId>
  <status>pending</status>
  <startTime>2008-05-07T12:51:50.000Z</startTime>
  <progress>60%</progress>
  <ownerId>111122223333</ownerId>
  <volumeSize>10</volumeSize>
  <description>Daily Backup</description>
</CreateSnapshotResponse>
`

// http://goo.gl/vwU1y
var DeleteSnapshotExample = `
<DeleteSnapshotResponse xmlns="http://ec2.amazonaws.com/doc/2012-10-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</DeleteSnapshotResponse>
`

// http://goo.gl/nkovs
var DescribeSnapshotsExample = `
<DescribeSnapshotsResponse xmlns="http://ec2.amazonaws.com/doc/2012-10-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <snapshotSet>
      <item>
         <snapshotId>snap-1a2b3c4d</snapshotId>
         <volumeId>vol-8875daef</volumeId>
         <status>pending</status>
         <startTime>2010-07-29T04:12:01.000Z</startTime>
         <progress>30%</progress>
         <ownerId>111122223333</ownerId>
         <volumeSize>15</volumeSize>
         <description>Daily Backup</description>
         <tagSet>
            <item>
               <key>Purpose</key>
               <value>demo_db_14_backup</value>
            </item>
         </tagSet>
      </item>
   </snapshotSet>
</DescribeSnapshotsResponse>
`

// http://goo.gl/YUjO4G
var ModifyImageAttributeExample = `
<ModifyImageAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2013-06-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</ModifyImageAttributeResponse>
`

// http://goo.gl/hQwPCK
var CopyImageExample = `
<CopyImageResponse xmlns="http://ec2.amazonaws.com/doc/2013-06-15/">
   <requestId>60bc441d-fa2c-494d-b155-5d6a3EXAMPLE</requestId>
   <imageId>ami-4d3c2b1a</imageId>
</CopyImageResponse>
`

var CreateKeyPairExample = `
<CreateKeyPairResponse xmlns="http://ec2.amazonaws.com/doc/2013-02-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <keyName>foo</keyName>
  <keyFingerprint>
     00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
  </keyFingerprint>
  <keyMaterial>---- BEGIN RSA PRIVATE KEY ----
MIICiTCCAfICCQD6m7oRw0uXOjANBgkqhkiG9w0BAQUFADCBiDELMAkGA1UEBhMC
VVMxCzAJBgNVBAgTAldBMRAwDgYDVQQHEwdTZWF0dGxlMQ8wDQYDVQQKEwZBbWF6
b24xFDASBgNVBAsTC0lBTSBDb25zb2xlMRIwEAYDVQQDEwlUZXN0Q2lsYWMxHzAd
BgkqhkiG9w0BCQEWEG5vb25lQGFtYXpvbi5jb20wHhcNMTEwNDI1MjA0NTIxWhcN
MTIwNDI0MjA0NTIxWjCBiDELMAkGA1UEBhMCVVMxCzAJBgNVBAgTAldBMRAwDgYD
VQQHEwdTZWF0dGxlMQ8wDQYDVQQKEwZBbWF6b24xFDASBgNVBAsTC0lBTSBDb25z
b2xlMRIwEAYDVQQDEwlUZXN0Q2lsYWMxHzAdBgkqhkiG9w0BCQEWEG5vb25lQGFt
YXpvbi5jb20wgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGBAMaK0dn+a4GmWIWJ
21uUSfwfEvySWtC2XADZ4nB+BLYgVIk60CpiwsZ3G93vUEIO3IyNoH/f0wYK8m9T
rDHudUZg3qX4waLG5M43q7Wgc/MbQITxOUSQv7c7ugFFDzQGBzZswY6786m86gpE
Ibb3OhjZnzcvQAaRHhdlQWIMm2nrAgMBAAEwDQYJKoZIhvcNAQEFBQADgYEAtCu4
nUhVVxYUntneD9+h8Mg9q6q+auNKyExzyLwaxlAoo7TJHidbtS4J5iNmZgXL0Fkb
FFBjvSfpJIlJ00zbhNYS5f6GuoEDmFJl0ZxBHjJnyp378OD8uTs7fLvjx79LjSTb
NYiytVbZPQUQ5Yaxu2jXnimvw3rrszlaEXAMPLE=
-----END RSA PRIVATE KEY-----
</keyMaterial>
</CreateKeyPairResponse>
`

var DeleteKeyPairExample = `
<DeleteKeyPairResponse xmlns="http://ec2.amazonaws.com/doc/2013-02-01/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</DeleteKeyPairResponse>
`

// http://goo.gl/Eo7Yl
var CreateSecurityGroupExample = `
<CreateSecurityGroupResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
   <groupId>sg-67ad940e</groupId>
</CreateSecurityGroupResponse>
`

// http://goo.gl/k12Uy
var DescribeSecurityGroupsExample = `
<DescribeSecurityGroupsResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <securityGroupInfo>
    <item>
      <ownerId>999988887777</ownerId>
      <groupName>WebServers</groupName>
      <groupId>sg-67ad940e</groupId>
      <groupDescription>Web Servers</groupDescription>
      <ipPermissions>
        <item>
           <ipProtocol>tcp</ipProtocol>
           <fromPort>80</fromPort>
           <toPort>80</toPort>
           <groups/>
           <ipRanges>
             <item>
               <cidrIp>0.0.0.0/0</cidrIp>
             </item>
           </ipRanges>
        </item>
      </ipPermissions>
      <ipPermissionsEgress>
        <item>
          <ipProtocol>tcp</ipProtocol>
          <fromPort>80</fromPort>
          <toPort>80</toPort>
          <groups/>
          <ipRanges>
            <item>
              <cidrIp>0.0.0.0/0</cidrIp>
            </item>
          </ipRanges>
        </item>
      </ipPermissionsEgress>
    </item>
    <item>
      <ownerId>999988887777</ownerId>
      <groupName>RangedPortsBySource</groupName>
      <groupId>sg-76abc467</groupId>
      <groupDescription>Group A</groupDescription>
      <ipPermissions>
        <item>
           <ipProtocol>tcp</ipProtocol>
           <fromPort>6000</fromPort>
           <toPort>7000</toPort>
           <groups/>
           <ipRanges/>
        </item>
      </ipPermissions>
    </item>
  </securityGroupInfo>
</DescribeSecurityGroupsResponse>
`

// A dump which includes groups within ip permissions.
var DescribeSecurityGroupsDump = `
<?xml version="1.0" encoding="UTF-8"?>
<DescribeSecurityGroupsResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
    <requestId>87b92b57-cc6e-48b2-943f-f6f0e5c9f46c</requestId>
    <securityGroupInfo>
        <item>
            <ownerId>12345</ownerId>
            <groupName>default</groupName>
            <groupDescription>default group</groupDescription>
            <ipPermissions>
                <item>
                    <ipProtocol>icmp</ipProtocol>
                    <fromPort>-1</fromPort>
                    <toPort>-1</toPort>
                    <groups>
                        <item>
                            <userId>12345</userId>
                            <groupName>default</groupName>
                            <groupId>sg-67ad940e</groupId>
                        </item>
                    </groups>
                    <ipRanges/>
                </item>
                <item>
                    <ipProtocol>tcp</ipProtocol>
                    <fromPort>0</fromPort>
                    <toPort>65535</toPort>
                    <groups>
                        <item>
                            <userId>12345</userId>
                            <groupName>other</groupName>
                            <groupId>sg-76abc467</groupId>
                        </item>
                    </groups>
                    <ipRanges/>
                </item>
            </ipPermissions>
        </item>
    </securityGroupInfo>
</DescribeSecurityGroupsResponse>
`

// http://goo.gl/QJJDO
var DeleteSecurityGroupExample = `
<DeleteSecurityGroupResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</DeleteSecurityGroupResponse>
`

// http://goo.gl/u2sDJ
var AuthorizeSecurityGroupIngressExample = `
<AuthorizeSecurityGroupIngressResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</AuthorizeSecurityGroupIngressResponse>
`

// http://goo.gl/u2sDJ
var AuthorizeSecurityGroupEgressExample = `
<AuthorizeSecurityGroupEgressResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</AuthorizeSecurityGroupEgressResponse>
`

// http://goo.gl/Mz7xr
var RevokeSecurityGroupIngressExample = `
<RevokeSecurityGroupIngressResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</RevokeSecurityGroupIngressResponse>
`

// http://goo.gl/Vmkqc
var CreateTagsExample = `
<CreateTagsResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</CreateTagsResponse>
`

// http://goo.gl/awKeF
var StartInstancesExample = `
<StartInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <instancesSet>
    <item>
      <instanceId>i-10a64379</instanceId>
      <currentState>
          <code>0</code>
          <name>pending</name>
      </currentState>
      <previousState>
          <code>80</code>
          <name>stopped</name>
      </previousState>
    </item>
  </instancesSet>
</StartInstancesResponse>
`

// http://goo.gl/436dJ
var StopInstancesExample = `
<StopInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <instancesSet>
    <item>
      <instanceId>i-10a64379</instanceId>
      <currentState>
          <code>64</code>
          <name>stopping</name>
      </currentState>
      <previousState>
          <code>16</code>
          <name>running</name>
      </previousState>
    </item>
  </instancesSet>
</StopInstancesResponse>
`

// http://goo.gl/baoUf
var RebootInstancesExample = `
<RebootInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2011-12-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</RebootInstancesResponse>
`

// http://goo.gl/9rprDN
var AllocateAddressExample = `
<AllocateAddressResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <publicIp>198.51.100.1</publicIp>
   <domain>vpc</domain>
   <allocationId>eipalloc-5723d13e</allocationId>
</AllocateAddressResponse>
`

// http://goo.gl/DFySJY
var DescribeInstanceStatusExample = `
<DescribeInstanceStatusResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
    <requestId>3be1508e-c444-4fef-89cc-0b1223c4f02fEXAMPLE</requestId>
    <instanceStatusSet>
        <item>
            <instanceId>i-1a2b3c4d</instanceId>
            <availabilityZone>us-east-1d</availabilityZone>
            <instanceState>
                <code>16</code>
                <name>running</name>
            </instanceState>
            <systemStatus>
                <status>impaired</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>failed</status>
                        <impairedSince>YYYY-MM-DDTHH:MM:SS.000Z</impairedSince>
                    </item>
                </details>
            </systemStatus>
            <instanceStatus>
                <status>impaired</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>failed</status>
                        <impairedSince>YYYY-MM-DDTHH:MM:SS.000Z</impairedSince>
                    </item>
                </details>
            </instanceStatus>
            <eventsSet>
              <item>
                <code>instance-retirement</code>
                <description>The instance is running on degraded hardware</description>
                <notBefore>YYYY-MM-DDTHH:MM:SS+0000</notBefore>
                <notAfter>YYYY-MM-DDTHH:MM:SS+0000</notAfter>
              </item>
            </eventsSet>
        </item>
        <item>
            <instanceId>i-2a2b3c4d</instanceId>
            <availabilityZone>us-east-1d</availabilityZone>
            <instanceState>
                <code>16</code>
                <name>running</name>
            </instanceState>
            <systemStatus>
                <status>ok</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>passed</status>
                    </item>
                </details>
            </systemStatus>
            <instanceStatus>
                <status>ok</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>passed</status>
                    </item>
                </details>
            </instanceStatus>
            <eventsSet>
              <item>
                <code>instance-reboot</code>
                <description>The instance is scheduled for a reboot</description>
                <notBefore>YYYY-MM-DDTHH:MM:SS+0000</notBefore>
                <notAfter>YYYY-MM-DDTHH:MM:SS+0000</notAfter>
              </item>
            </eventsSet>
        </item>
        <item>
            <instanceId>i-3a2b3c4d</instanceId>
            <availabilityZone>us-east-1c</availabilityZone>
            <instanceState>
                <code>16</code>
                <name>running</name>
            </instanceState>
            <systemStatus>
                <status>ok</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>passed</status>
                    </item>
                </details>
            </systemStatus>
            <instanceStatus>
                <status>ok</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>passed</status>
                    </item>
                </details>
            </instanceStatus>
        </item>
        <item>
            <instanceId>i-4a2b3c4d</instanceId>
            <availabilityZone>us-east-1c</availabilityZone>
            <instanceState>
                <code>16</code>
                <name>running</name>
            </instanceState>
            <systemStatus>
                <status>ok</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>passed</status>
                    </item>
                </details>
            </systemStatus>
            <instanceStatus>
                <status>insufficient-data</status>
                <details>
                    <item>
                        <name>reachability</name>
                        <status>insufficient-data</status>
                    </item>
                </details>
            </instanceStatus>
         </item>
    </instanceStatusSet>
</DescribeInstanceStatusResponse>
`

// http://goo.gl/3Q0oCc
var ReleaseAddressExample = `
<ReleaseAddressResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</ReleaseAddressResponse>
`

// http://goo.gl/uOSQE
var AssociateAddressExample = `
<AssociateAddressResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
   <associationId>eipassoc-fc5ca095</associationId>
</AssociateAddressResponse>
`

// http://goo.gl/LrOa0
var DisassociateAddressExample = `
<DisassociateAddressResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</DisassociateAddressResponse>
`

// http://goo.gl/icuXh5
var ModifyInstanceExample = `
<ModifyImageAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2013-06-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</ModifyImageAttributeResponse>
`

var CreateVpcExample = `
<CreateVpcResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
   <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
   <vpc>
      <vpcId>vpc-1a2b3c4d</vpcId>
      <state>pending</state>
      <cidrBlock>10.0.0.0/16</cidrBlock>
      <dhcpOptionsId>dopt-1a2b3c4d2</dhcpOptionsId>
      <instanceTenancy>default</instanceTenancy>
      <tagSet/>
   </vpc>
</CreateVpcResponse>
`

var DescribeVpcsExample = `
<DescribeVpcsResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
  <vpcSet>
    <item>
      <vpcId>vpc-1a2b3c4d</vpcId>
      <state>available</state>
      <cidrBlock>10.0.0.0/23</cidrBlock>
      <dhcpOptionsId>dopt-7a8b9c2d</dhcpOptionsId>
      <instanceTenancy>default</instanceTenancy>
      <isDefault>false</isDefault>
      <tagSet/>
    </item>
  </vpcSet>
</DescribeVpcsResponse>
`

var CreateSubnetExample = `
<CreateSubnetResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
  <subnet>
    <subnetId>subnet-9d4a7b6c</subnetId>
    <state>pending</state>
    <vpcId>vpc-1a2b3c4d</vpcId>
    <cidrBlock>10.0.1.0/24</cidrBlock>
    <availableIpAddressCount>251</availableIpAddressCount>
    <availabilityZone>us-east-1a</availabilityZone>
    <tagSet/>
  </subnet>
</CreateSubnetResponse>
`

// http://goo.gl/tu2Kxm
var ModifySubnetAttributeExample = `
<ModifySubnetAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</ModifySubnetAttributeResponse>
`

// http://goo.gl/r6ZCPm
var ResetImageAttributeExample = `
<ResetImageAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
  <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
  <return>true</return>
</ResetImageAttributeResponse>
`

// http://goo.gl/ylxT4R
var DescribeAvailabilityZonesExample1 = `
<DescribeAvailabilityZonesResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <availabilityZoneInfo>
   <item>
      <zoneName>us-east-1a</zoneName>
      <zoneState>available</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet/>
   </item>
   <item>
      <zoneName>us-east-1b</zoneName>
      <zoneState>available</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet/>
   </item>
   <item>
      <zoneName>us-east-1c</zoneName>
      <zoneState>available</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet/>
   </item>
   <item>
      <zoneName>us-east-1d</zoneName>
      <zoneState>available</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet/>
   </item>
   </availabilityZoneInfo>
</DescribeAvailabilityZonesResponse>
`

// http://goo.gl/ylxT4R
var DescribeAvailabilityZonesExample2 = `
<DescribeAvailabilityZonesResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <availabilityZoneInfo>
   <item>
      <zoneName>us-east-1a</zoneName>
      <zoneState>impaired</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet/>
   </item>
   <item>
      <zoneName>us-east-1b</zoneName>
      <zoneState>unavailable</zoneState>
      <regionName>us-east-1</regionName>
      <messageSet>
         <item>us-east-1b is currently down for maintenance.</item>
      </messageSet>
   </item>
   </availabilityZoneInfo>
</DescribeAvailabilityZonesResponse>
`

// http://goo.gl/sdomyE
var CreateNetworkAclExample = `
<CreateNetworkAclResponse xmlns="http://ec2.amazonaws.com/doc/2014-10-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <networkAcl>
      <networkAclId>acl-5fb85d36</networkAclId>
      <vpcId>vpc-11ad4878</vpcId>
      <default>false</default>
      <entrySet>
         <item>
            <ruleNumber>32767</ruleNumber>
            <protocol>-1</protocol>
            <ruleAction>deny</ruleAction>
            <egress>true</egress>
            <cidrBlock>0.0.0.0/0</cidrBlock>
         </item>
         <item>
            <ruleNumber>32767</ruleNumber>
            <protocol>-1</protocol>
            <ruleAction>deny</ruleAction>
            <egress>false</egress>
            <cidrBlock>0.0.0.0/0</cidrBlock>
         </item>
      </entrySet>
      <associationSet/>
      <tagSet/>
   </networkAcl>
</CreateNetworkAclResponse>
`

// http://goo.gl/6sYloC
var CreateNetworkAclEntryRespExample = `
<CreateNetworkAclEntryResponse xmlns="http://ec2.amazonaws.com/doc/2014-10-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <return>true</return>
</CreateNetworkAclEntryResponse>
`

// http://goo.gl/5tqceF
var DescribeNetworkAclsExample = `
<DescribeNetworkAclsResponse xmlns="http://ec2.amazonaws.com/doc/2014-10-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <networkAclSet>
   <item>
     <networkAclId>acl-5566953c</networkAclId>
     <vpcId>vpc-5266953b</vpcId>
     <default>true</default>
     <entrySet>
       <item>
         <ruleNumber>100</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>allow</ruleAction>
         <egress>true</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
       <item>
         <ruleNumber>32767</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>deny</ruleAction>
         <egress>true</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
       <item>
         <ruleNumber>100</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>allow</ruleAction>
         <egress>false</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
       <item>
         <ruleNumber>32767</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>deny</ruleAction>
         <egress>false</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
     </entrySet>
     <associationSet/>
     <tagSet/>
   </item>
   <item>
     <networkAclId>acl-5d659634</networkAclId>
     <vpcId>vpc-5266953b</vpcId>
     <default>false</default>
     <entrySet>
       <item>
         <ruleNumber>110</ruleNumber>
         <protocol>6</protocol>
         <ruleAction>allow</ruleAction>
         <egress>true</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
         <portRange>
           <from>49152</from>
           <to>65535</to>
         </portRange>
       </item>
       <item>
         <ruleNumber>32767</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>deny</ruleAction>
         <egress>true</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
       <item>
         <ruleNumber>110</ruleNumber>
         <protocol>6</protocol>
         <ruleAction>allow</ruleAction>
         <egress>false</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
         <portRange>
           <from>80</from>
           <to>80</to>
         </portRange>
       </item>
       <item>
         <ruleNumber>120</ruleNumber>
         <protocol>6</protocol>
         <ruleAction>allow</ruleAction>
         <egress>false</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
         <portRange>
           <from>443</from>
           <to>443</to>
         </portRange>
       </item>
       <item>
         <ruleNumber>32767</ruleNumber>
         <protocol>-1</protocol>
         <ruleAction>deny</ruleAction>
         <egress>false</egress>
         <cidrBlock>0.0.0.0/0</cidrBlock>
       </item>
     </entrySet>
     <associationSet>
       <item>
         <networkAclAssociationId>aclassoc-5c659635</networkAclAssociationId>
         <networkAclId>acl-5d659634</networkAclId>
         <subnetId>subnet-ff669596</subnetId>
       </item>
       <item>
         <networkAclAssociationId>aclassoc-c26596ab</networkAclAssociationId>
         <networkAclId>acl-5d659634</networkAclId>
         <subnetId>subnet-f0669599</subnetId>
       </item>
     </associationSet>
     <tagSet/>
   </item>
 </networkAclSet>
</DescribeNetworkAclsResponse>
`

var ReplaceNetworkAclAssociationResponseExample = `
<ReplaceNetworkAclAssociationResponse xmlns="http://ec2.amazonaws.com/doc/2014-10-01/">
   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>
   <newAssociationId>aclassoc-17b85d7e</newAssociationId>
</ReplaceNetworkAclAssociationResponse>
`

var CreateCustomerGatewayResponseExample = `
<CreateCustomerGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
   <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
   <customerGateway>
      <customerGatewayId>cgw-b4dc3961</customerGatewayId>
      <state>pending</state>
      <type>ipsec.1</type>
      <ipAddress>10.0.0.20</ipAddress>
      <bgpAsn>65534</bgpAsn>
      <tagSet/>
   </customerGateway>
</CreateCustomerGatewayResponse>
`

var DescribeCustomerGatewaysResponseExample = `
<DescribeCustomerGatewaysResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
  <customerGatewaySet>
    <item>
      <customerGatewayId>cgw-b4dc3961</customerGatewayId>
      <state>available</state>
      <type>ipsec.1</type>
      <ipAddress>12.1.2.3</ipAddress>
      <bgpAsn>65534</bgpAsn>
      <tagSet/>
    </item>
    <item>
      <customerGatewayId>cgw-b4dc3962</customerGatewayId>
      <state>pending</state>
      <type>ipsec.1</type>
      <ipAddress>12.1.2.4</ipAddress>
      <bgpAsn>65500</bgpAsn>
      <tagSet/>
    </item>
  </customerGatewaySet>
</DescribeCustomerGatewaysResponse>
`
var DeleteCustomerGatewayResponseExample = `
<DeleteCustomerGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2014-06-15/">
   <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>
   <return>true</return>
</DeleteCustomerGatewayResponse>`
