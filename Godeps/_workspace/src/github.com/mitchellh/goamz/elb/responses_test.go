package elb_test

var ErrorDump = `
<?xml version="1.0" encoding="UTF-8"?>
<Response><Errors><Error><Code>UnsupportedOperation</Code>
<Message></Message>
</Error></Errors><RequestID>0503f4e9-bbd6-483c-b54f-c4ae9f3b30f4</RequestID></Response>
`

// http://goo.gl/OkMdtJ
var AddTagsExample = `
<AddTagsResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <AddTagsResult/>
  <ResponseMetadata>
    <RequestId>360e81f7-1100-11e4-b6ed-0f30EXAMPLE</RequestId>
  </ResponseMetadata>
</AddTagsResponse>
`

// http://goo.gl/nT2E89
var RemoveTagsExample = `
<RemoveTagsResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <RemoveTagsResult/>
  <ResponseMetadata>
    <RequestId>83c88b9d-12b7-11e3-8b82-87b12EXAMPLE</RequestId>
  </ResponseMetadata>
</RemoveTagsResponse>
`

// http://goo.gl/gQRD2H
var CreateLoadBalancerExample = `
<CreateLoadBalancerResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <CreateLoadBalancerResult>
    <DNSName>MyLoadBalancer-1234567890.us-east-1.elb.amazonaws.com</DNSName>
  </CreateLoadBalancerResult>
  <ResponseMetadata>
    <RequestId>1549581b-12b7-11e3-895e-1334aEXAMPLE</RequestId>
  </ResponseMetadata>
</CreateLoadBalancerResponse>
`

// http://goo.gl/GLZeBN
var DeleteLoadBalancerExample = `
<DeleteLoadBalancerResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <ResponseMetadata>
    <RequestId>1549581b-12b7-11e3-895e-1334aEXAMPLE</RequestId>
  </ResponseMetadata>
</DeleteLoadBalancerResponse>
`

// http://goo.gl/8UgpQ8
var DescribeLoadBalancersExample = `
<DescribeLoadBalancersResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <DescribeLoadBalancersResult>
      <LoadBalancerDescriptions>
        <member>
          <SecurityGroups/>
          <LoadBalancerName>MyLoadBalancer</LoadBalancerName>
          <CreatedTime>2013-05-24T21:15:31.280Z</CreatedTime>
          <HealthCheck>
            <Interval>90</Interval>
            <Target>HTTP:80/</Target>
            <HealthyThreshold>2</HealthyThreshold>
            <Timeout>60</Timeout>
            <UnhealthyThreshold>10</UnhealthyThreshold>
          </HealthCheck>
          <ListenerDescriptions>
            <member>
              <PolicyNames/>
              <Listener>
                <Protocol>HTTP</Protocol>
                <LoadBalancerPort>80</LoadBalancerPort>
                <InstanceProtocol>HTTP</InstanceProtocol>
                <SSLCertificateId>needToAddASSLCertToYourAWSAccount</SSLCertificateId>
                <InstancePort>80</InstancePort>
              </Listener>
            </member>
          </ListenerDescriptions>
          <Instances>
            <member>
              <InstanceId>i-e4cbe38d</InstanceId>
            </member>
          </Instances>
          <Policies>
            <AppCookieStickinessPolicies/>
            <OtherPolicies/>
            <LBCookieStickinessPolicies/>
          </Policies>
          <AvailabilityZones>
            <member>us-east-1a</member>
          </AvailabilityZones>
          <CanonicalHostedZoneNameID>ZZZZZZZZZZZ123X</CanonicalHostedZoneNameID>
          <CanonicalHostedZoneName>MyLoadBalancer-123456789.us-east-1.elb.amazonaws.com</CanonicalHostedZoneName>
          <Scheme>internet-facing</Scheme>
          <SourceSecurityGroup>
            <OwnerAlias>amazon-elb</OwnerAlias>
            <GroupName>amazon-elb-sg</GroupName>
          </SourceSecurityGroup>
          <DNSName>MyLoadBalancer-123456789.us-east-1.elb.amazonaws.com</DNSName>
          <BackendServerDescriptions/>
          <Subnets/>
        </member>
      </LoadBalancerDescriptions>
    </DescribeLoadBalancersResult>
  <ResponseMetadata>
      <RequestId>83c88b9d-12b7-11e3-8b82-87b12EXAMPLE</RequestId>
  </ResponseMetadata>
</DescribeLoadBalancersResponse>
`

// http://goo.gl/Uz1N66
var RegisterInstancesWithLoadBalancerExample = `
<RegisterInstancesWithLoadBalancerResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <RegisterInstancesWithLoadBalancerResult>
    <Instances>
      <member>
        <InstanceId>i-315b7e51</InstanceId>
      </member>
    </Instances>
  </RegisterInstancesWithLoadBalancerResult>
<ResponseMetadata>
    <RequestId>83c88b9d-12b7-11e3-8b82-87b12EXAMPLE</RequestId>
</ResponseMetadata>
</RegisterInstancesWithLoadBalancerResponse>
 `

// http://goo.gl/5OMv62
var DeregisterInstancesFromLoadBalancerExample = `
<DeregisterInstancesFromLoadBalancerResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <DeregisterInstancesFromLoadBalancerResult>
    <Instances>
      <member>
        <InstanceId>i-6ec63d59</InstanceId>
      </member>
    </Instances>
  </DeregisterInstancesFromLoadBalancerResult>
<ResponseMetadata>
    <RequestId>83c88b9d-12b7-11e3-8b82-87b12EXAMPLE</RequestId>
</ResponseMetadata>
</DeregisterInstancesFromLoadBalancerResponse>
`

// http://docs.aws.amazon.com/ElasticLoadBalancing/latest/APIReference/API_ConfigureHealthCheck.html
var ConfigureHealthCheckExample = `
<ConfigureHealthCheckResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
<ConfigureHealthCheckResult>
    <HealthCheck>
      <Interval>30</Interval>
      <Target>HTTP:80/ping</Target>
      <HealthyThreshold>2</HealthyThreshold>
      <Timeout>3</Timeout>
      <UnhealthyThreshold>2</UnhealthyThreshold>
    </HealthCheck>
</ConfigureHealthCheckResult>
<ResponseMetadata>
    <RequestId>83c88b9d-12b7-11e3-8b82-87b12EXAMPLE</RequestId>
</ResponseMetadata>
</ConfigureHealthCheckResponse>`

// http://goo.gl/cGNxfj
var DescribeInstanceHealthExample = `
<DescribeInstanceHealthResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2012-06-01/">
  <DescribeInstanceHealthResult>
    <InstanceStates>
      <member>
        <Description>N/A</Description>
        <InstanceId>i-90d8c2a5</InstanceId>
        <State>InService</State>
        <ReasonCode>N/A</ReasonCode>
      </member>
      <member>
        <Description>N/A</Description>
        <InstanceId>i-06ea3e60</InstanceId>
        <State>OutOfService</State>
        <ReasonCode>N/A</ReasonCode>
      </member>
    </InstanceStates>
  </DescribeInstanceHealthResult>
  <ResponseMetadata>
    <RequestId>1549581b-12b7-11e3-895e-1334aEXAMPLE</RequestId>
  </ResponseMetadata>
</DescribeInstanceHealthResponse>`
