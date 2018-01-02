// +build go1.7

package virtualmachine

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/xml"
	"testing"
)

func TestDocumentedDeploymentRequest(t *testing.T) {
	// xml based on https://msdn.microsoft.com/en-us/library/azure/jj157194.aspx
	// fixed typos, replaced strongly typed fields with values of correct type
	xmlString := `<Deployment xmlns="http://schemas.microsoft.com/windowsazure" xmlns:i="http://www.w3.org/2001/XMLSchema-instance">
  <Name>name-of-deployment</Name>
  <DeploymentSlot>deployment-environment</DeploymentSlot>
  <Label>identifier-of-deployment</Label>
  <RoleList>
    <Role>
      <RoleName>name-of-the-virtual-machine</RoleName>
      <RoleType>PersistentVMRole</RoleType>
      <ConfigurationSets>
        <ConfigurationSet i:type="WindowsProvisioningConfigurationSet">
          <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
          <ComputerName>name-of-computer</ComputerName>
          <AdminPassword>administrator-password</AdminPassword>
          <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
          <TimeZone>time-zone</TimeZone>
          <DomainJoin>
            <Credentials>
              <Domain>domain-to-join</Domain>
              <Username>user-name-in-the-domain</Username>
              <Password>password-for-the-user-name</Password>
            </Credentials>
            <JoinDomain>domain-to-join</JoinDomain>
            <MachineObjectOU>distinguished-name-of-the-ou</MachineObjectOU>
          </DomainJoin>
          <StoredCertificateSettings>
            <CertificateSetting>
              <StoreLocation>LocalMachine</StoreLocation>
              <StoreName>name-of-store-on-the-machine</StoreName>
              <Thumbprint>certificate-thumbprint</Thumbprint>
            </CertificateSetting>
          </StoredCertificateSettings>
          <WinRM>
            <Listeners>
              <Listener>
                <Protocol>listener-protocol</Protocol>
              </Listener>
              <Listener>
                <CertificateThumbprint>certificate-thumbprint</CertificateThumbprint>
                <Protocol>listener-protocol</Protocol>
              </Listener>
            </Listeners>
          </WinRM>
          <AdminUsername>name-of-administrator-account</AdminUsername>
          <CustomData>base-64-encoded-data</CustomData>
          <AdditionalUnattendContent>
            <Passes>
              <UnattendPass>
                <PassName>name-of-pass</PassName>
                <Components>
                  <UnattendComponent>
                    <ComponentName>name-of-component</ComponentName>
                    <ComponentSettings>
                      <ComponentSetting>
                        <SettingName>name-of-setting</SettingName>
                        <Content>base-64-encoded-XML-content</Content>
                      </ComponentSetting>
                    </ComponentSettings>
                  </UnattendComponent>
                </Components>
              </UnattendPass>
            </Passes>
          </AdditionalUnattendContent>
        </ConfigurationSet>
        <ConfigurationSet i:type="LinuxProvisioningConfigurationSet">
          <ConfigurationSetType>LinuxProvisioningConfiguration</ConfigurationSetType>
          <HostName>host-name-for-the-virtual-machine</HostName>
          <UserName>new-user-name</UserName>
          <UserPassword>password-for-the-new-user</UserPassword>
          <DisableSshPasswordAuthentication>true</DisableSshPasswordAuthentication>
          <SSH>
            <PublicKeys>
              <PublicKey>
                <FingerPrint>certificate-fingerprint</FingerPrint>
                <Path>SSH-public-key-storage-location</Path>
              </PublicKey>
            </PublicKeys>
            <KeyPairs>
              <KeyPair>
                <FingerPrint>certificate-fingerprint</FingerPrint>
                <Path>SSH-public-key-storage-location</Path>
              </KeyPair>
            </KeyPairs>
          </SSH>
          <CustomData>base-64-encoded-data</CustomData>
        </ConfigurationSet>
        <ConfigurationSet>
          <ConfigurationSetType>NetworkConfiguration</ConfigurationSetType>
          <InputEndpoints>
            <InputEndpoint>
              <LoadBalancedEndpointSetName>name-of-load-balanced-set</LoadBalancedEndpointSetName>
              <LocalPort>22</LocalPort>
              <Name>ZZH</Name>
              <Port>33</Port>
              <LoadBalancerProbe>
                <Path>/probe/me</Path>
                <Port>80</Port>
                <Protocol>http</Protocol>
                <IntervalInSeconds>30</IntervalInSeconds>
                <TimeoutInSeconds>5</TimeoutInSeconds>
              </LoadBalancerProbe>
              <Protocol>endpoint-protocol</Protocol>
              <EnableDirectServerReturn>enable-direct-server-return</EnableDirectServerReturn>
              <EndpointACL>
                <Rules>
                  <Rule>
                    <Order>priority-of-the-rule</Order>
                    <Action>permit-rule</Action>
                    <RemoteSubnet>subnet-of-the-rule</RemoteSubnet>
                    <Description>description-of-the-rule</Description>
                  </Rule>
                </Rules>
              </EndpointACL>
              <LoadBalancerName>name-of-internal-loadbalancer</LoadBalancerName>
              <IdleTimeoutInMinutes>9</IdleTimeoutInMinutes>
            </InputEndpoint>
          </InputEndpoints>
          <SubnetNames>
            <SubnetName>name-of-subnet</SubnetName>
          </SubnetNames>
          <StaticVirtualNetworkIPAddress>ip-address</StaticVirtualNetworkIPAddress>
          <PublicIPs>
            <PublicIP>
              <Name>name-of-public-ip</Name>
              <IdleTimeoutInMinutes>11</IdleTimeoutInMinutes>
            </PublicIP>
          </PublicIPs>
        </ConfigurationSet>
      </ConfigurationSets>
      <ResourceExtensionReferences>
        <ResourceExtensionReference>
          <ReferenceName>name-of-reference</ReferenceName>
          <Publisher>name-of-publisher</Publisher>
          <Name>name-of-extension</Name>
          <Version>version-of-extension</Version>
          <ResourceExtensionParameterValues>
            <ResourceExtensionParameterValue>
              <Key>name-of-parameter-key</Key>
              <Value>parameter-value</Value>
              <Type>type-of-parameter</Type>
            </ResourceExtensionParameterValue>
          </ResourceExtensionParameterValues>
          <State>state-of-resource</State>
          <Certificates>
            <Certificate>
              <Thumbprint>certificate-thumbprint</Thumbprint>
              <ThumbprintAlgorithm>certificate-algorithm</ThumbprintAlgorithm>
            </Certificate>
          </Certificates>
        </ResourceExtensionReference>
      </ResourceExtensionReferences>
      <VMImageName>name-of-vm-image</VMImageName>
      <MediaLocation>path-to-vhd</MediaLocation>
      <AvailabilitySetName>name-of-availability-set</AvailabilitySetName>
      <DataVirtualHardDisks>
        <DataVirtualHardDisk>
          <HostCaching>caching-mode</HostCaching>
          <DiskLabel>label-of-data-disk</DiskLabel>
          <DiskName>name-of-disk</DiskName>
          <Lun>0</Lun>
          <LogicalDiskSizeInGB>50</LogicalDiskSizeInGB>
          <MediaLink>path-to-vhd</MediaLink>
        </DataVirtualHardDisk>
      </DataVirtualHardDisks>
      <OSVirtualHardDisk>
        <HostCaching>caching-mode</HostCaching>
        <DiskLabel>label-of-operating-system-disk</DiskLabel>
        <DiskName>name-of-disk</DiskName>
        <MediaLink>path-to-vhd</MediaLink>
        <SourceImageName>name-of-source-image</SourceImageName>
        <OS>operating-system-of-image</OS>
        <RemoteSourceImageLink>path-to-source-image</RemoteSourceImageLink>
        <ResizedSizeInGB>125</ResizedSizeInGB>
      </OSVirtualHardDisk>
      <RoleSize>size-of-virtual-machine</RoleSize>
      <ProvisionGuestAgent>true</ProvisionGuestAgent>
      <VMImageInput>
        <OSDiskConfiguration>
          <ResizedSizeInGB>126</ResizedSizeInGB>
        </OSDiskConfiguration>
        <DataDiskConfigurations>
          <DataDiskConfiguration>
            <Name>disk-name</Name>
            <ResizedSizeInGB>127</ResizedSizeInGB>
          </DataDiskConfiguration>
        </DataDiskConfigurations>
      </VMImageInput>
    </Role>
  </RoleList>
  <VirtualNetworkName>name-of-virtual-network</VirtualNetworkName>
  <Dns>
    <DnsServers>
      <DnsServer>
        <Name>dns-name</Name>
        <Address>dns-ip-address</Address>
      </DnsServer>
    </DnsServers>
  </Dns>
  <ReservedIPName>name-of-reserved-ip</ReservedIPName>
  <LoadBalancers>
    <LoadBalancer>
      <Name>name-of-internal-load-balancer</Name>
      <FrontendIpConfiguration>
        <Type>Private</Type>
        <SubnetName>name-of-subnet</SubnetName>
        <StaticVirtualNetworkIPAddress>static-ip-address</StaticVirtualNetworkIPAddress>
      </FrontendIpConfiguration>
    </LoadBalancer>
  </LoadBalancers>
</Deployment>`

	deployment := DeploymentRequest{}
	if err := xml.Unmarshal([]byte(xmlString), &deployment); err != nil {
		t.Fatal(err)
	}

	if deployment.Name != "name-of-deployment" {
		t.Fatalf("Expected deployment.Name=\"name-of-deployment\", but got \"%s\"",
			deployment.Name)
	}

	// ======

	t.Logf("deployment.RoleList[0]: %+v", deployment.RoleList[0])
	if expected := "name-of-the-virtual-machine"; deployment.RoleList[0].RoleName != expected {
		t.Fatalf("Expected deployment.RoleList[0].RoleName=%v, but got %v", expected, deployment.RoleList[0].RoleName)
	}

	// ======

	t.Logf("deployment.DNSServers[0]: %+v", deployment.DNSServers[0])
	if deployment.DNSServers[0].Name != "dns-name" {
		t.Fatalf("Expected deployment.DNSServers[0].Name=\"dns-name\", but got \"%s\"",
			deployment.DNSServers[0].Name)
	}

	// ======

	t.Logf("deployment.LoadBalancers[0]: %+v", deployment.LoadBalancers[0])
	if deployment.LoadBalancers[0].Name != "name-of-internal-load-balancer" {
		t.Fatalf("Expected deployment.LoadBalancers[0].Name=\"name-of-internal-load-balancer\", but got \"%s\"",
			deployment.LoadBalancers[0].Name)
	}

	if deployment.LoadBalancers[0].Type != IPAddressTypePrivate {
		t.Fatalf("Expected deployment.LoadBalancers[0].Type=IPAddressTypePrivate, but got \"%s\"",
			deployment.LoadBalancers[0].Type)
	}

	if deployment.LoadBalancers[0].StaticVirtualNetworkIPAddress != "static-ip-address" {
		t.Fatalf("Expected deployment.LoadBalancers[0].StaticVirtualNetworkIPAddress=\"static-ip-address\", but got \"%s\"",
			deployment.LoadBalancers[0].StaticVirtualNetworkIPAddress)
	}

	// ======

	extensionReferences := (*deployment.RoleList[0].ResourceExtensionReferences)
	t.Logf("(*deployment.RoleList[0].ResourceExtensionReferences)[0]: %+v", extensionReferences[0])
	if extensionReferences[0].Name != "name-of-extension" {
		t.Fatalf("Expected (*deployment.RoleList[0].ResourceExtensionReferences)[0].Name=\"name-of-extension\", but got \"%s\"",
			extensionReferences[0].Name)
	}

	if extensionReferences[0].ParameterValues[0].Key != "name-of-parameter-key" {
		t.Fatalf("Expected (*deployment.RoleList[0].ResourceExtensionReferences)[0].ParameterValues[0].Key=\"name-of-parameter-key\", but got %v",
			extensionReferences[0].ParameterValues[0].Key)
	}

	// ======

	if deployment.RoleList[0].VMImageInput.DataDiskConfigurations[0].ResizedSizeInGB != 127 {
		t.Fatalf("Expected deployment.RoleList[0].VMImageInput.DataDiskConfigurations[0].ResizedSizeInGB=127, but got %v",
			deployment.RoleList[0].VMImageInput.DataDiskConfigurations[0].ResizedSizeInGB)
	}

	// ======

	winRMlisteners := *deployment.RoleList[0].ConfigurationSets[0].WinRMListeners
	if string(winRMlisteners[0].Protocol) != "listener-protocol" {
		t.Fatalf("Expected winRMlisteners[0].Protocol to be listener-protocol, but got %s",
			string(winRMlisteners[0].Protocol))
	}

	winRMlisteners2 := *deployment.RoleList[0].ConfigurationSets[0].WinRMListeners
	if winRMlisteners2[1].CertificateThumbprint != "certificate-thumbprint" {
		t.Fatalf("Expected winRMlisteners2[1].CertificateThumbprint to be certificate-thumbprint, but got %s",
			winRMlisteners2[1].CertificateThumbprint)
	}

}
