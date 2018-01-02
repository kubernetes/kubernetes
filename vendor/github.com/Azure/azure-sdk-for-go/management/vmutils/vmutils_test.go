// +build go1.7

package vmutils

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

	vmdisk "github.com/Azure/azure-sdk-for-go/management/virtualmachinedisk"
)

func TestNewLinuxVmRemoteImage(t *testing.T) {
	role := NewVMConfiguration("myvm", "Standard_D3")
	ConfigureDeploymentFromRemoteImage(&role,
		"http://remote.host/some.vhd?sv=12&sig=ukhfiuwef78687", "Linux",
		"myvm-os-disk", "http://mystorageacct.blob.core.windows.net/vhds/mybrandnewvm.vhd",
		"OSDisk")
	ConfigureForLinux(&role, "myvm", "azureuser", "P@ssword", "2398yyKJGd78e2389ydfncuirowebhf89yh3IUOBY")
	ConfigureWithPublicSSH(&role)

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>myvm</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>LinuxProvisioningConfiguration</ConfigurationSetType>
      <StoredCertificateSettings></StoredCertificateSettings>
      <HostName>myvm</HostName>
      <UserName>azureuser</UserName>
      <UserPassword>P@ssword</UserPassword>
      <DisableSshPasswordAuthentication>false</DisableSshPasswordAuthentication>
      <SSH>
        <PublicKeys>
          <PublicKey>
            <Fingerprint>2398yyKJGd78e2389ydfncuirowebhf89yh3IUOBY</Fingerprint>
            <Path>/home/azureuser/.ssh/authorized_keys</Path>
          </PublicKey>
        </PublicKeys>
        <KeyPairs></KeyPairs>
      </SSH>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
    <ConfigurationSet>
      <ConfigurationSetType>NetworkConfiguration</ConfigurationSetType>
      <StoredCertificateSettings></StoredCertificateSettings>
      <InputEndpoints>
        <InputEndpoint>
          <LocalPort>22</LocalPort>
          <Name>SSH</Name>
          <Port>22</Port>
          <Protocol>TCP</Protocol>
        </InputEndpoint>
      </InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <OSVirtualHardDisk>
    <DiskLabel>OSDisk</DiskLabel>
    <DiskName>myvm-os-disk</DiskName>
    <MediaLink>http://mystorageacct.blob.core.windows.net/vhds/mybrandnewvm.vhd</MediaLink>
    <OS>Linux</OS>
    <RemoteSourceImageLink>http://remote.host/some.vhd?sv=12&amp;sig=ukhfiuwef78687</RemoteSourceImageLink>
  </OSVirtualHardDisk>
  <RoleSize>Standard_D3</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestNewLinuxVmPlatformImage(t *testing.T) {
	role := NewVMConfiguration("myplatformvm", "Standard_D3")
	ConfigureDeploymentFromPlatformImage(&role,
		"b39f27a8b8c64d52b05eac6a62ebad85__Ubuntu-14_04_2_LTS-amd64-server-20150309-en-us-30GB",
		"http://mystorageacct.blob.core.windows.net/vhds/mybrandnewvm.vhd", "mydisklabel")
	ConfigureForLinux(&role, "myvm", "azureuser", "", "2398yyKJGd78e2389ydfncuirdebhf89yh3IUOBY")

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>myplatformvm</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>LinuxProvisioningConfiguration</ConfigurationSetType>
      <StoredCertificateSettings></StoredCertificateSettings>
      <HostName>myvm</HostName>
      <UserName>azureuser</UserName>
      <SSH>
        <PublicKeys>
          <PublicKey>
            <Fingerprint>2398yyKJGd78e2389ydfncuirdebhf89yh3IUOBY</Fingerprint>
            <Path>/home/azureuser/.ssh/authorized_keys</Path>
          </PublicKey>
        </PublicKeys>
        <KeyPairs></KeyPairs>
      </SSH>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <OSVirtualHardDisk>
    <MediaLink>http://mystorageacct.blob.core.windows.net/vhds/mybrandnewvm.vhd</MediaLink>
    <SourceImageName>b39f27a8b8c64d52b05eac6a62ebad85__Ubuntu-14_04_2_LTS-amd64-server-20150309-en-us-30GB</SourceImageName>
  </OSVirtualHardDisk>
  <RoleSize>Standard_D3</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestNewVmFromVMImage(t *testing.T) {
	role := NewVMConfiguration("restoredbackup", "Standard_D1")
	ConfigureDeploymentFromPublishedVMImage(&role, "myvm-backup-20150209",
		"http://mystorageacct.blob.core.windows.net/vhds/myoldnewvm.vhd", false)

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>restoredbackup</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets></ConfigurationSets>
  <VMImageName>myvm-backup-20150209</VMImageName>
  <MediaLocation>http://mystorageacct.blob.core.windows.net/vhds/myoldnewvm.vhd</MediaLocation>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestNewVmFromExistingDisk(t *testing.T) {
	role := NewVMConfiguration("blobvm", "Standard_D14")
	ConfigureDeploymentFromExistingOSDisk(&role, "myvm-backup-20150209", "OSDisk")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWindowsToJoinDomain(&role, "user@domain.com", "youReN3verG0nnaGu3ss", "redmond.corp.contoso.com", "")
	ConfigureWithNewDataDisk(&role, "my-brand-new-disk", "http://account.blob.core.windows.net/vhds/newdatadisk.vhd",
		30, vmdisk.HostCachingTypeReadWrite)
	ConfigureWithExistingDataDisk(&role, "data-disk", vmdisk.HostCachingTypeReadOnly)

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>blobvm</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <DomainJoin>
        <Credentials>
          <Domain></Domain>
          <Username>user@domain.com</Username>
          <Password>youReN3verG0nnaGu3ss</Password>
        </Credentials>
        <JoinDomain>redmond.corp.contoso.com</JoinDomain>
      </DomainJoin>
      <StoredCertificateSettings></StoredCertificateSettings>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks>
    <DataVirtualHardDisk>
      <HostCaching>ReadWrite</HostCaching>
      <DiskLabel>my-brand-new-disk</DiskLabel>
      <LogicalDiskSizeInGB>30</LogicalDiskSizeInGB>
      <MediaLink>http://account.blob.core.windows.net/vhds/newdatadisk.vhd</MediaLink>
    </DataVirtualHardDisk>
    <DataVirtualHardDisk>
      <HostCaching>ReadOnly</HostCaching>
      <DiskName>data-disk</DiskName>
      <Lun>1</Lun>
    </DataVirtualHardDisk>
  </DataVirtualHardDisks>
  <OSVirtualHardDisk>
    <DiskLabel>OSDisk</DiskLabel>
    <DiskName>myvm-backup-20150209</DiskName>
  </OSVirtualHardDisk>
  <RoleSize>Standard_D14</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestWinRMOverHttps(t *testing.T) {
	role := NewVMConfiguration("winrmoverhttp", "Standard_D1")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWinRMOverHTTPS(&role, "abcdef")

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>winrmoverhttp</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <StoredCertificateSettings></StoredCertificateSettings>
      <WinRM>
        <Listeners>
          <Listener>
            <Protocol>Https</Protocol>
            <CertificateThumbprint>abcdef</CertificateThumbprint>
          </Listener>
        </Listeners>
      </WinRM>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestWinRMOverHttpsWithNoThumbprint(t *testing.T) {
	role := NewVMConfiguration("winrmoverhttp", "Standard_D1")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWinRMOverHTTPS(&role, "")

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>winrmoverhttp</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <StoredCertificateSettings></StoredCertificateSettings>
      <WinRM>
        <Listeners>
          <Listener>
            <Protocol>Https</Protocol>
          </Listener>
        </Listeners>
      </WinRM>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestWinRMOverHttp(t *testing.T) {
	role := NewVMConfiguration("winrmoverhttp", "Standard_D1")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWinRMOverHTTP(&role)

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>winrmoverhttp</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <StoredCertificateSettings></StoredCertificateSettings>
      <WinRM>
        <Listeners>
          <Listener>
            <Protocol>Http</Protocol>
          </Listener>
        </Listeners>
      </WinRM>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestSettingWinRMOverHttpTwice(t *testing.T) {
	role := NewVMConfiguration("winrmoverhttp", "Standard_D1")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWinRMOverHTTP(&role)
	ConfigureWinRMOverHTTP(&role)

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>winrmoverhttp</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <StoredCertificateSettings></StoredCertificateSettings>
      <WinRM>
        <Listeners>
          <Listener>
            <Protocol>Http</Protocol>
          </Listener>
        </Listeners>
      </WinRM>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}

func TestSettingWinRMOverHttpAndHttpsTwice(t *testing.T) {
	role := NewVMConfiguration("winrmoverhttp", "Standard_D1")
	ConfigureForWindows(&role, "WINVM", "azuser", "P2ssw@rd", true, "")
	ConfigureWinRMOverHTTP(&role)
	ConfigureWinRMOverHTTPS(&role, "")
	ConfigureWinRMOverHTTP(&role)
	ConfigureWinRMOverHTTPS(&role, "abcdef")

	bytes, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	expected := `<Role>
  <RoleName>winrmoverhttp</RoleName>
  <RoleType>PersistentVMRole</RoleType>
  <ConfigurationSets>
    <ConfigurationSet>
      <ConfigurationSetType>WindowsProvisioningConfiguration</ConfigurationSetType>
      <ComputerName>WINVM</ComputerName>
      <AdminPassword>P2ssw@rd</AdminPassword>
      <EnableAutomaticUpdates>true</EnableAutomaticUpdates>
      <StoredCertificateSettings></StoredCertificateSettings>
      <WinRM>
        <Listeners>
          <Listener>
            <Protocol>Http</Protocol>
          </Listener>
          <Listener>
            <Protocol>Https</Protocol>
            <CertificateThumbprint>abcdef</CertificateThumbprint>
          </Listener>
        </Listeners>
      </WinRM>
      <AdminUsername>azuser</AdminUsername>
      <InputEndpoints></InputEndpoints>
      <SubnetNames></SubnetNames>
      <PublicIPs></PublicIPs>
    </ConfigurationSet>
  </ConfigurationSets>
  <DataVirtualHardDisks></DataVirtualHardDisks>
  <RoleSize>Standard_D1</RoleSize>
  <ProvisionGuestAgent>true</ProvisionGuestAgent>
</Role>`

	if string(bytes) != expected {
		t.Fatalf("Expected marshalled xml to be %q, but got %q", expected, string(bytes))
	}
}
