<!--[metadata]>
+++
title = "PowerShell DSC Usage"
description = "Using DSC to configure a new Docker host"
keywords = ["powershell, dsc, installation, usage, docker,  documentation"]
[menu.main]
parent = "smn_win_osx"
+++
<![end-metadata]-->

# Using PowerShell DSC

Windows PowerShell Desired State Configuration (DSC) is a configuration
management tool that extends the existing functionality of Windows PowerShell.
DSC uses a declarative syntax to define the state in which a target should be
configured. More information about PowerShell DSC can be found at
[http://technet.microsoft.com/en-us/library/dn249912.aspx](http://technet.microsoft.com/en-us/library/dn249912.aspx).

## Requirements

To use this guide you'll need a Windows host with PowerShell v4.0 or newer.

The included DSC configuration script also uses the official PPA so
only an Ubuntu target is supported. The Ubuntu target must already have the
required OMI Server and PowerShell DSC for Linux providers installed. More
information can be found at [https://github.com/MSFTOSSMgmt/WPSDSCLinux](https://github.com/MSFTOSSMgmt/WPSDSCLinux).
The source repository listed below also includes PowerShell DSC for Linux
installation and init scripts along with more detailed installation information.

## Installation

The DSC configuration example source is available in the following repository:
[https://github.com/anweiss/DockerClientDSC](https://github.com/anweiss/DockerClientDSC). It can be cloned with:

    $ git clone https://github.com/anweiss/DockerClientDSC.git

## Usage

The DSC configuration utilizes a set of shell scripts to determine whether or
not the specified Docker components are configured on the target node(s). The
source repository also includes a script (`RunDockerClientConfig.ps1`) that can
be used to establish the required CIM session(s) and execute the
`Set-DscConfiguration` cmdlet.

More detailed usage information can be found at
[https://github.com/anweiss/DockerClientDSC](https://github.com/anweiss/DockerClientDSC).

### Install Docker
The Docker installation configuration is equivalent to running:

```
apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys\
36A1D7869245C8950F966E92D8576A8BA88D21E9
sh -c "echo deb https://get.docker.com/ubuntu docker main\
> /etc/apt/sources.list.d/docker.list"
apt-get update
apt-get install lxc-docker
```

Ensure that your current working directory is set to the `DockerClientDSC`
source and load the DockerClient configuration into the current PowerShell
session

```powershell
. .\DockerClient.ps1
```

Generate the required DSC configuration .mof file for the targeted node

```powershell
DockerClient -Hostname "myhost"
```

A sample DSC configuration data file has also been included and can be modified
and used in conjunction with or in place of the `Hostname` parameter:

```powershell
DockerClient -ConfigurationData .\DockerConfigData.psd1
```

Start the configuration application process on the targeted node

```powershell
.\RunDockerClientConfig.ps1 -Hostname "myhost"
```

The `RunDockerClientConfig.ps1` script can also parse a DSC configuration data
file and execute configurations against multiple nodes as such:

```powershell
.\RunDockerClientConfig.ps1 -ConfigurationData .\DockerConfigData.psd1
```

### Images
Image configuration is equivalent to running: `docker pull [image]` or
`docker rmi -f [IMAGE]`.

Using the same steps defined above, execute `DockerClient` with the `Image`
parameter and apply the configuration:

```powershell
DockerClient -Hostname "myhost" -Image "node"
.\RunDockerClientConfig.ps1 -Hostname "myhost"
```

You can also configure the host to pull multiple images:

```powershell
DockerClient -Hostname "myhost" -Image "node","mongo"
.\RunDockerClientConfig.ps1 -Hostname "myhost"
```

To remove images, use a hashtable as follows:

```powershell
DockerClient -Hostname "myhost" -Image @{Name="node"; Remove=$true}
.\RunDockerClientConfig.ps1 -Hostname $hostname
```

### Containers
Container configuration is equivalent to running:

```
docker run -d --name="[containername]" -p '[port]' -e '[env]' --link '[link]'\
'[image]' '[command]'
```
or

```
docker rm -f [containername]
```

To create or remove containers, you can use the `Container` parameter with one
or more hashtables. The hashtable(s) passed to this parameter can have the
following properties:

- Name (required)
- Image (required unless Remove property is set to `$true`)
- Port
- Env
- Link
- Command
- Remove

For example, create a hashtable with the settings for your container:

```powershell
$webContainer = @{Name="web"; Image="anweiss/docker-platynem"; Port="80:80"}
```

Then, using the same steps defined above, execute
`DockerClient` with the `-Image` and `-Container` parameters:

```powershell
DockerClient -Hostname "myhost" -Image node -Container $webContainer
.\RunDockerClientConfig.ps1 -Hostname "myhost"
```

Existing containers can also be removed as follows:

```powershell
$containerToRemove = @{Name="web"; Remove=$true}
DockerClient -Hostname "myhost" -Container $containerToRemove
.\RunDockerClientConfig.ps1 -Hostname "myhost"
```

Here is a hashtable with all of the properties that can be used to create a
container:

```powershell
$containerProps = @{Name="web"; Image="node:latest"; Port="80:80"; `
Env="PORT=80"; Link="db:db"; Command="grunt"}
```
