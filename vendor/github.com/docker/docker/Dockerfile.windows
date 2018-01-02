# escape=`

# -----------------------------------------------------------------------------------------
# This file describes the standard way to build Docker in a container on Windows
# Server 2016 or Windows 10.
#
# Maintainer: @jhowardmsft
# -----------------------------------------------------------------------------------------


# Prerequisites:
# --------------
#
# 1. Windows Server 2016 or Windows 10 with all Windows updates applied. The major 
#    build number must be at least 14393. This can be confirmed, for example, by 
#    running the following from an elevated PowerShell prompt - this sample output 
#    is from a fully up to date machine as at mid-November 2016:
#
#    >> PS C:\> $(gin).WindowsBuildLabEx
#    >> 14393.447.amd64fre.rs1_release_inmarket.161102-0100
#
# 2. Git for Windows (or another git client) must be installed. https://git-scm.com/download/win.
#
# 3. The machine must be configured to run containers. For example, by following
#    the quick start guidance at https://msdn.microsoft.com/en-us/virtualization/windowscontainers/quick_start/quick_start or
#    https://github.com/docker/labs/blob/master/windows/windows-containers/Setup.md
#
# 4. If building in a Hyper-V VM: For Windows Server 2016 using Windows Server
#    containers as the default option, it is recommended you have at least 1GB 
#    of memory assigned; For Windows 10 where Hyper-V Containers are employed, you
#    should have at least 4GB of memory assigned. Note also, to run Hyper-V 
#    containers in a VM, it is necessary to configure the VM for nested virtualization.

# -----------------------------------------------------------------------------------------


# Usage:
# -----
#
#  The following steps should be run from an (elevated*) Windows PowerShell prompt. 
#
#  (*In a default installation of containers on Windows following the quick-start guidance at
#    https://msdn.microsoft.com/en-us/virtualization/windowscontainers/quick_start/quick_start,
#    the docker.exe client must run elevated to be able to connect to the daemon).
#
# 1. Clone the sources from github.com:
#
#    >>   git clone https://github.com/docker/docker.git C:\go\src\github.com\docker\docker
#    >>   Cloning into 'C:\go\src\github.com\docker\docker'...
#    >>   remote: Counting objects: 186216, done.
#    >>   remote: Compressing objects: 100% (21/21), done.
#    >>   remote: Total 186216 (delta 5), reused 0 (delta 0), pack-reused 186195
#    >>   Receiving objects: 100% (186216/186216), 104.32 MiB | 8.18 MiB/s, done.
#    >>   Resolving deltas: 100% (123139/123139), done.
#    >>   Checking connectivity... done.
#    >>   Checking out files: 100% (3912/3912), done.
#    >>   PS C:\>
#
#
# 2. Change directory to the cloned docker sources:
#
#    >>   cd C:\go\src\github.com\docker\docker 
#
#
# 3. Build a docker image with the components required to build the docker binaries from source
#    by running one of the following:
#
#    >>   docker build -t nativebuildimage -f Dockerfile.windows .          
#    >>   docker build -t nativebuildimage -f Dockerfile.windows -m 2GB .    (if using Hyper-V containers)
#
#
# 4. Build the docker executable binaries by running one of the following:
#
#    >>   $DOCKER_GITCOMMIT=(git rev-parse --short HEAD)
#    >>   docker run --name binaries -e DOCKER_GITCOMMIT=$DOCKER_GITCOMMIT nativebuildimage hack\make.ps1 -Binary
#    >>   docker run --name binaries -e DOCKER_GITCOMMIT=$DOCKER_GITCOMMIT -m 2GB nativebuildimage hack\make.ps1 -Binary    (if using Hyper-V containers)
#
#
# 5. Copy the binaries out of the container, replacing HostPath with an appropriate destination 
#    folder on the host system where you want the binaries to be located.
#
#    >>   docker cp binaries:C:\go\src\github.com\docker\docker\bundles\docker.exe C:\HostPath\docker.exe
#    >>   docker cp binaries:C:\go\src\github.com\docker\docker\bundles\dockerd.exe C:\HostPath\dockerd.exe
#
#
# 6. (Optional) Remove the interim container holding the built executable binaries:
#
#    >>    docker rm binaries
#
#
# 7. (Optional) Remove the image used for the container in which the executable
#    binaries are build. Tip - it may be useful to keep this image around if you need to
#    build multiple times. Then you can take advantage of the builder cache to have an
#    image which has all the components required to build the binaries already installed.
#
#    >>    docker rmi nativebuildimage
#

# -----------------------------------------------------------------------------------------


#  The validation tests can only run directly on the host. This is because they calculate
#  information from the git repo, but the .git directory is not passed into the image as
#  it is excluded via .dockerignore. Run the following from a Windows PowerShell prompt
#  (elevation is not required): (Note Go must be installed to run these tests)
#
#    >>   hack\make.ps1 -DCO -PkgImports -GoFormat


# -----------------------------------------------------------------------------------------


#  To run unit tests, ensure you have created the nativebuildimage above. Then run one of
#  the following from an (elevated) Windows PowerShell prompt:
#
#    >>   docker run --rm nativebuildimage hack\make.ps1 -TestUnit
#    >>   docker run --rm -m 2GB nativebuildimage hack\make.ps1 -TestUnit    (if using Hyper-V containers)


# -----------------------------------------------------------------------------------------


#  To run unit tests and binary build, ensure you have created the nativebuildimage above. Then 
#  run one of the following from an (elevated) Windows PowerShell prompt:
#
#    >>   docker run nativebuildimage hack\make.ps1 -All
#    >>   docker run -m 2GB nativebuildimage hack\make.ps1 -All    (if using Hyper-V containers)

# -----------------------------------------------------------------------------------------


# Important notes:
# ---------------
#
# Don't attempt to use a bind-mount to pass a local directory as the bundles target
# directory. It does not work (golang attempts for follow a mapped folder incorrectly). 
# Instead, use docker cp as per the example.
#
# go.zip is not removed from the image as it is used by the Windows CI servers
# to ensure the host and image are running consistent versions of go.
#
# Nanoserver support is a work in progress. Although the image will build if the 
# FROM statement is updated, it will not work when running autogen through hack\make.ps1. 
# It is suspected that the required GCC utilities (eg gcc, windres, windmc) silently
# quit due to the use of console hooks which are not available.
#
# The docker integration tests do not currently run in a container on Windows, predominantly
# due to Windows not supporting privileged mode, so anything using a volume would fail.
# They (along with the rest of the docker CI suite) can be run using 
# https://github.com/jhowardmsft/docker-w2wCIScripts/blob/master/runCI/Invoke-DockerCI.ps1.
#
# -----------------------------------------------------------------------------------------


# The number of build steps below are explicitly minimised to improve performance.
FROM microsoft/windowsservercore

# Use PowerShell as the default shell
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Environment variable notes:
#  - GO_VERSION must be consistent with 'Dockerfile' used by Linux.
#  - FROM_DOCKERFILE is used for detection of building within a container.
ENV GO_VERSION=1.8.3 `
    GIT_VERSION=2.11.1 `
    GOPATH=C:\go `
    FROM_DOCKERFILE=1

RUN `
  Function Test-Nano() { `
    $EditionId = (Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Name 'EditionID').EditionId; `
    return (($EditionId -eq 'ServerStandardNano') -or ($EditionId -eq 'ServerDataCenterNano') -or ($EditionId -eq 'NanoServer')); `
  }`
  `
  Function Download-File([string] $source, [string] $target) { `
    if (Test-Nano) { `
      $handler = New-Object System.Net.Http.HttpClientHandler; `
      $client = New-Object System.Net.Http.HttpClient($handler); `
      $client.Timeout = New-Object System.TimeSpan(0, 30, 0); `
      $cancelTokenSource = [System.Threading.CancellationTokenSource]::new(); `
      $responseMsg = $client.GetAsync([System.Uri]::new($source), $cancelTokenSource.Token); `
      $responseMsg.Wait(); `
      if (!$responseMsg.IsCanceled) { `
        $response = $responseMsg.Result; `
        if ($response.IsSuccessStatusCode) { `
          $downloadedFileStream = [System.IO.FileStream]::new($target, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write); `
          $copyStreamOp = $response.Content.CopyToAsync($downloadedFileStream); `
          $copyStreamOp.Wait(); `
          $downloadedFileStream.Close(); `
          if ($copyStreamOp.Exception -ne $null) { throw $copyStreamOp.Exception } `
        } `
      } else { `
      Throw ("Failed to download " + $source) `
      }`
    } else { `
      $webClient = New-Object System.Net.WebClient; `
      $webClient.DownloadFile($source, $target); `
    } `
  } `
  `
  setx /M PATH $('C:\git\cmd;C:\git\usr\bin;'+$Env:PATH+';C:\gcc\bin;C:\go\bin'); `
  `
  Write-Host INFO: Downloading git...; `
  $location='https://www.nuget.org/api/v2/package/GitForWindows/'+$Env:GIT_VERSION; `
  Download-File $location C:\gitsetup.zip; `
  `
  Write-Host INFO: Downloading go...; `
  Download-File $('https://golang.org/dl/go'+$Env:GO_VERSION+'.windows-amd64.zip') C:\go.zip; `
  `
  Write-Host INFO: Downloading compiler 1 of 3...; `
  Download-File https://raw.githubusercontent.com/jhowardmsft/docker-tdmgcc/master/gcc.zip C:\gcc.zip; `
  `
  Write-Host INFO: Downloading compiler 2 of 3...; `
  Download-File https://raw.githubusercontent.com/jhowardmsft/docker-tdmgcc/master/runtime.zip C:\runtime.zip; `
  `
  Write-Host INFO: Downloading compiler 3 of 3...; `
  Download-File https://raw.githubusercontent.com/jhowardmsft/docker-tdmgcc/master/binutils.zip C:\binutils.zip; `
  `
  Write-Host INFO: Extracting git...; `
  Expand-Archive C:\gitsetup.zip C:\git-tmp; `
  New-Item -Type Directory C:\git | Out-Null; `
  Move-Item C:\git-tmp\tools\* C:\git\.; `
  Remove-Item -Recurse -Force C:\git-tmp; `
  `
  Write-Host INFO: Expanding go...; `
  Expand-Archive C:\go.zip -DestinationPath C:\; `
  `
  Write-Host INFO: Expanding compiler 1 of 3...; `
  Expand-Archive C:\gcc.zip -DestinationPath C:\gcc -Force; `
  Write-Host INFO: Expanding compiler 2 of 3...; `
  Expand-Archive C:\runtime.zip -DestinationPath C:\gcc -Force; `
  Write-Host INFO: Expanding compiler 3 of 3...; `
  Expand-Archive C:\binutils.zip -DestinationPath C:\gcc -Force; `
  `
  Write-Host INFO: Removing downloaded files...; `
  Remove-Item C:\gcc.zip; `
  Remove-Item C:\runtime.zip; `
  Remove-Item C:\binutils.zip; `
  Remove-Item C:\gitsetup.zip; `
  `
  Write-Host INFO: Creating source directory...; `
  New-Item -ItemType Directory -Path C:\go\src\github.com\docker\docker | Out-Null; `
  `
  Write-Host INFO: Configuring git core.autocrlf...; `
  C:\git\cmd\git config --global core.autocrlf true; `
  `
  Write-Host INFO: Completed

# Make PowerShell the default entrypoint
ENTRYPOINT ["powershell.exe"]

# Set the working directory to the location of the sources
WORKDIR C:\go\src\github.com\docker\docker

# Copy the sources into the container
COPY . .
