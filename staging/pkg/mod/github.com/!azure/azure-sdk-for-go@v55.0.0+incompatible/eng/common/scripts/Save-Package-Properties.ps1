[CmdletBinding()]
Param (
  [Parameter(Mandatory=$True)]
  [string] $serviceDirectory,
  [Parameter(Mandatory=$True)]
  [string] $outDirectory
)

. (Join-Path $PSScriptRoot common.ps1)
$allPackageProperties = Get-AllPkgProperties $serviceDirectory
if ($allPackageProperties)
{
    if (-not (Test-Path -Path $outDirectory))
    {
      New-Item -ItemType Directory -Force -Path $outDirectory
    }
    foreach($pkg in $allPackageProperties)
    {
        if ($pkg.IsNewSdk)
        {
            Write-Host "Package Name: $($pkg.Name)"
            Write-Host "Package Version: $($pkg.Version)"
            Write-Host "Package SDK Type: $($pkg.SdkType)"
            Write-Host "Artifact Name: $($pkg.ArtifactName)"
            $configFilePrefix = $pkg.Name
            if ($pkg.ArtifactName)
            {
              $configFilePrefix = $pkg.ArtifactName
            }
            $outputPath = Join-Path -Path $outDirectory "$configFilePrefix.json"
            $outputObject = $pkg | ConvertTo-Json
            Set-Content -Path $outputPath -Value $outputObject
        }        
    }

    Get-ChildItem -Path $outDirectory
}
else
{
    Write-Error "Package properties are not available for service directory $($serviceDirectory)"
    exit 1
}
