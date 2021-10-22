# given a CHANGELOG.md file, extract the relevant info we need to decorate a release
param (
  [Parameter(Mandatory = $true)]
  [String]$ChangeLogLocation,
  [String]$VersionString
)

$ErrorActionPreference = 'Stop'

$RELEASE_TITLE_REGEX = "(?<releaseNoteTitle>^\#+.*(?<version>\b\d+\.\d+\.\d+([^0-9\s][^\s:]+)?))"

$releaseNotes = @{}
$contentArrays = @{}
if ($ChangeLogLocation.Length -eq 0)
{
  return $releaseNotes
}

try 
{
  $contents = Get-Content $ChangeLogLocation

  # walk the document, finding where the version specifiers are and creating lists
  $version = ""
  foreach($line in $contents){
      if ($line -match $RELEASE_TITLE_REGEX)
      {
        $version = $matches["version"]
        $contentArrays[$version] = @()
      }

      $contentArrays[$version] += $line
  }

  # resolve each of discovered version specifier string arrays into real content
  foreach($key in $contentArrays.Keys)
  {
      $releaseNotes[$key] = New-Object PSObject -Property @{
      ReleaseVersion = $key
      ReleaseContent = $contentArrays[$key] -join [Environment]::NewLine
      }
  }
}
catch
{
  Write-Host "Error parsing $ChangeLogLocation."
  Write-Host $_.Exception.Message
}

if ([System.String]::IsNullOrEmpty($VersionString)) 
{
  return $releaseNotes
}
else 
{
  if ($releaseNotes.ContainsKey($VersionString)) 
  {
    $releaseNotesForVersion = $releaseNotes[$VersionString].ReleaseContent
    $processedNotes = $releaseNotesForVersion -Split [Environment]::NewLine | where { $_ -notmatch $RELEASE_TITLE_REGEX }
    return $processedNotes -Join [Environment]::NewLine
  }
  Write-Error "Release Notes for the Specified version ${VersionString} was not found"
  exit 1
}

