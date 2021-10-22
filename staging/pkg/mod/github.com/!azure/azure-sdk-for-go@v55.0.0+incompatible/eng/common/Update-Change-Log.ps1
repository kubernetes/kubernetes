# Note: This script will add or replace version title in change log

# Parameter description
# Version : Version to add or replace in change log
# ChangeLogPath: Path to change log file. If change log path is set to directory then script will probe for change log file in that path 
# Unreleased: Default is true. If it is set to false, then today's date will be set in verion title. If it is True then title will show "Unreleased"
# ReplaceVersion: This is useful when replacing current version title with new title.( Helpful to update the title before package release)

param (
  [Parameter(Mandatory = $true)]
  [String]$Version,
  [Parameter(Mandatory = $true)]
  [String]$ChangeLogPath,
  [String]$Unreleased = $True,
  [String]$ReplaceVersion = $False,
  [String]$ReleaseDate
)


$RELEASE_TITLE_REGEX = "(?<releaseNoteTitle>^\#+.*(?<version>\b\d+\.\d+\.\d+([^0-9\s][^\s:]+)?))"
$UNRELEASED_TAG = "(Unreleased)"
function Version-Matches($line)
{
    return ($line -match $RELEASE_TITLE_REGEX)
}

function Get-ChangelogPath($Path)
{ 
   # Check if CHANGELOG.md is present in path 
   $ChangeLogPath = Join-Path -Path $Path -ChildPath "CHANGELOG.md"
   if ((Test-Path -Path $ChangeLogPath) -eq $False) {
      # Check if change log exists with name HISTORY.md
      $ChangeLogPath = Join-Path -Path $Path -ChildPath "HISTORY.md"
      if ((Test-Path -Path $ChangeLogPath) -eq $False) {
         Write-Host "Change log is not found in path[$Path]"
         exit(1)
      }
   }
  
   Write-Host "Change log is found at path [$ChangeLogPath]"
   return $ChangeLogPath
}


function Get-VersionTitle($Version, $Unreleased)
{
   # Generate version title
   $newVersionTitle = "## $Version $UNRELEASED_TAG"
   if ($Unreleased -eq $False) {
      $actualReleaseDate = $ReleaseDate;

      if (!$actualReleaseDate) {
         $actualReleaseDate = Get-Date -Format "yyyy-MM-dd"
      }
      $newVersionTitle = "## $Version ($actualReleaseDate)"
   }
   return $newVersionTitle
}


function Get-NewChangeLog( [System.Collections.ArrayList]$ChangelogLines, $Version, $Unreleased, $ReplaceVersion)
{

   # version parameter is to pass new version to add or replace
   # Unreleased parameter can be set to False to set today's date instead of "Unreleased in title"
   # ReplaceVersion param can be set to true to replace current version title( useful at release time to change title)

   # find index of current version
   $Index = 0
   $CurrentTitle = ""
   $CurrentIndex = 0
   # Version increment tool passes replaceversion as False and Unreleased as True
   $is_version_increment = $ReplaceVersion -eq $False -and $Unreleased -eq $True

   for (; $Index -lt $ChangelogLines.Count; $Index++) {
      if (Version-Matches($ChangelogLines[$Index])) {
         # Find current title in change log
         if( -not $CurrentTitle) {
            $CurrentTitle = $ChangelogLines[$Index]
            $CurrentIndex = $Index
            Write-Host "Current Version title: $CurrentTitle"
         }

         # Ensure change log doesn't have new version when incrementing version
         # update change log script is triggered for all packages with current version for Java ( or any language where version is maintained in common file)
         # and this can cause an issue if someone changes changelog manually to prepare for release without updating actual version in central version file
         # Do not add new line or replace existing title when version is already present and script is triggered to add new line
         if ($is_version_increment -and $ChangelogLines[$Index].Contains($Version)) {
            Write-Host "Version is already present in change log."
            exit(0)
         }
      }
   }

   # Generate version title
   $newVersionTitle = Get-VersionTitle -Version $Version -Unreleased $Unreleased

   if( $newVersionTitle -eq $CurrentTitle) {
      Write-Host "No change is required in change log. Version is already present."
      exit(0)
   }

   if (($ReplaceVersion -eq $True) -and ($Unreleased -eq $False) -and $CurrentTitle.Contains($version) -and (-not $CurrentTitle.Contains($UNRELEASED_TAG)) -and (-not $ReleaseDate)) {
      Write-Host "Version is already present in change log with a release date."
      exit(0)
   }

   # if current version title already has new version then we should replace title to update it
   if ($CurrentTitle.Contains($Version) -and $ReplaceVersion -eq $False) {
      Write-Host "Version is already present in title. Updating version title"
      $ReplaceVersion = $True
   }

   # if version is already found and not replacing then nothing to do
   if ($ReplaceVersion -eq $False) {
      Write-Host "Adding version title $newVersionTitle"
      $ChangelogLines.insert($CurrentIndex, "")
      $ChangelogLines.insert($CurrentIndex, "")
      $ChangelogLines.insert($CurrentIndex, $newVersionTitle)
   }
   else{
      # Script is executed to replace an existing version title
      Write-Host "Replacing current version title to $newVersionTitle"
      $ChangelogLines[$CurrentIndex] = $newVersionTitle
   }

   return $ChangelogLines
}


# Make sure path is valid
if ((Test-Path -Path $ChangeLogPath) -eq $False) {
   Write-Host "Change log path is invalid. [$ChangeLogPath]"
   exit(1)
}

# probe change log path if path is directory 
if (Test-Path -Path $ChangeLogPath -PathType Container) {
   $ChangeLogPath = Get-ChangelogPath -Path $ChangeLogPath
}

# Read current change logs and add/update version
$ChangelogLines = [System.Collections.ArrayList](Get-Content -Path $ChangeLogPath)

if ($null -eq $ChangelogLines) { 
   $ChangelogLines = @() 
}

$NewContents = Get-NewChangeLog -ChangelogLines $ChangelogLines -Version $Version -Unreleased $Unreleased -ReplaceVersion $ReplaceVersion

Write-Host "Writing change log to file [$ChangeLogPath]"
Set-Content -Path $ChangeLogPath $NewContents
Write-Host "Version is added/updated in change log"
