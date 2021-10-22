# Generates an index page for cataloging different versions of the Docs
[CmdletBinding()]
Param (
    $DocFx,
    $RepoRoot,
    $DocGenDir,
    $DocOutDir = "${RepoRoot}/docfx_project",
    $DocfxJsonPath = "${PSScriptRoot}\docfx.json",
    $MainJsPath = "${PSScriptRoot}\templates\matthews\styles\main.js"
)
. "${PSScriptRoot}\..\scripts\common.ps1"

# Given the github io blob storage url and language regex,
# the helper function will return a list of artifact names.
function Get-BlobStorage-Artifacts($blobStorageUrl, $blobDirectoryRegex, $blobArtifactsReplacement) {
    LogDebug "Reading artifact from storage blob ..."
    $returnedArtifacts = @()
    $pageToken = ""
    Do {
      $resp = ""
      if (!$pageToken) {
        # First page call.
        $resp = Invoke-RestMethod -Method Get -Uri $blobStorageUrl
      }
      else {
        # Next page call
        $blobStorageUrlPageToken = $blobStorageUrl + "&marker=$pageToken"
        $resp = Invoke-RestMethod -Method Get -Uri $blobStorageUrlPageToken
      }
      # Convert to xml documents. 
      $xmlDoc = [xml](removeBomFromString $resp)
      foreach ($elem in $xmlDoc.EnumerationResults.Blobs.BlobPrefix) {
        # What service return like "dotnet/Azure.AI.Anomalydetector/", needs to fetch out "Azure.AI.Anomalydetector"
        $artifact = $elem.Name -replace $blobDirectoryRegex, $blobArtifactsReplacement
        $returnedArtifacts += $artifact
      }
      # Fetch page token
      $pageToken = $xmlDoc.EnumerationResults.NextMarker
    } while ($pageToken)
    return $returnedArtifacts
  }
  
# The sequence of Bom bytes differs by different encoding. 
# The helper function here is only to strip the utf-8 encoding system as it is used by blob storage list api.
# Return the original string if not in BOM utf-8 sequence.
function RemoveBomFromString([string]$bomAwareString) {
    if ($bomAwareString.length -le 3) {
        return $bomAwareString
    }
    $bomPatternByteArray = [byte[]] (0xef, 0xbb, 0xbf)
    # The default encoding for powershell is ISO-8859-1, so converting bytes with the encoding.
    $bomAwareBytes = [Text.Encoding]::GetEncoding(28591).GetBytes($bomAwareString.Substring(0, 3))
    if (@(Compare-Object $bomPatternByteArray $bomAwareBytes -SyncWindow 0).Length -eq 0) {
        return $bomAwareString.Substring(3)
    }
    return $bomAwareString
}
  
function Get-TocMapping { 
    Param (
        [Parameter(Mandatory = $true)] [Object[]] $metadata,
        [Parameter(Mandatory = $true)] [String[]] $artifacts
    )
    # Used for sorting the toc display order
    $orderServiceMapping = @{}

    foreach ($artifact in $artifacts) {
        $packageInfo = $metadata | ? { $_.Package -eq $artifact -and $_.Hide -ne "true" }
        $serviceName = ""
        $displayName = ""
        if (!$packageInfo) {
            LogDebug "There is no service name for artifact $artifact or it is marked as hidden. Please check csv of Azure/azure-sdk/_data/release/latest repo if this is intended. "
            continue
        }
        elseif (!$packageInfo[0].ServiceName) {
            LogWarning "There is no service name for artifact $artifact. Please check csv of Azure/azure-sdk/_data/release/latest repo if this is intended. "
            # If no service name retrieved, print out warning message, and put it into Other page.
            $serviceName = "Other"
            $displayName = $packageInfo[0].DisplayName.Trim()
        }
        else {
            if ($packageInfo.Length -gt 1) {
                LogWarning "There are more than 1 packages fetched out for artifact $artifact. Please check csv of Azure/azure-sdk/_data/release/latest repo if this is intended. "
            }
            $serviceName = $packageInfo[0].ServiceName.Trim()
            $displayName = $packageInfo[0].DisplayName.Trim()
        }
        $orderServiceMapping[$artifact] = @($serviceName, $displayName)
    }
    return $orderServiceMapping                   
}

function GenerateDocfxTocContent([Hashtable]$tocContent, [String]$lang) {
    LogDebug "Start generating the docfx toc and build docfx site..."

    LogDebug "Initializing Default DocFx Site..."
    & $($DocFx) init -q -o "${DocOutDir}"
    # The line below is used for testing in local
    #docfx init -q -o "${DocOutDir}"
    LogDebug "Copying template and configuration..."
    New-Item -Path "${DocOutDir}" -Name "templates" -ItemType "directory" -Force
    Copy-Item "${DocGenDir}/templates/*" -Destination "${DocOutDir}/templates" -Force -Recurse
    Copy-Item "${DocGenDir}/docfx.json" -Destination "${DocOutDir}/" -Force
    $YmlPath = "${DocOutDir}/api"
    New-Item -Path $YmlPath -Name "toc.yml" -Force
    $visitedService = @{}
    # Sort and display toc service name by alphabetical order, and then sort artifact by order.
    foreach ($serviceMapping in ($tocContent.GetEnumerator() | Sort-Object Value, Key)) {
        $artifact = $serviceMapping.Key
        $serviceName = $serviceMapping.Value[0]
        $displayName = $serviceMapping.Value[1]
        
        $fileName = ($serviceName -replace '\s', '').ToLower().Trim()
        if ($visitedService.ContainsKey($serviceName)) {
            if ($displayName) {
                Add-Content -Path "$($YmlPath)/${fileName}.md" -Value "#### $artifact`n##### ($displayName)"
            }
            else {
                Add-Content -Path "$($YmlPath)/${fileName}.md" -Value "#### $artifact"
            }
        }
        else {
            Add-Content -Path "$($YmlPath)/toc.yml" -Value "- name: ${serviceName}`r`n  href: ${fileName}.md"
            New-Item -Path $YmlPath -Name "${fileName}.md" -Force
            if ($displayName) {
                Add-Content -Path "$($YmlPath)/${fileName}.md" -Value "#### $artifact`n##### ($displayName)"
            }
            else {
                Add-Content -Path "$($YmlPath)/${fileName}.md" -Value "#### $artifact"
            }
            $visitedService[$serviceName] = $true
        }
    }

    # Generate toc homepage.
    LogDebug "Creating Site Title and Navigation..."
    New-Item -Path "${DocOutDir}" -Name "toc.yml" -Force
    Add-Content -Path "${DocOutDir}/toc.yml" -Value "- name: Azure SDK for $lang APIs`r`n  href: api/`r`n  homepage: api/index.md"

    LogDebug "Copying root markdowns"
    Copy-Item "$($RepoRoot)/README.md" -Destination "${DocOutDir}/api/index.md" -Force
    Copy-Item "$($RepoRoot)/CONTRIBUTING.md" -Destination "${DocOutDir}/api/CONTRIBUTING.md" -Force

    LogDebug "Building site..."
    & $($DocFx) build "${DocOutDir}/docfx.json"
    # The line below is used for testing in local
    #docfx build "${DocOutDir}/docfx.json"
    Copy-Item "${DocGenDir}/assets/logo.svg" -Destination "${DocOutDir}/_site/" -Force    
}

function UpdateDocIndexFiles {
    Param (
        [Parameter(Mandatory=$false)] [String]$appTitleLang = $Language,
        [Parameter(Mandatory=$false)] [String]$lang = $Language,
        [Parameter(Mandatory=$false)] [String]$packageRegex = "`"`"",
        [Parameter(Mandatory=$false)] [String]$regexReplacement = ""
    )
    # Update docfx.json
    $docfxContent = Get-Content -Path $DocfxJsonPath -Raw
    $docfxContent = $docfxContent -replace "`"_appTitle`": `"`"", "`"_appTitle`": `"Azure SDK for $appTitleLang`""
    $docfxContent = $docfxContent -replace "`"_appFooter`": `"`"", "`"_appFooter`": `"Azure SDK for $appTitleLang`""
    Set-Content -Path $DocfxJsonPath -Value $docfxContent -NoNewline
    # Update main.js var lang
    $mainJsContent = Get-Content -Path $MainJsPath -Raw
    $mainJsContent = $mainJsContent -replace "var SELECTED_LANGUAGE = ''", "var SELECTED_LANGUAGE = '$lang'"
    # Update main.js package regex and replacement
    $mainJsContent = $mainJsContent -replace "var PACKAGE_REGEX = ''", "var PACKAGE_REGEX = $packageRegex"
    $mainJsContent = $mainJsContent -replace "var PACKAGE_REPLACEMENT = ''", "var PACKAGE_REPLACEMENT = `"$regexReplacement`""
    Set-Content -Path $MainJsPath -Value $mainJsContent -NoNewline
}

if ($GetGithubIoDocIndexFn -and (Test-Path "function:$GetGithubIoDocIndexFn"))
{
    &$GetGithubIoDocIndexFn
}
else
{
    LogWarning "The function for 'GetGithubIoDocIndexFn' was not found.`
    Make sure it is present in eng/scripts/Language-Settings.ps1 and referenced in eng/common/scripts/common.ps1.`
    See https://github.com/Azure/azure-sdk-tools/blob/master/doc/common/common_engsys.md#code-structure"
}
