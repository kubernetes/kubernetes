
$ReleaseDevOpsOrgParameters =  @("--organization", "https://dev.azure.com/azure-sdk")
$ReleaseDevOpsCommonParameters =  $ReleaseDevOpsOrgParameters + @("--output", "json")
$ReleaseDevOpsCommonParametersWithProject = $ReleaseDevOpsCommonParameters + @("--project", "Release")

function Invoke-AzBoardsCmd($subCmd, $parameters, $output = $true)
{
  $azCmdStr = "az boards ${subCmd} $($parameters -join ' ')"
  if ($output) {
    Write-Host $azCmdStr
  }
  return Invoke-Expression "$azCmdStr" | ConvertFrom-Json -AsHashTable
}

function LoginToAzureDevops([string]$devops_pat)
{
  if (!$devops_pat) {
    return
  }
  $azCmdStr = "'$devops_pat' | az devops login $($ReleaseDevOpsOrgParameters -join ' ')"
  Invoke-Expression $azCmdStr
}

function BuildHashKeyNoNull()
{
  $filterNulls = $args | Where-Object { $_ }
  # if we had any nulls then return null
  if (!$filterNulls -or $args.Count -ne $filterNulls.Count) {
    return $null
  }
  return BuildHashKey $args
}

function BuildHashKey()
{
  # if no args or the first arg is null return null
  if ($args.Count -lt 1 -or !$args[0]) {
    return $null
  }

  # exclude null values
  $keys = $args | Where-Object { $_ }
  return $keys -join "|"
}

$parentWorkItems = @{}
function FindParentWorkItem($serviceName, $packageDisplayName, $outputCommand = $false)
{
  $key = BuildHashKey $serviceName $packageDisplayName
  if ($key -and $parentWorkItems.ContainsKey($key)) {
    return $parentWorkItems[$key]
  }

  if ($serviceName) {
    $serviceCondition = "[ServiceName] = '${serviceName}'"
    if ($packageDisplayName) {
      $serviceCondition += " AND [PackageDisplayName] = '${packageDisplayName}'"
    }
    else {
      $serviceCondition += " AND [PackageDisplayName] = ''"
    }
  }
  else {
    $serviceCondition = "[ServiceName] <> ''"
  }

  $parameters = $ReleaseDevOpsCommonParametersWithProject
  $parameters += "--wiql"
  $parameters += "`"SELECT [ID], [ServiceName], [PackageDisplayName], [Parent] FROM WorkItems WHERE [Work Item Type] = 'Epic' AND ${serviceCondition}`""

  $workItems = Invoke-AzBoardsCmd "query" $parameters $outputCommand

  foreach ($wi in $workItems) {
    $localKey = BuildHashKey $wi.fields["Custom.ServiceName"] $wi.fields["Custom.PackageDisplayName"]
    if (!$localKey) { continue }
    if ($parentWorkItems.ContainsKey($localKey) -and $parentWorkItems[$localKey].id -ne $wi.id) {
      Write-Warning "Already found parent [$($parentWorkItems[$localKey].id)] with key [$localKey], using that one instead of [$($wi.id)]."
    }
    else {
      Write-Verbose "[$($wi.id)]$localKey - Cached"
      $parentWorkItems[$localKey] = $wi
    }
  }

  if ($key -and $parentWorkItems.ContainsKey($key)) {
    return $parentWorkItems[$key]
  }
  return $null
}

$packageWorkItems = @{}
$packageWorkItemWithoutKeyFields = @{}

function FindLatestPackageWorkItem($lang, $packageName, $outputCommand = $true)
{
  # Cache all the versions of this package and language work items
  $null = FindPackageWorkItem $lang $packageName -includeClosed $true -outputCommand $outputCommand

  $latestWI = $null
  foreach ($wi in $packageWorkItems.Values)
  {
    if ($wi.fields["Custom.Language"] -ne $lang) { continue }
    if ($wi.fields["Custom.Package"] -ne $packageName) { continue }

    if (!$latestWI) {
      $latestWI = $wi
      continue
    }

    # Note this only does string sorting which is enough for our current usages
    # if we need absolute sorting at some point we would need to parse these versions
    if ($wi.fields["Custom.PackageVersionMajorMinor"] -gt $latestWI.fields["Custom.PackageVersionMajorMinor"]) {
      $latestWI = $wi
    }
  }
  return $latestWI
}

function FindPackageWorkItem($lang, $packageName, $version, $outputCommand = $true, $includeClosed = $false)
{
  $key = BuildHashKeyNoNull $lang $packageName $version
  if ($key -and $packageWorkItems.ContainsKey($key)) {
    return $packageWorkItems[$key]
  }

  $fields = @()
  $fields += "ID"
  $fields += "State"
  $fields += "System.AssignedTo"
  $fields += "Parent"
  $fields += "Language"
  $fields += "Package"
  $fields += "PackageDisplayName"
  $fields += "Title"
  $fields += "PackageType"
  $fields += "PackageTypeNewLibrary"
  $fields += "PackageVersionMajorMinor"
  $fields += "PackageRepoPath"
  $fields += "ServiceName"
  $fields += "Planned Packages"
  $fields += "Shipped Packages"
  $fields += "PackageBetaVersions"
  $fields += "PackageGAVersion"
  $fields += "PackagePatchVersions"
  $fields += "Generated"
  $fields += "RoadmapState"

  $fieldList = ($fields | ForEach-Object { "[$_]"}) -join ", "
  $query = "SELECT ${fieldList} FROM WorkItems WHERE [Work Item Type] = 'Package'"

  if (!$includeClosed -and !$lang) {
    $query += " AND [State] <> 'No Active Development' AND [PackageTypeNewLibrary] = true"
  }
  if ($lang) {
    $query += " AND [Language] = '${lang}'"
  }
  if ($packageName) {
    $query += " AND [Package] = '${packageName}'"
  }
  if ($version) {
    $query += " AND [PackageVersionMajorMinor] = '${version}'"
  }
  $parameters = $ReleaseDevOpsCommonParametersWithProject
  $parameters += "--wiql", "`"${query}`""

  $workItems = Invoke-AzBoardsCmd "query" $parameters $outputCommand

  if ($workItems -and $workItems.Count -eq 1000) {
    Write-Warning "Retrieved the max of 1000 items so item list might not be complete."
  }

  foreach ($wi in $workItems)
  {
    $localKey = BuildHashKeyNoNull $wi.fields["Custom.Language"] $wi.fields["Custom.Package"] $wi.fields["Custom.PackageVersionMajorMinor"]
    if (!$localKey) {
      $packageWorkItemWithoutKeyFields[$wi.id] = $wi
      Write-Host "Skipping package [$($wi.id)]$($wi.fields['System.Title']) which is missing required fields language, package, or version."
      continue
    }
    if ($packageWorkItems.ContainsKey($localKey) -and $packageWorkItems[$localKey].id -ne $wi.id) {
      Write-Warning "Already found package [$($packageWorkItems[$localKey].id)] with key [$localKey], using that one instead of [$($wi.id)]."
    }
    else {
      Write-Verbose "Caching package [$($wi.id)] for [$localKey]"
      $packageWorkItems[$localKey] = $wi
    }
  }

  if ($key -and $packageWorkItems.ContainsKey($key)) {
    return $packageWorkItems[$key]
  }
  return $null
}

function InitializeWorkItemCache($outputCommand = $true, $includeClosed = $false)
{
  # Pass null to cache all service parents
  $null = FindParentWorkItem -serviceName $null -packageDisplayName $null -outputCommand $outputCommand

  # Pass null to cache all the package items
  $null = FindPackageWorkItem -lang $null -packageName $null -version $null -outputCommand $outputCommand -includeClosed $includeClosed
}

function GetCachedPackageWorkItems()
{
  return $packageWorkItems.Values
}

function UpdateWorkItemParent($childWorkItem, $parentWorkItem, $outputCommand = $true)
{
  $childId = $childWorkItem.id
  $existingParentId = $childWorkItem.fields["System.Parent"]
  $newParentId = $parentWorkItem.id

  if ($existingParentId -eq $newParentId) {
    return
  }

  CreateWorkItemParent $childId $newParentId $existingParentId -outputCommand $outputCommand
  $childWorkItem.fields["System.Parent"] = $newParentId
}

function CreateWorkItemParent($id, $parentId, $oldParentId, $outputCommand = $true)
{
  # Have to remove old parent first if you want to add a new parent.
  if ($oldParentId)
  {
     $parameters = $ReleaseDevOpsCommonParameters
     $parameters += "--yes"
     $parameters += "--id", $id
     $parameters += "--relation-type", "parent"
     $parameters += "--target-id", $oldParentId

     Invoke-AzBoardsCmd "work-item relation remove" $parameters $outputCommand | Out-Null
  }

  $parameters = $ReleaseDevOpsCommonParameters
  $parameters += "--id", $id
  $parameters += "--relation-type", "parent"
  $parameters += "--target-id", $parentId

  Invoke-AzBoardsCmd "work-item relation add" $parameters $outputCommand | Out-Null
}
function CreateWorkItem($title, $type, $iteration, $area, $fields, $assignedTo, $parentId, $outputCommand = $true)
{
  $parameters = $ReleaseDevOpsCommonParametersWithProject
  $parameters += "--title", "`"${title}`""
  $parameters += "--type", "`"${type}`""
  $parameters += "--iteration", "`"${iteration}`""
  $parameters += "--area", "`"${area}`""
  if ($assignedTo) {
    $parameters += "--assigned-to", "`"${assignedTo}`""
  }
  if ($fields) {
    $parameters += "--fields"
    $parameters += $fields
  }

  $workItem = Invoke-AzBoardsCmd "work-item create" $parameters $outputCommand

  if ($parentId) {
    $parameters = $ReleaseDevOpsCommonParameters
    $parameters += "--id", $workItem.id
    $parameters += "--relation-type", "parent"
    $parameters += "--target-id", $parentId

    Invoke-AzBoardsCmd "work-item relation add" $parameters $outputCommand | Out-Null
  }

  return $workItem
}

function UpdateWorkItem($id, $fields, $title, $state, $assignedTo, $outputCommand = $true)
{
  $parameters = $ReleaseDevOpsCommonParameters
  $parameters += "--id", $id
  if ($title) {
    $parameters += "--title", "`"${title}`""
  }
  if ($state) {
    $parameters += "--state", "`"${state}`""
  }
  if ($assignedTo) {
    $parameters += "--assigned-to", "`"${assignedTo}`""
  }
  if ($fields) {
    $parameters += "--fields"
    $parameters += $fields
  }

  return Invoke-AzBoardsCmd "work-item update" $parameters $outputCommand
}

function UpdatePackageWorkItemReleaseState($id, $state, $releaseType, $outputCommand = $true)
{
  $fields = "`"Custom.ReleaseType=${releaseType}`""
  return UpdateWorkItem -id $id -state $state -fields $fields -outputCommand $outputCommand
}

function FindOrCreateClonePackageWorkItem($lang, $pkg, $verMajorMinor, $allowPrompt = $false, $outputCommand = $false)
{
  $workItem = FindPackageWorkItem -lang $lang -packageName $pkg.Package -version $verMajorMinor -includeClosed $true -outputCommand $outputCommand

  if (!$workItem) {
    $latestVersionItem = FindLatestPackageWorkItem -lang $lang -packageName $pkg.Package -outputCommand $outputCommand
    $assignedTo = "me"
    $extraFields = @()
    if ($latestVersionItem) {
      Write-Verbose "Copying data from latest matching [$($latestVersionItem.id)] with version $($latestVersionItem.fields["Custom.PackageVersionMajorMinor"])"
      if ($latestVersionItem.fields["System.AssignedTo"]) {
        $assignedTo = $latestVersionItem.fields["System.AssignedTo"]["uniqueName"]
      }
      $pkg.DisplayName = $latestVersionItem.fields["Custom.PackageDisplayName"]
      $pkg.ServiceName = $latestVersionItem.fields["Custom.ServiceName"]
      if (!$pkg.RepoPath -and $pkg.RepoPath -ne "NA" -and $pkg.fields["Custom.PackageRepoPath"]) {
        $pkg.RepoPath = $pkg.fields["Custom.PackageRepoPath"]
      }

      $extraFields += "`"Generated=" + $latestVersionItem.fields["Custom.Generated"] + "`""
      $extraFields += "`"RoadmapState=" +  $latestVersionItem.fields["Custom.RoadmapState"] + "`""
    }

    if ($allowPrompt) {
      if (!$pkg.DisplayName) {
        Write-Host "We need a package display name to be used in various places and it should be consistent across languages for similar packages."
        while (($readInput = Read-Host -Prompt "Input the display name") -eq "") { }
        $packageInfo.DisplayName = $readInput
      }

      if (!$pkg.ServiceName) {
        Write-Host "We need a package service name to be used in various places and it should be consistent across languages for similar packages."
        while (($readInput = Read-Host -Prompt "Input the service name") -eq "") { }
        $packageInfo.ServiceName = $readInput
      }
    }


    $workItem = CreateOrUpdatePackageWorkItem $lang $pkg $verMajorMinor -existingItem $null -assignedTo $assignedTo -extraFields $extraFields -outputCommand $outputCommand
  }

  return $workItem
}

function CreateOrUpdatePackageWorkItem($lang, $pkg, $verMajorMinor, $existingItem, $assignedTo = $null, $extraFields = $null, $outputCommand = $true)
{
  if (!$lang -or !$pkg -or !$verMajorMinor) {
    Write-Host "Cannot create or update because one of lang, pkg or verMajorMinor aren't set. [$lang|$($pkg.Package)|$verMajorMinor]"
    return
  }
  $pkgName = $pkg.Package
  $pkgDisplayName = $pkg.DisplayName
  $pkgType = $pkg.Type
  $pkgNewLibrary = $pkg.New
  $pkgRepoPath = $pkg.RepoPath
  $serviceName = $pkg.ServiceName
  $title = $lang + " - " + $pkg.DisplayName + " - " + $verMajorMinor

  $fields = @()
  $fields += "`"Language=${lang}`""
  $fields += "`"Package=${pkgName}`""
  $fields += "`"PackageDisplayName=${pkgDisplayName}`""
  $fields += "`"PackageType=${pkgType}`""
  $fields += "`"PackageTypeNewLibrary=${pkgNewLibrary}`""
  $fields += "`"PackageVersionMajorMinor=${verMajorMinor}`""
  $fields += "`"ServiceName=${serviceName}`""
  $fields += "`"PackageRepoPath=${pkgRepoPath}`""

  if ($extraFields) {
    $fields += $extraFields
  }

  if ($existingItem)
  {
    $changedField = $null

    if ($lang -ne $existingItem.fields["Custom.Language"]) { $changedField = "Custom.Language" }
    if ($pkgName -ne $existingItem.fields["Custom.Package"]) { $changedField = "Custom.Package" }
    if ($verMajorMinor -ne $existingItem.fields["Custom.PackageVersionMajorMinor"]) { $changedField = "Custom.PackageVersionMajorMinor" }
    if ($pkgDisplayName -ne $existingItem.fields["Custom.PackageDisplayName"]) { $changedField = "Custom.PackageDisplayName" }
    if ($pkgType -ne $existingItem.fields["Custom.PackageType"]) { $changedField = "Custom.PackageType" }
    if ($pkgNewLibrary -ne $existingItem.fields["Custom.PackageTypeNewLibrary"]) { $changedField = "Custom.PackageTypeNewLibrary" }
    if ($pkgRepoPath -ne $existingItem.fields["Custom.PackageRepoPath"]) { $changedField = "Custom.PackageRepoPath" }
    if ($serviceName -ne $existingItem.fields["Custom.ServiceName"]) { $changedField = "Custom.ServiceName" }
    if ($title -ne $existingItem.fields["System.Title"]) { $changedField = "System.Title" }

    if ($changedField) {
      Write-Host "At least field $changedField ($($existingItem.fields[$changedField])) changed so updating."
    }

    if ($changedField) {
      $beforeState = $existingItem.fields["System.State"]

      # Need to set to New to be able to update
      $existingItem = UpdateWorkItem -id $existingItem.id -fields $fields -title $title -state "New" -assignedTo $assignedTo -outputCommand $outputCommand
      Write-Host "[$($existingItem.id)]$lang - $pkgName($verMajorMinor) - Updated"

      if ($beforeState -ne $existingItem.fields['System.State']) {
        Write-Verbose "Resetting state for [$($existingItem.id)] from '$($existingItem.fields['System.State'])' to '$beforeState'"
        $existingItem = UpdateWorkItem $existingItem.id -state $beforeState -outputCommand $outputCommand
      }
    }

    $newparentItem = FindOrCreatePackageGroupParent $serviceName $pkgDisplayName -outputCommand $false
    UpdateWorkItemParent $existingItem $newParentItem -outputCommand $outputCommand
    return $existingItem
  }

  $parentItem = FindOrCreatePackageGroupParent $serviceName $pkgDisplayName -outputCommand $false
  $workItem = CreateWorkItem $title "Package" "Release" "Release" $fields $assignedTo $parentItem.id -outputCommand $outputCommand
  Write-Host "[$($workItem.id)]$lang - $pkgName($verMajorMinor) - Created"
  return $workItem
}

function FindOrCreatePackageGroupParent($serviceName, $packageDisplayName, $outputCommand = $true)
{
  $existingItem = FindParentWorkItem $serviceName $packageDisplayName -outputCommand $outputCommand
  if ($existingItem) {
    $newparentItem = FindOrCreateServiceParent $serviceName -outputCommand $outputCommand
    UpdateWorkItemParent $existingItem $newParentItem
    return $existingItem
  }

  $fields = @()
  $fields += "`"PackageDisplayName=${packageDisplayName}`""
  $fields += "`"ServiceName=${serviceName}`""
  $serviceParentItem = FindOrCreateServiceParent $serviceName -outputCommand $outputCommand
  $workItem = CreateWorkItem $packageDisplayName "Epic" "Release" "Release" $fields $null $serviceParentItem.id

  $localKey = BuildHashKey $serviceName $packageDisplayName
  Write-Host "[$($workItem.id)]$localKey - Created Parent"
  $parentWorkItems[$localKey] = $workItem
  return $workItem
}

function FindOrCreateServiceParent($serviceName, $outputCommand = $true)
{
  $serviceParent = FindParentWorkItem $serviceName -outputCommand $outputCommand
  if ($serviceParent) {
    return $serviceParent
  }

  $fields = @()
  $fields += "`"PackageDisplayName=`""
  $fields += "`"ServiceName=${serviceName}`""
  $parentId = $null
  $workItem = CreateWorkItem $serviceName "Epic" "Release" "Release" $fields $null $parentId -outputCommand $outputCommand

  $localKey = BuildHashKey $serviceName
  Write-Host "[$($workItem.id)]$localKey - Created"
  $parentWorkItems[$localKey] = $workItem
  return $workItem
}

function ParseVersionSetFromMDField([string]$field)
{
  $MDTableRegex = "\|\s*(?<t>\S*)\s*\|\s*(?<v>\S*)\s*\|\s*(?<d>\S*)\s*\|"
  $versionSet = @{}
  $tableMatches = [Regex]::Matches($field, $MDTableRegex)

  foreach ($match in $tableMatches)
  {
    if ($match.Groups["t"].Value -eq "Type" -or $match.Groups["t"].Value -eq "-") {
      continue
    }
    $version = New-Object PSObject -Property @{
      Type = $match.Groups["t"].Value
      Version = $match.Groups["v"].Value
      Date = $match.Groups["d"].Value
    }
    if (!$versionSet.ContainsKey($version.Version)) {
      $versionSet[$version.Version] = $version
    }
  }
  return $versionSet
}

function GetTextVersionFields($versionList, $pkgWorkItem)
{
  $betaVersions = $gaVersions = $patchVersions = ""
  foreach ($v in $versionList) {
    $vstr = "$($v.Version),$($v.Date)"
    if ($v.Type -eq "Beta") {
      if ($betaVersions.Length + $vstr.Length -lt 255) {
        if ($betaVersions.Length -gt 0) { $betaVersions += "|" }
        $betaVersions += $vstr
      }
    }
    elseif ($v.Type -eq "GA") {
      if ($gaVersions.Length + $vstr.Length -lt 255) {
        if ($gaVersions.Length -gt 0) { $gaVersions += "|" }
        $gaVersions += $vstr
      }
    }
    elseif ($v.Type -eq "Patch") {
      if ($patchVersions.Length + $vstr.Length -lt 255) {
        if ($patchVersions.Length -gt 0) { $patchVersions += "|" }
        $patchVersions += $vstr
      }
    }
  }

  $fieldUpdates = @()
  if ("$($pkgWorkItem.fields["Custom.PackageBetaVersions"])" -ne $betaVersions)
  {
    $fieldUpdates += @"
{
  "op": "replace",
  "path": "/fields/PackageBetaVersions",
  "value": "$betaVersions"
}
"@
  }

  if ("$($pkgWorkItem.fields["Custom.PackageGAVersion"])" -ne $gaVersions)
  {
    $fieldUpdates += @"
{
  "op": "replace",
  "path": "/fields/PackageGAVersion",
  "value": "$gaVersions"
}
"@
  }

  if ("$($pkgWorkItem.fields["Custom.PackagePatchVersions"])" -ne $patchVersions)
  {
    $fieldUpdates += @"
{
  "op": "replace",
  "path": "/fields/PackagePatchVersions",
  "value": "$patchVersions"
}
"@
  }
  return ,$fieldUpdates
}

function GetMDVersionValue($versionlist)
{
  $mdVersions = ""
  $mdFormat = "| {0} | {1} | {2} |`n"

  $htmlVersions = ""
  $htmlFormat = @"
<tr>
<td>{0}</td>
<td>{1}</td>
<td>{2}</td>
</tr>

"@

  foreach ($version in $versionList) {
    $mdVersions += ($mdFormat -f $version.Type, $version.Version, $version.Date)
    $htmlVersions += ($htmlFormat -f $version.Type, $version.Version, $version.Date)
  }

  $htmlTemplate = @"
<div style='display:none;width:0;height:0;overflow:hidden;position:absolute;font-size:0;' id=__md>| Type | Version | Date |
| - | - | - |
mdVersions
</div><style id=__mdStyle>
.rendered-markdown img {
cursor:pointer;
}

.rendered-markdown h1, .rendered-markdown h2, .rendered-markdown h3, .rendered-markdown h4, .rendered-markdown h5, .rendered-markdown h6 {
color:#007acc;
font-weight:400;
}

.rendered-markdown h1 {
border-bottom:1px solid #e6e6e6;
font-size:26px;
font-weight:600;
margin-bottom:20px;
}

.rendered-markdown h2 {
font-size:18px;
border-bottom:1px solid #e6e6e6;
font-weight:600;
color:#303030;
margin-bottom:10px;
margin-top:20px;
}

.rendered-markdown h3 {
font-size:16px;
font-weight:600;
margin-bottom:10px;
}

.rendered-markdown h4 {
font-size:14px;
margin-bottom:10px;
}

.rendered-markdown h5 {
font-size:12px;
margin-bottom:10px;
}

.rendered-markdown h6 {
font-size:12px;
font-weight:300;
margin-bottom:10px;
}

.rendered-markdown.metaitem {
font-size:12px;
padding-top:15px;
}

.rendered-markdown.metavalue {
font-size:12px;
padding-left:4px;
}

.rendered-markdown.metavalue>img {
height:32px;
width:32px;
margin-bottom:3px;
padding-left:1px;
}

.rendered-markdown li.metavaluelink {
list-style-type:disc;
list-style-position:inside;
}

.rendered-markdown li.metavalue>a {
border:none;
padding:0;
display:inline;
}

.rendered-markdown li.metavalue>a:hover {
background-color:inherit;
text-decoration:underline;
}

.rendered-markdown code, .rendered-markdown pre, .rendered-markdown samp {
font-family:Monaco,Menlo,Consolas,'Droid Sans Mono','Inconsolata','Courier New',monospace;
}

.rendered-markdown code {
color:#333;
background-color:#f8f8f8;
border:1px solid #ccc;
border-radius:3px;
padding:2px 4px;
font-size:90%;
line-height:2;
white-space:nowrap;
}

.rendered-markdown pre {
color:#333;
background-color:#f8f8f8;
border:1px solid #ccc;
display:block;
padding:6px;
font-size:13px;
word-break:break-all;
word-wrap:break-word;
}

.rendered-markdown pre code {
padding:0;
font-size:inherit;
color:inherit;
white-space:pre-wrap;
background-color:transparent;
line-height:1.428571429;
border:none;
}

.rendered-markdown.pre-scrollable {
max-height:340px;
overflow-y:scroll;
}

.rendered-markdown table {
border-collapse:collapse;
}

.rendered-markdown table {
width:auto;
}

.rendered-markdown table, .rendered-markdown th, .rendered-markdown td {
border:1px solid #ccc;
padding:4px;
}

.rendered-markdown th {
font-weight:bold;
background-color:#f8f8f8;
}
</style><div class=rendered-markdown><table>
<thead>
<tr>
<th>Type</th>
<th>Version</th>
<th>Date</th>
</tr>
</thead>
<tbody>htmlVersions</tbody>
</table>
</div>
"@ -replace "'", '\"'

  return $htmlTemplate.Replace("mdVersions", $mdVersions).Replace("htmlVersions", "`n$htmlVersions");
}

function UpdatePackageVersions($pkgWorkItem, $plannedVersions, $shippedVersions)
{
  # Create the planned and shipped versions, adding the new ones if any
  $updatePlanned = $false
  $plannedVersionSet = ParseVersionSetFromMDField $pkgWorkItem.fields["Custom.PlannedPackages"]
  foreach ($version in $plannedVersions)
  {
    if (!$plannedVersionSet.ContainsKey($version.Version))
    {
      $plannedVersionSet[$version.Version] = $version
      $updatePlanned = $true
    }
    else
    {
      # Lets check to see if someone wanted to update a date
      $existingVersion = $plannedVersionSet[$version.Version]
      if ($existingVersion.Date -ne $version.Date) {
        $existingVersion.Date = $version.Date
        $updatePlanned = $true
      }
    }
  }

  $updateShipped = $false
  $shippedVersionSet = ParseVersionSetFromMDField $pkgWorkItem.fields["Custom.ShippedPackages"]
  foreach ($version in $shippedVersions)
  {
    if (!$shippedVersionSet.ContainsKey($version.Version))
    {
      $shippedVersionSet[$version.Version] = $version
      $updateShipped = $true
    }
  }

  $versionSet = @{}
  foreach ($version in $shippedVersionSet.Keys)
  {
    if (!$versionSet.ContainsKey($version))
    {
      $versionSet[$version] = $shippedVersionSet[$version]
    }
  }

  foreach ($version in @($plannedVersionSet.Keys))
  {
    if (!$versionSet.ContainsKey($version))
    {
      $versionSet[$version] = $plannedVersionSet[$version]
    }
    else
    {
      # Looks like we shipped this version so remove it from the planned set
      $plannedVersionSet.Remove($version)
      $updatePlanned = $true
    }
  }

  $fieldUpdates = @()
  if ($updatePlanned)
  {
    $plannedPackages = GetMDVersionValue ($plannedVersionSet.Values | Sort-Object {$_.Date -as [DateTime]}, Version -Descending)
    $fieldUpdates += @"
{
  "op": "replace",
  "path": "/fields/Planned Packages",
  "value": "$plannedPackages"
}
"@
  }

  if ($updateShipped)
  {
    $newShippedVersions = $shippedVersionSet.Values | Sort-Object {$_.Date -as [DateTime]}, Version -Descending
    $shippedPackages = GetMDVersionValue $newShippedVersions
    $fieldUpdates += @"
{
  "op": "replace",
  "path": "/fields/Shipped Packages",
  "value": "$shippedPackages"
}
"@
  }

  # Full merged version set
  $versionList = $versionSet.Values | Sort-Object {$_.Date -as [DateTime]}, Version -Descending

  $versionFieldUpdates = GetTextVersionFields $versionList $pkgWorkItem
  if ($versionFieldUpdates.Count -gt 0)
  {
    $fieldUpdates += $versionFieldUpdates
  }

  # If no version files to update do nothing
  if ($fieldUpdates.Count -eq 0) {
    return $pkgWorkItem
  }

  $versionsForDebug = ($versionList | Foreach-Object { $_.Version }) -join ","
  $id = $pkgWorkItem.id
  $loggingString = "[$($pkgWorkItem.id)]"
  $loggingString += "$($pkgWorkItem.fields['Custom.Language'])"
  $loggingString += " - $($pkgWorkItem.fields['Custom.Package'])"
  $loggingString += "($($pkgWorkItem.fields['Custom.PackageVersionMajorMinor']))"
  $loggingString += " - Updating versions $versionsForDebug"
  Write-Host $loggingString

  $body = "[" + ($fieldUpdates -join ',') + "]"

  $headers = $null
  if (Get-Variable -Name "devops_pat" -ValueOnly -ErrorAction "Ignore")
  {
    $encodedToken = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes([string]::Format("{0}:{1}", "", $devops_pat)))
    $headers = @{ Authorization = "Basic $encodedToken" }
  }
  else
  {
    # Get a temp access token from the logged in az cli user for azure devops resource
    $jwt_accessToken = (az account get-access-token --resource "499b84ac-1321-427f-aa17-267ca6975798" --query "accessToken" --output tsv)
    $headers = @{ Authorization = "Bearer $jwt_accessToken" }
  }
  $response = Invoke-RestMethod -Method PATCH `
    -Uri "https://dev.azure.com/azure-sdk/_apis/wit/workitems/${id}?api-version=6.0" `
    -Headers $headers -Body $body -ContentType "application/json-patch+json" | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashTable
  return $response
}