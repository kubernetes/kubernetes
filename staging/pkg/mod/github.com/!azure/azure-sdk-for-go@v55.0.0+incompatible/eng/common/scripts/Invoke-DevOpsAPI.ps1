. "${PSScriptRoot}\logging.ps1"

$DevOpsAPIBaseURI = "https://dev.azure.com/{0}/{1}/_apis/{2}/{3}?{4}api-version=6.0"

function Get-DevOpsApiHeaders ($Base64EncodedToken) {
  $headers = @{
    Authorization = "Basic $Base64EncodedToken"
  }
  return $headers
}

function Start-DevOpsBuild {
  param (
    $Organization="azure-sdk",
    $Project="internal",
    [Parameter(Mandatory = $true)]
    $SourceBranch,
    [Parameter(Mandatory = $true)]
    $DefinitionId,
    [ValidateNotNullOrEmpty()]
    [Parameter(Mandatory = $true)]
    $Base64EncodedAuthToken
  )

  $uri = "$DevOpsAPIBaseURI" -F $Organization, $Project , "build" , "builds", ""

  $parameters = @{
    sourceBranch = $SourceBranch
    definition = @{ id = $DefinitionId }
  }

  return Invoke-RestMethod `
          -Method POST `
          -Body ($parameters | ConvertTo-Json) `
          -Uri $uri `
          -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
          -MaximumRetryCount 3 `
          -ContentType "application/json"
}

function Update-DevOpsBuild {
  param (
    $Organization="azure-sdk",
    $Project="internal",
    [ValidateNotNullOrEmpty()]
    [Parameter(Mandatory = $true)]
    $BuildId,
    $Status, # pass canceling to cancel build
    [ValidateNotNullOrEmpty()]
    [Parameter(Mandatory = $true)]
    $Base64EncodedAuthToken
  )

  $uri = "$DevOpsAPIBaseURI" -F $Organization, $Project, "build", "builds/$BuildId", ""
  $parameters = @{}

  if ($Status) { $parameters["status"] = $Status}

  return Invoke-RestMethod `
          -Method PATCH `
          -Body ($parameters | ConvertTo-Json) `
          -Uri $uri `
          -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
          -MaximumRetryCount 3 `
          -ContentType "application/json"
}

function Get-DevOpsBuilds {
  param (
    $Organization="azure-sdk",
    $Project="internal",
    $BranchName, #Should start with 'refs/heads/'
    $Definitions, # Comma seperated string of definition IDs
    $StatusFilter, # Comma seperated string 'cancelling, completed, inProgress, notStarted'
    [ValidateNotNullOrEmpty()]
    [Parameter(Mandatory = $true)]
    $Base64EncodedAuthToken
  )

  $query = ""

  if ($BranchName) { $query += "branchName=$BranchName&" }
  if ($Definitions) { $query += "definitions=$Definitions&" }
  if ($StatusFilter) { $query += "statusFilter=$StatusFilter&" }
  $uri = "$DevOpsAPIBaseURI" -F $Organization, $Project , "build" , "builds", $query

  return Invoke-RestMethod `
          -Method GET `
          -Uri $uri `
          -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
          -MaximumRetryCount 3
}

function Delete-RetentionLease {
  param (
    $Organization,
    $Project,
    $LeaseId,
    $Base64EncodedAuthToken
  )

  $uri = "https://dev.azure.com/$Organization/$Project/_apis/build/retention/leases?ids=$LeaseId&api-version=6.0-preview.1"

  return Invoke-RestMethod `
    -Method DELETE `
    -Uri $uri `
    -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
    -MaximumRetryCount 3
}

function Get-RetentionLeases {
  param (
    $Organization,
    $Project,
    $DefinitionId,
    $RunId,
    $OwnerId,
    $Base64EncodedAuthToken
  )

  $uri = "https://dev.azure.com/$Organization/$Project/_apis/build/retention/leases?ownerId=$OwnerId&definitionId=$DefinitionId&runId=$RunId&api-version=6.0-preview.1"

  return Invoke-RestMethod `
    -Method GET `
    -Uri $uri `
    -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
    -MaximumRetryCount 3
}

function Add-RetentionLease {
  param (
    $Organization,
    $Project,
    $DefinitionId,
    $RunId,
    $OwnerId,
    $DaysValid,
    $Base64AuthToken
  )

  $parameter = @{}
  $parameter["definitionId"] = $DefinitionId
  $parameter["runId"] = $RunId
  $parameter["ownerId"] = $OwnerId
  $parameter["daysValid"] = $DaysValid


  $body = $parameter | ConvertTo-Json

  $uri = "https://dev.azure.com/$Organization/$Project/_apis/build/retention/leases?api-version=6.0-preview.1"

  return Invoke-RestMethod `
          -Method POST `
          -Body "[$body]" `
          -Uri $uri `
          -Headers (Get-DevOpsApiHeaders -Base64EncodedToken $Base64EncodedAuthToken) `
          -MaximumRetryCount 3 `
          -ContentType "application/json"

}