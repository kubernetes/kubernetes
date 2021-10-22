param (
  $TargetDirectory, # should be in relative form from root of repo. EG: sdk/servicebus
  $RootDirectory, # ideally $(Build.SourcesDirectory)
  $VsoVariable = "" # target devops output variable
)
$target = $TargetDirectory.ToLower().Trim("/")
$codeOwnersLocation = Join-Path $RootDirectory -ChildPath ".github/CODEOWNERS"
$ownedFolders = @{}

if (!(Test-Path $codeOwnersLocation)) {
  Write-Host "Unable to find CODEOWNERS file in target directory $RootDirectory"
  exit 1
}

$codeOwnersContent = Get-Content $codeOwnersLocation

foreach ($contentLine in $codeOwnersContent) {
  if (-not $contentLine.StartsWith("#") -and $contentLine){
    $splitLine = $contentLine -split "\s+"
    
    # CODEOWNERS file can also have labels present after the owner aliases
    # gh aliases start with @ in codeowners. don't pass on to API calls
    $ownedFolders[$splitLine[0].ToLower().Trim("/")] = ($splitLine[1..$($splitLine.Length)] `
      | ? { $_.StartsWith("@") } `
      | % { return $_.substring(1) }) -join ","
  }
}

$results = $ownedFolders[$target]

if ($results) {
  Write-Host "Found a folder $results to match $target"
  
  if ($VsoVariable) {
    $alreadyPresent = [System.Environment]::GetEnvironmentVariable($VsoVariable)

    if ($alreadyPresent) { 
      $results += ",$alreadyPresent"
    }
    Write-Host "##vso[task.setvariable variable=$VsoVariable;]$results"
  }

  return $results
}
else {
  Write-Host "Unable to match path $target in CODEOWNERS file located at $codeOwnersLocation."
  Write-Host ($ownedFolders | ConvertTo-Json)
  return ""
}

