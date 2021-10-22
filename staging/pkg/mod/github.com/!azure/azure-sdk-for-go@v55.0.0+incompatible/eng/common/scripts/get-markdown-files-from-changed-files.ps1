param (
  # The root repo we scaned with.
  [string] $RootRepo = '$PSScriptRoot/../../..',
  # The target branch to compare with.
  [string] $targetBranch = ("origin/${env:SYSTEM_PULLREQUEST_TARGETBRANCH}" -replace "/refs/heads/")
)
$deletedFiles = (git diff $targetBranch HEAD --name-only --diff-filter=D)
$renamedFiles = (git diff $targetBranch HEAD --diff-filter=R)
$changedMarkdowns = (git diff $targetBranch HEAD --name-only -- '*.md')

$beforeRenameFiles = @()
# Retrieve the 'renamed from' files. Git command only returns back the files after rename. 
# In order to have the files path before rename, it has to do some regex checking. 
# It is better to be replaced by more reliable commands if any.
foreach ($file in $renamedFiles) {
  if ($file -match "^rename from (.*)$") {
    $beforeRenameFiles += $file -replace "^rename from (.*)$", '$1'
  }
}
# A combined list of deleted and renamed files.
$relativePathLinks = ($deletedFiles + $beforeRenameFiles)
# Removed the deleted markdowns. 
$changedMarkdowns = $changedMarkdowns | Where-Object { $deletedFiles -notcontains $_ }
# Scan all markdowns and find if it contains the deleted or renamed files.
$markdownContainLinks = @()
$allMarkdownFiles = Get-ChildItem -Path $RootRepo -Recurse -Include *.md
foreach ($f in $allMarkdownFiles) {
  $filePath = $f.FullName
  $content = Get-Content -Path $filePath -Raw
  foreach ($l in $relativePathLinks) {
    if ($content -match $l) {
      $markdownContainLinks += $filePath
      break
    }
  }
}

# Convert markdowns path of the PR to absolute path.
$adjustedReadmes = $changedMarkdowns | Foreach-Object { Resolve-Path $_ }
$markdownContainLinks += $adjustedReadmes

# Get rid of any duplicated ones.
$allMarkdowns = [string[]]($markdownContainLinks | Sort-Object | Get-Unique)

Write-Host "Here are all markdown files we need to check based on the changed files:"
foreach ($file in $allMarkdowns) {
  Write-Host "    $file"
}
return $allMarkdowns
