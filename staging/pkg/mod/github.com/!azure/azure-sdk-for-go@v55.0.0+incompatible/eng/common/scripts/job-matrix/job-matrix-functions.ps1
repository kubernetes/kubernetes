Set-StrictMode -Version "4.0"

class MatrixConfig {
    [PSCustomObject]$displayNames
    [Hashtable]$displayNamesLookup
    [PSCustomObject]$matrix
    [MatrixParameter[]]$matrixParameters
    [Array]$include
    [Array]$exclude
}

class MatrixParameter {
    MatrixParameter([String]$name, [System.Object]$value) {
        $this.Value = $value
        $this.Name = $name
    }

    [System.Object]$Value
    [System.Object]$Name

    Set($value, [String]$keyRegex = '')
    {
        if ($this.Value -is [PSCustomObject]) {
            $set = $false
            foreach ($prop in $this.Value.PSObject.Properties) {
                if ($prop.Name -match $keyRegex) {
                    $prop.Value = $value
                    $set = $true
                    break
                }
            }
            if (!$set) {
                throw "Property `"$keyRegex`" does not exist for MatrixParameter."
            }
        } else {
            $this.Value = $value
        }
    }

    [System.Object]Flatten()
    {
        if ($this.Value -is [PSCustomObject]) {
            return $this.Value.PSObject.Properties | ForEach-Object {
                [MatrixParameter]::new($_.Name, $_.Value)
            }
        } elseif ($this.Value -is [Array]) {
            return $this.Value | ForEach-Object {
                [MatrixParameter]::new($this.Name, $_)
            }
        } else {
            return $this
        }
    }

    [Int]Length()
    {
        if ($this.Value -is [PSCustomObject]) {
            return ($this.Value.PSObject.Properties | Measure-Object).Count
        } elseif ($this.Value -is [Array]) {
            return $this.Value.Length
        } else {
            return 1
        }
    }

    [String]CreateDisplayName([Hashtable]$displayNamesLookup)
    {
        if ($null -eq $this.Value) {
            $displayName = ""
        } elseif ($this.Value -is [PSCustomObject]) {
            $displayName = $this.Name
        } else {
            $displayName = $this.Value.ToString()
        }

        if ($displayNamesLookup.ContainsKey($displayName)) {
            $displayName = $displayNamesLookup[$displayName]
        }

        # Matrix naming restrictions:
        # https://docs.microsoft.com/en-us/azure/devops/pipelines/process/phases?view=azure-devops&tabs=yaml#multi-job-configuration
        $displayName = $displayName -replace "[^A-Za-z0-9_]", ""
        return $displayName
    }
}

$IMPORT_KEYWORD = '$IMPORT'

function GenerateMatrix(
    [MatrixConfig]$config,
    [String]$selectFromMatrixType,
    [String]$displayNameFilter = ".*",
    [Array]$filters = @(),
    [Array]$replace = @(),
    [Array]$nonSparseParameters = @()
) {
    $matrixParameters, $importedMatrix, $combinedDisplayNameLookup = `
            ProcessImport $config.matrixParameters $selectFromMatrixType $nonSparseParameters $config.displayNamesLookup
    if ($selectFromMatrixType -eq "sparse") {
        $matrix = GenerateSparseMatrix $matrixParameters $config.displayNamesLookup $nonSparseParameters
    } elseif ($selectFromMatrixType -eq "all") {
        $matrix = GenerateFullMatrix $matrixParameters $config.displayNamesLookup
    } else {
        throw "Matrix generator not implemented for selectFromMatrixType: $($platform.selectFromMatrixType)"
    }

    # Combine with imported after matrix generation, since a sparse selection should result in a full combination of the
    # top level and imported sparse matrices (as opposed to a sparse selection of both matrices).
    if ($importedMatrix) {
        $matrix = CombineMatrices $matrix $importedMatrix $combinedDisplayNameLookup
    }
    if ($config.exclude) {
        $matrix = ProcessExcludes $matrix $config.exclude
    }
    if ($config.include) {
        $matrix = ProcessIncludes $config $matrix $selectFromMatrixType
    }

    $matrix = FilterMatrix $matrix $filters
    $matrix = ProcessReplace $matrix $replace $config.displayNamesLookup
    $matrix = FilterMatrixDisplayName $matrix $displayNameFilter
    return $matrix
}

function ProcessNonSparseParameters(
    [MatrixParameter[]]$parameters,
    [Array]$nonSparseParameters
) {
    if (!$nonSparseParameters) {
        return $parameters, $null
    }

    $sparse = [MatrixParameter[]]@()
    $nonSparse = [MatrixParameter[]]@()

    foreach ($param in $parameters) {
        if ($param.Name -in $nonSparseParameters) {
            $nonSparse += $param
        } else {
            $sparse += $param
        }
    }

    return $sparse, $nonSparse
}

function FilterMatrixDisplayName([array]$matrix, [string]$filter) {
    return $matrix | Where-Object { $_ } | ForEach-Object {
        if ($_.Name -match $filter) {
            return $_
        }
    }
}

# Filters take the format of key=valueregex,key2=valueregex2
function FilterMatrix([array]$matrix, [array]$filters) {
    $matrix = $matrix | ForEach-Object {
        if (MatchesFilters $_ $filters) {
            return $_
        }
    }
    return $matrix
}

function MatchesFilters([hashtable]$entry, [array]$filters) {
    foreach ($filter in $filters) {
        $key, $regex = ParseFilter $filter
        # Default all regex checks to go against empty string when keys are missing.
        # This simplifies the filter syntax/interface to be regex only.
        $value = ""
        if ($null -ne $entry -and $entry.parameters.Contains($key)) {
            $value = $entry.parameters[$key]
        }
        if ($value -notmatch $regex) {
            return $false
        }
    }

    return $true
}

function ParseFilter([string]$filter) {
    # Lazy match key in case value contains '='
    if ($filter -match "(.+?)=(.+)") {
        $key = $matches[1]
        $regex = $matches[2]
        return $key, $regex
    } else {
        throw "Invalid filter: `"${filter}`", expected <key>=<regex> format"
    }
}

# Importing the JSON as PSCustomObject preserves key ordering,
# whereas ConvertFrom-Json -AsHashtable does not
function GetMatrixConfigFromJson([String]$jsonConfig)
{
    [MatrixConfig]$config = $jsonConfig | ConvertFrom-Json
    $config.matrixParameters = @()
    $config.displayNamesLookup = @{}
    $include = [MatrixParameter[]]@()
    $exclude = [MatrixParameter[]]@()

    if ($null -ne $config.displayNames) {
        $config.displayNames.PSObject.Properties | ForEach-Object {
            $config.displayNamesLookup.Add($_.Name, $_.Value)
        }
    }
    if ($null -ne $config.matrix) {
        $config.matrixParameters = PsObjectToMatrixParameterArray $config.matrix
    }
    foreach ($includeMatrix in $config.include) {
        $include += ,@(PsObjectToMatrixParameterArray $includeMatrix)
    }
    foreach ($excludeMatrix in $config.exclude) {
        $exclude += ,@(PsObjectToMatrixParameterArray $excludeMatrix)
    }

    $config.include = $include
    $config.exclude = $exclude

    return $config
}

function PsObjectToMatrixParameterArray([PSCustomObject]$obj)
{
    if ($obj -eq $null) {
        return $null
    }
    return $obj.PSObject.Properties | ForEach-Object {
        [MatrixParameter]::new($_.Name, $_.Value)
    }
}

function ProcessExcludes([Array]$matrix, [Array]$excludes)
{
    $deleteKey = "%DELETE%"
    $exclusionMatrix = @()

    foreach ($exclusion in $excludes) {
        $full = GenerateFullMatrix $exclusion
        $exclusionMatrix += $full
    }

    foreach ($element in $matrix) {
        foreach ($exclusion in $exclusionMatrix) {
            $match = MatrixElementMatch $element.parameters $exclusion.parameters
            if ($match) {
                $element.parameters[$deleteKey] = $true
            }
        }
    }

    return $matrix | Where-Object { !$_.parameters.Contains($deleteKey) }
}

function ProcessIncludes([MatrixConfig]$config, [Array]$matrix)
{
    $inclusionMatrix = @()
    foreach ($inclusion in $config.include) {
        $full = GenerateFullMatrix $inclusion $config.displayNamesLookup
        $inclusionMatrix += $full
    }

    return $matrix + $inclusionMatrix
}

function ParseReplacement([String]$replacement) {
    $parsed = '', '', ''
    $idx = 0
    $escaped = $false
    $operators = '=', '/'
    $err = "Invalid replacement syntax, expecting <key>=<value>/<replace>"

    foreach ($c in $replacement -split '') {
        if ($idx -ge $parsed.Length) {
            throw $err
        }
        if (!$escaped -and $c -in $operators) {
            $idx++
        } else {
            $parsed[$idx] += $c
        }
        $escaped = $c -eq '\'
    }

    if ($idx -lt $parsed.Length - 1) {
        throw $err
    }

    $replace = $parsed[2] -replace "\\([$($operators -join '')])", '$1'

    return @{
        "key" = '^' + $parsed[0] + '$'
        # Force full matches only.
        "value" = '^' + $parsed[1] + '$'
        "replace" = $replace
    }
}

function ProcessReplace
{
    param(
        [Array]$matrix,
        [Array]$replacements,
        [Hashtable]$displayNamesLookup
    )

    if (!$replacements) {
        return $matrix
    }

    $replaceMatrix = @()

    foreach ($element in $matrix) {
        $replacement = [MatrixParameter[]]@()

        foreach ($perm in $element._permutation) {
            $replace = $perm

            # Iterate nested permutations or run once for singular values (int, string, bool)
            foreach ($flattened in $perm.Flatten()) {
                foreach ($query in $replacements) {
                    $parsed = ParseReplacement $query
                    if ($flattened.Name -match $parsed.key -and $flattened.Value -match $parsed.value) {
                        # In most cases, this will just swap one value for another, however -replace
                        # is used here in order to support replace values which may use regex capture groups
                        # e.g. 'foo-1' -replace '(foo)-1', '$1-replaced'
                        $replaceValue = $flattened.Value -replace $parsed.value, $parsed.replace
                        $perm.Set($replaceValue, $parsed.key)
                        break
                    }
                }
            }

            $replacement += $perm
        }

        $replaceMatrix += CreateMatrixCombinationScalar $replacement $displayNamesLookup
    }

    return $replaceMatrix
}

function ProcessImport([MatrixParameter[]]$matrix, [String]$selection, [Array]$nonSparseParameters, [Hashtable]$displayNamesLookup)
{
    $importPath = ""
    $matrix = $matrix | ForEach-Object {
        if ($_.Name -ne $IMPORT_KEYWORD) {
            return $_
        } else {
            $importPath = $_.Value
        }
    }
    if ((!$matrix -and !$importPath) -or !$importPath) {
        return $matrix, @()
    }

    $importedMatrixConfig = GetMatrixConfigFromJson (Get-Content $importPath)
    $importedMatrix = GenerateMatrix `
                        -config $importedMatrixConfig `
                        -selectFromMatrixType $selection `
                        -nonSparseParameters $nonSparseParameters

    $combinedDisplayNameLookup = $importedMatrixConfig.displayNamesLookup
    foreach ($lookup in $displayNamesLookup.GetEnumerator()) {
        $combinedDisplayNameLookup[$lookup.Name] = $lookup.Value
    }

    return $matrix, $importedMatrix, $importedMatrixConfig.displayNamesLookup
}

function CombineMatrices([Array]$matrix1, [Array]$matrix2, [Hashtable]$displayNamesLookup = @{})
{
    $combined = @()
    if (!$matrix1) {
        return $matrix2
    }
    if (!$matrix2) {
        return $matrix1
    }

    foreach ($entry1 in $matrix1) {
        foreach ($entry2 in $matrix2) {
            $combined += CreateMatrixCombinationScalar ($entry1._permutation + $entry2._permutation) $displayNamesLookup
        }
    }

    return $combined
}

function MatrixElementMatch([System.Collections.Specialized.OrderedDictionary]$source, [System.Collections.Specialized.OrderedDictionary]$target)
{
    if ($target.Count -eq 0) {
        return $false
    }

    foreach ($key in $target.Keys) {
        if (!$source.Contains($key) -or $source[$key] -ne $target[$key]) {
            return $false
        }
    }

    return $true
}

function CloneOrderedDictionary([System.Collections.Specialized.OrderedDictionary]$dictionary) {
    $newDictionary = [Ordered]@{}
    foreach ($element in $dictionary.GetEnumerator()) {
        $newDictionary[$element.Name] = $element.Value
    }
    return $newDictionary
}

function SerializePipelineMatrix([Array]$matrix)
{
    $pipelineMatrix = [Ordered]@{}
    foreach ($entry in $matrix) {
        if ($pipelineMatrix.Contains($entry.Name)) {
            Write-Warning "Found duplicate configurations for job `"$($entry.name)`". Multiple values may have been replaced with the same value."
            continue
        }
        $pipelineMatrix.Add($entry.name, [Ordered]@{})
        foreach ($key in $entry.parameters.Keys) {
            $pipelineMatrix[$entry.name].Add($key, $entry.parameters[$key])
        }
    }

    return @{
        compressed = $pipelineMatrix | ConvertTo-Json -Compress ;
        pretty = $pipelineMatrix | ConvertTo-Json;
    }
}

function GenerateSparseMatrix(
    [MatrixParameter[]]$parameters,
    [Hashtable]$displayNamesLookup,
    [Array]$nonSparseParameters = @()
) {
    $parameters, $nonSparse = ProcessNonSparseParameters $parameters $nonSparseParameters
    $dimensions = GetMatrixDimensions $parameters
    $matrix = GenerateFullMatrix $parameters $displayNamesLookup

    $sparseMatrix = @()
    [array]$indexes = GetSparseMatrixIndexes $dimensions
    foreach ($idx in $indexes) {
        $sparseMatrix += GetNdMatrixElement $idx $matrix $dimensions
    }

    if ($nonSparse) {
        $allOfMatrix = GenerateFullMatrix $nonSparse $displayNamesLookup
        return CombineMatrices $allOfMatrix $sparseMatrix $displayNamesLookup
    }

    return $sparseMatrix
}

function GetSparseMatrixIndexes([Array]$dimensions)
{
    $size = ($dimensions | Measure-Object -Maximum).Maximum
    $indexes = @()

    # With full matrix, retrieve items by doing diagonal lookups across the matrix N times.
    # For example, given a matrix with dimensions 3, 2, 2:
    # 0, 0, 0
    # 1, 1, 1
    # 2, 2, 2
    # 3, 0, 0 <- 3, 3, 3 wraps to 3, 0, 0 given the dimensions
    for ($i = 0; $i -lt $size; $i++) {
        $idx = @()
        for ($j = 0; $j -lt $dimensions.Length; $j++) {
            $idx += $i % $dimensions[$j]
        }
        $indexes += ,$idx
    }

    return ,$indexes
}

function GenerateFullMatrix(
    [MatrixParameter[]] $parameters,
    [Hashtable]$displayNamesLookup = @{}
) {
    # Handle when the config does not have a matrix specified (e.g. only the include field is specified)
    if (!$parameters) {
        return @()
    }

    $matrix = [System.Collections.ArrayList]::new()
    InitializeMatrix $parameters $displayNamesLookup $matrix

    return $matrix
}

function CreateMatrixCombinationScalar([MatrixParameter[]]$permutation, [Hashtable]$displayNamesLookup = @{})
{
    $names = @()
    $flattenedParameters = [Ordered]@{}

    foreach ($entry in $permutation) {
        $nameSegment = ""

        # Unwind nested permutations or run once for singular values (int, string, bool)
        foreach ($param in $entry.Flatten()) {
            if ($flattenedParameters.Contains($param.Name)) {
                throw "Found duplicate parameter `"$($param.Name)`" when creating matrix combination."
            }
            $flattenedParameters.Add($param.Name, $param.Value)
        }

        $nameSegment = $entry.CreateDisplayName($displayNamesLookup)
        if ($nameSegment) {
            $names += $nameSegment
        }
    }

    # The maximum allowed matrix name length is 100 characters
    $name = $names -join "_"
    if ($name.Length -gt 100) {
        $name = $name[0..99] -join ""
    }
    $stripped = $name -replace "^[^A-Za-z]*", ""  # strip leading digits
    if ($stripped -eq "") {
        $name = "job_" + $name  # Handle names that consist entirely of numbers
    } else {
        $name = $stripped
    }

    return @{
        name = $name
        parameters = $flattenedParameters
        # Keep the original permutation around in case we need to re-process this entry when transforming the matrix
        _permutation = $permutation
    }
}

function InitializeMatrix
{
    param(
        [MatrixParameter[]]$parameters,
        [Hashtable]$displayNamesLookup,
        [System.Collections.ArrayList]$permutations,
        $permutation = [MatrixParameter[]]@()
    )
    $head, $tail = $parameters

    if (!$head) {
        $entry = CreateMatrixCombinationScalar $permutation $displayNamesLookup
        $permutations.Add($entry) | Out-Null
        return
    }

    # This behavior implicitly treats non-array values as single elements
    foreach ($param in $head.Flatten()) {
        $newPermutation = $permutation + $param
        InitializeMatrix $tail $displayNamesLookup $permutations $newPermutation
    }
}

function GetMatrixDimensions([MatrixParameter[]]$parameters)
{
    $dimensions = @()
    foreach ($param in $parameters) {
        $dimensions += $param.Length()
    }

    return $dimensions
}

function SetNdMatrixElement
{
    param(
        $element,
        [ValidateNotNullOrEmpty()]
        [Array]$idx,
        [ValidateNotNullOrEmpty()]
        [Array]$matrix,
        [ValidateNotNullOrEmpty()]
        [Array]$dimensions
    )

    if ($idx.Length -ne $dimensions.Length) {
        throw "Matrix index query $($idx.Length) must be the same length as its dimensions $($dimensions.Length)"
    }

    $arrayIndex = GetNdMatrixArrayIndex $idx $dimensions
    $matrix[$arrayIndex] = $element
}

function GetNdMatrixArrayIndex
{
    param(
        [ValidateNotNullOrEmpty()]
        [Array]$idx,
        [ValidateNotNullOrEmpty()]
        [Array]$dimensions
    )

    if ($idx.Length -ne $dimensions.Length) {
        throw "Matrix index query length ($($idx.Length)) must be the same as dimension length ($($dimensions.Length))"
    }

    $stride = 1
    # Commented out does lookup with wrap handling
    # $index = $idx[$idx.Length-1] % $dimensions[$idx.Length-1]
    $index = $idx[$idx.Length-1]

    for ($i = $dimensions.Length-1; $i -ge 1; $i--) {
        $stride *= $dimensions[$i]
        # Commented out does lookup with wrap handling
        # $index += ($idx[$i-1] % $dimensions[$i-1]) * $stride
        $index += $idx[$i-1] * $stride
    }

    return $index
}

function GetNdMatrixElement
{
    param(
        [ValidateNotNullOrEmpty()]
        [Array]$idx,
        [ValidateNotNullOrEmpty()]
        [Array]$matrix,
        [ValidateNotNullOrEmpty()]
        [Array]$dimensions
    )

    $arrayIndex = GetNdMatrixArrayIndex $idx $dimensions
    return $matrix[$arrayIndex]
}

function GetNdMatrixIndex
{
    param(
        [int]$index,
        [ValidateNotNullOrEmpty()]
        [Array]$dimensions
    )

    $matrixIndex = @()
    $stride = 1

    for ($i = $dimensions.Length-1; $i -ge 1; $i--) {
        $stride *= $dimensions[$i]
        $page = [math]::floor($index / $stride) % $dimensions[$i-1]
        $matrixIndex = ,$page + $matrixIndex
    }
    $col = $index % $dimensions[$dimensions.Length-1]
    $matrixIndex += $col

    return $matrixIndex
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The below functions are non-dynamic examples that   #
# help explain the above N-dimensional algorithm      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
function Get4dMatrixElement([Array]$idx, [Array]$matrix, [Array]$dimensions)
{
    $stride1 = $idx[0] * $dimensions[1] * $dimensions[2] * $dimensions[3]
    $stride2 = $idx[1] * $dimensions[2] * $dimensions[3]
    $stride3 = $idx[2] * $dimensions[3]
    $stride4 = $idx[3]

    return $matrix[$stride1 + $stride2 + $stride3 + $stride4]
}

function Get4dMatrixIndex([int]$index, [Array]$dimensions)
{
    $stride1 = $dimensions[3]
    $stride2 = $dimensions[2]
    $stride3 = $dimensions[1]
    $page1 = [math]::floor($index / $stride1) % $dimensions[2]
    $page2 = [math]::floor($index / ($stride1 * $stride2)) % $dimensions[1]
    $page3 = [math]::floor($index / ($stride1 * $stride2 * $stride3)) % $dimensions[0]
    $remainder = $index % $dimensions[3]

    return @($page3, $page2, $page1, $remainder)
}

