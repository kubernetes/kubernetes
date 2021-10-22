Import-Module Pester

BeforeAll {
    . $PSScriptRoot/../job-matrix-functions.ps1

    $matrixConfig = @"
{
    "displayNames": {
        "--enableFoo": "withfoo"
    },
    "matrix": {
        "operatingSystem": [
          "windows-2019",
          "ubuntu-18.04",
          "macOS-10.15"
        ],
        "framework": [
          "net461",
          "netcoreapp2.1"
        ],
        "additionalArguments": [
            "",
            "--enableFoo"
        ]
    },
    "include": [
        {
            "operatingSystem": "windows-2019",
            "framework": ["net461", "netcoreapp2.1", "net50"],
            "additionalArguments": "--enableWindowsFoo"
        }
    ],
    "exclude": [
        {
            "operatingSystem": "windows-2019",
            "framework": "net461"
        },
        {
            "operatingSystem": "macOS-10.15",
            "framework": "netcoreapp2.1"
        },
        {
            "operatingSystem": ["macOS-10.15", "ubuntu-18.04"],
            "additionalArguments": "--enableFoo"
        }
    ]
}
"@
}

Describe "Matrix-Lookup" -Tag "lookup" {
    It "Should navigate a 2d matrix: <row> <col>" -TestCases @(
         @{ row = 0; col = 0; expected = 1 },
         @{ row = 0; col = 1; expected = 2 },
         @{ row = 1; col = 0; expected = 3 },
         @{ row = 1; col = 1; expected = 4 }
    ) {
        $dimensions = @(2, 2)
        $matrix = @(
           1, 2, 3, 4
        )
        GetNdMatrixElement @($row, $col) $matrix $dimensions | Should -Be $expected
    }

    It "Should navigate a 3d matrix: <z> <row> <col>" -TestCases @(
         @{ z = 0; row = 0; col = 0; expected = 1 }
         @{ z = 0; row = 0; col = 1; expected = 2 }
         @{ z = 0; row = 1; col = 0; expected = 3 }
         @{ z = 0; row = 1; col = 1; expected = 4 }
         @{ z = 1; row = 0; col = 0; expected = 5 }
         @{ z = 1; row = 0; col = 1; expected = 6 }
         @{ z = 1; row = 1; col = 0; expected = 7 }
         @{ z = 1; row = 1; col = 1; expected = 8 }
    ) {
        $dimensions = @(2, 2, 2)
        $matrix = @(
           1, 2, 3, 4, 5, 6, 7, 8
        )
        GetNdMatrixElement @($z, $row, $col) $matrix $dimensions | Should -Be $expected
    }

    It "Should navigate a 4d matrix: <t> <z> <row> <col>" -TestCases @(
         @{ t = 0; z = 0; row = 0; col = 0; expected = 1 }
         @{ t = 0; z = 0; row = 0; col = 1; expected = 2 }
         @{ t = 0; z = 0; row = 1; col = 0; expected = 3 }
         @{ t = 0; z = 0; row = 1; col = 1; expected = 4 }
         @{ t = 0; z = 1; row = 0; col = 0; expected = 5 }
         @{ t = 0; z = 1; row = 0; col = 1; expected = 6 }
         @{ t = 0; z = 1; row = 1; col = 0; expected = 7 }
         @{ t = 0; z = 1; row = 1; col = 1; expected = 8 }
         @{ t = 1; z = 0; row = 0; col = 0; expected = 9 }
         @{ t = 1; z = 0; row = 0; col = 1; expected = 10 }
         @{ t = 1; z = 0; row = 1; col = 0; expected = 11 }
         @{ t = 1; z = 0; row = 1; col = 1; expected = 12 }
         @{ t = 1; z = 1; row = 0; col = 0; expected = 13 }
         @{ t = 1; z = 1; row = 0; col = 1; expected = 14 }
         @{ t = 1; z = 1; row = 1; col = 0; expected = 15 }
         @{ t = 1; z = 1; row = 1; col = 1; expected = 16 }
    ) {
        $dimensions = @(2, 2, 2, 2)
        $matrix = @(
           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        )
        GetNdMatrixElement @($t, $z, $row, $col) $matrix $dimensions | Should -Be $expected
    }

    It "Should navigate a 4d matrix: <t> <z> <row> <col>" -TestCases @(
         @{ t = 0; z = 0; row = 0; col = 0; expected = 1 }
         @{ t = 0; z = 0; row = 0; col = 1; expected = 2 }
         @{ t = 0; z = 0; row = 0; col = 2; expected = 3 }
         @{ t = 0; z = 0; row = 0; col = 3; expected = 4 }

         @{ t = 0; z = 0; row = 1; col = 0; expected = 5 }
         @{ t = 0; z = 0; row = 1; col = 1; expected = 6 }
         @{ t = 0; z = 0; row = 1; col = 2; expected = 7 }
         @{ t = 0; z = 0; row = 1; col = 3; expected = 8 }

         @{ t = 0; z = 1; row = 0; col = 0; expected = 9 }
         @{ t = 0; z = 1; row = 0; col = 1; expected = 10 }
         @{ t = 0; z = 1; row = 0; col = 2; expected = 11 }
         @{ t = 0; z = 1; row = 0; col = 3; expected = 12 }

         @{ t = 0; z = 1; row = 1; col = 0; expected = 13 }
         @{ t = 0; z = 1; row = 1; col = 1; expected = 14 }
         @{ t = 0; z = 1; row = 1; col = 2; expected = 15 }
         @{ t = 0; z = 1; row = 1; col = 3; expected = 16 }
    ) {
        $dimensions = @(1, 2, 2, 4)
        $matrix = @(
           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        )
        GetNdMatrixElement @($t, $z, $row, $col) $matrix $dimensions | Should -Be $expected
    }

    # Skipping since by default wrapping behavior on indexing is disabled.
    # Keeping here in case we want to enable it.
    It -Skip "Should handle index wrapping: <row> <col>" -TestCases @(
         @{ row = 2; col = 2; expected = 1 }
         @{ row = 2; col = 3; expected = 2 }
         @{ row = 4; col = 4; expected = 1 }
         @{ row = 4; col = 5; expected = 2 }
    ) {
        $dimensions = @(2, 2)
        $matrix = @(
           1, 2, 3, 4
        )
        GetNdMatrixElement @($row, $col) $matrix $dimensions | Should -Be $expected
    }
}

Describe "Matrix-Reverse-Lookup" -Tag "lookup" {
    It "Should lookup a 2d matrix index: <index>" -TestCases @(
         @{ index = 0; expected = @(0,0) }
         @{ index = 1; expected = @(0,1) }
         @{ index = 2; expected = @(1,0) }
         @{ index = 3; expected = @(1,1) }
    ) {
        $dimensions = @(2, 2)
        $matrix = @(1, 2, 3, 4)
        GetNdMatrixElement $expected $matrix $dimensions | Should -Be $matrix[$index]
        GetNdMatrixIndex $index $dimensions | Should -Be $expected
    }

    It "Should lookup a 3d matrix index: <index>" -TestCases @(
         @{ index = 0; expected = @(0,0,0) }
         @{ index = 1; expected = @(0,0,1) }
         @{ index = 2; expected = @(0,1,0) }
         @{ index = 3; expected = @(0,1,1) }

         @{ index = 4; expected = @(1,0,0) }
         @{ index = 5; expected = @(1,0,1) }
         @{ index = 6; expected = @(1,1,0) }
         @{ index = 7; expected = @(1,1,1) }
    ) {
        $dimensions = @(2, 2, 2)
        $matrix = @(0, 1, 2, 3, 4, 5, 6, 7)
        GetNdMatrixElement $expected $matrix $dimensions | Should -Be $matrix[$index]
        GetNdMatrixIndex $index $dimensions | Should -Be $expected
    }

    It "Should lookup a 3d matrix index: <index>" -TestCases @(
         @{ index = 0; expected = @(0,0,0) }
         @{ index = 1; expected = @(0,0,1) }
         @{ index = 2; expected = @(0,0,2) }

         @{ index = 3; expected = @(0,1,0) }
         @{ index = 4; expected = @(0,1,1) }
         @{ index = 5; expected = @(0,1,2) }

         @{ index = 6; expected = @(1,0,0) }
         @{ index = 7; expected = @(1,0,1) }
         @{ index = 8; expected = @(1,0,2) }

         @{ index = 9; expected = @(1,1,0) }
         @{ index = 10; expected = @(1,1,1) }
         @{ index = 11; expected = @(1,1,2) }
    ) {
        $dimensions = @(2, 2, 3)
        $matrix = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        GetNdMatrixElement $expected $matrix $dimensions | Should -Be $matrix[$index]
        GetNdMatrixIndex $index $dimensions | Should -Be $expected
    }

    It "Should lookup a 3d matrix index: <index>" -TestCases @(
         @{ index = 0; expected = @(0,0,0) }
         @{ index = 1; expected = @(0,0,1) }
         @{ index = 2; expected = @(0,1,0) }
         @{ index = 3; expected = @(0,1,1) }

         @{ index = 4; expected = @(1,0,0) }
         @{ index = 5; expected = @(1,0,1) }
         @{ index = 6; expected = @(1,1,0) }
         @{ index = 7; expected = @(1,1,1) }

         @{ index = 8; expected = @(2,0,0) }
         @{ index = 9; expected = @(2,0,1) }
         @{ index = 10; expected = @(2,1,0) }
         @{ index = 11; expected = @(2,1,1) }
    ) {
        $dimensions = @(3, 2, 2)
        $matrix = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        GetNdMatrixElement $expected $matrix $dimensions | Should -Be $matrix[$index]
        GetNdMatrixIndex $index $dimensions | Should -Be $expected
    }

    It "Should lookup a 4d matrix index: <index>" -TestCases @(
         @{ index = 0; expected = @(0,0,0,0) }
         @{ index = 1; expected = @(0,0,0,1) }
         @{ index = 2; expected = @(0,0,0,2) }
         @{ index = 3; expected = @(0,0,0,3) }
                      
         @{ index = 4; expected = @(0,0,1,0) }
         @{ index = 5; expected = @(0,0,1,1) }
         @{ index = 6; expected = @(0,0,1,2) }
         @{ index = 7; expected = @(0,0,1,3) }
                      
         @{ index = 8; expected = @(0,1,0,0) }
         @{ index = 9; expected = @(0,1,0,1) }
         @{ index = 10; expected = @(0,1,0,2) }
         @{ index = 11; expected = @(0,1,0,3) }
                      
         @{ index = 12; expected = @(0,1,1,0) }
         @{ index = 13; expected = @(0,1,1,1) }
         @{ index = 14; expected = @(0,1,1,2) }
         @{ index = 15; expected = @(0,1,1,3) }
    ) {
        $dimensions = @(1, 2, 2, 4)
        $matrix = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        GetNdMatrixElement $expected $matrix $dimensions | Should -Be $matrix[$index]
        GetNdMatrixIndex $index $dimensions | Should -Be $expected
    }
}

Describe 'Matrix-Set' -Tag "set" {
    It "Should set a matrix element" -TestCases @(
        @{ value = "set"; index = @(0,0,0,0); arrayIndex = 0 }
        @{ value = "ones"; index = @(0,1,1,1); arrayIndex = 13 }
    ) {
        $dimensions = @(1, 2, 2, 4)
        $matrix = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        SetNdMatrixElement $value $index $matrix $dimensions
        $matrix[$arrayIndex] | Should -Be $value
    }
}

Describe "Platform Matrix Generation" -Tag "generate" {
    BeforeEach {
        $matrixConfigForGenerate = @"
{
    "displayNames": {
        "--enableFoo": "withfoo"
    },
    "matrix": {
        "operatingSystem": [
          "windows-2019",
          "ubuntu-18.04",
          "macOS-10.15"
        ],
        "framework": [
          "net461",
          "netcoreapp2.1"
        ],
        "additionalArguments": [
            "",
            "--enableFoo"
        ]
    },
    "include": [
      {
        "operatingSystem": "windows-2019",
        "framework": "net461",
        "additionalTestArguments": "/p:UseProjectReferenceToAzureClients=true"
      }
    ],
    "exclude": [
      {
        "foo": "bar"
      },
      {
        "foo2": "bar2"
      }
    ]
}
"@
        $generateConfig = GetMatrixConfigFromJson $matrixConfigForGenerate
    }

    It "Should get matrix dimensions from Nd parameters" {
        GetMatrixDimensions $generateConfig.matrixParameters | Should -Be 3, 2, 2
    }

    It "Should use name overrides from displayNames" {
        $dimensions = GetMatrixDimensions $generateConfig.matrixParameters
        $matrix = GenerateFullMatrix $generateConfig.matrixParameters $generateconfig.displayNamesLookup

        $element = GetNdMatrixElement @(0, 0, 0) $matrix $dimensions
        $element.name | Should -Be "windows2019_net461"

        $element = GetNdMatrixElement @(1, 1, 1) $matrix $dimensions
        $element.name | Should -Be "ubuntu1804_netcoreapp21_withFoo"

        $element = GetNdMatrixElement @(2, 1, 1) $matrix $dimensions
        $element.name | Should -Be "macOS1015_netcoreapp21_withFoo"
    }

    It "Should initialize an N-dimensional matrix from all parameter permutations" {
        $dimensions = GetMatrixDimensions $generateConfig.matrixParameters
        $matrix = GenerateFullMatrix $generateConfig.matrixParameters $generateConfig.displayNamesLookup
        $matrix.Count | Should -Be 12

        $element = $matrix[0].parameters
        $element.operatingSystem | Should -Be "windows-2019"
        $element.framework | Should -Be "net461"
        $element.additionalArguments | Should -Be ""

        $element = GetNdMatrixElement @(1, 1, 1) $matrix $dimensions
        $element.parameters.operatingSystem | Should -Be "ubuntu-18.04"
        $element.parameters.framework | Should -Be "netcoreapp2.1"
        $element.parameters.additionalArguments | Should -Be "--enableFoo"

        $element = GetNdMatrixElement @(2, 1, 1) $matrix $dimensions
        $element.parameters.operatingSystem | Should -Be "macOS-10.15"
        $element.parameters.framework | Should -Be "netcoreapp2.1"
        $element.parameters.additionalArguments | Should -Be "--enableFoo"
    }

    It "Should initialize a sparse matrix from an N-dimensional matrix" -TestCases @(
        @{ i = 0; name = "windows2019_net461"; operatingSystem = "windows-2019"; framework = "net461"; additionalArguments = ""; }
        @{ i = 1; name = "ubuntu1804_netcoreapp21_withfoo"; operatingSystem = "ubuntu-18.04"; framework = "netcoreapp2.1"; additionalArguments = "--enableFoo"; }
        @{ i = 2; name = "macOS1015_net461"; operatingSystem = "macOS-10.15"; framework = "net461"; additionalArguments = ""; }
    ) {
        $sparseMatrix = GenerateSparseMatrix $generateConfig.matrixParameters $generateConfig.displayNamesLookup
        $dimensions = GetMatrixDimensions $generateConfig.matrixParameters
        $size = ($dimensions | Measure-Object -Maximum).Maximum
        $sparseMatrix.Count | Should -Be $size

        $sparseMatrix[$i].name | Should -Be $name
        $element = $sparseMatrix[$i].parameters
        $element.operatingSystem | Should -Be $operatingSystem
        $element.framework | Should -Be $framework
        $element.additionalArguments | Should -Be $additionalArguments
    }

    It "Should generate a sparse matrix from an N-dimensional matrix config" {
        $sparseMatrix = GenerateMatrix $generateConfig "sparse"
        $sparseMatrix.Length | Should -Be 4
    }

    It "Should initialize a full matrix from an N-dimensional matrix config" {
        $matrix = GenerateMatrix $generateConfig "all"
        $matrix.Length | Should -Be 13
    }
}

Describe "Config File Object Conversion" -Tag "convert" {
    BeforeEach {
        $config = GetMatrixConfigFromJson $matrixConfig
    }

    It "Should convert a matrix config" {
        $config.matrixParameters[0].Name | Should -Be "operatingSystem"
        $config.matrixParameters[0].Flatten()[0].Value | Should -Be "windows-2019"

        $config.displayNamesLookup | Should -BeOfType [Hashtable]
        $config.displayNamesLookup["--enableFoo"] | Should -Be "withFoo"

        $config.include.Length | Should -Be 1
        $config.exclude.Length | Should -Be 3
    }
}

Describe "Platform Matrix Post Transformation" -Tag "transform" {
    BeforeEach {
        $config = GetMatrixConfigFromJson $matrixConfig
    }

    It "Should match partial matrix elements" -TestCases @(
        @{ source = [Ordered]@{ a = 1; b = 2; }; target = [Ordered]@{ a = 1 }; expected = $true }
        @{ source = [Ordered]@{ a = 1; b = 2; }; target = [Ordered]@{ a = 1; b = 2 }; expected = $true }
        @{ source = [Ordered]@{ a = 1; b = 2; }; target = [Ordered]@{ a = 1; b = 2; c = 3 }; expected = $false }
        @{ source = [Ordered]@{ a = 1; b = 2; }; target = [Ordered]@{ }; expected = $false }
        @{ source = [Ordered]@{ }; target = [Ordered]@{ a = 1; b = 2; }; expected = $false }
    ) {
        MatrixElementMatch $source $target | Should -Be $expected
    }

    It "Should remove matrix elements based on exclude filters" {
        $matrix = GenerateFullMatrix $config.matrixParameters $config.displayNamesLookup
        $withExclusion = ProcessExcludes $matrix $config.exclude
        $withExclusion.Length | Should -Be 5

        $matrix = GenerateSparseMatrix $config.matrixParameters $config.displayNamesLookup
        [array]$withExclusion = ProcessExcludes $matrix $config.exclude
        $withExclusion.Length | Should -Be 1
    }

    It "Should add matrix elements based on include elements" {
        $matrix = GenerateFullMatrix $config.matrixParameters $config.displayNamesLookup
        $withInclusion = ProcessIncludes $config $matrix "all"
        $withInclusion.Length | Should -Be 15
    }

    It "Should include and exclude values with a matrix" {
        [Array]$matrix = GenerateMatrix $config "all"
        $matrix.Length | Should -Be 8

        $matrix[0].name | Should -Be "windows2019_netcoreapp21"
        $matrix[0].parameters.operatingSystem | Should -Be "windows-2019"
        $matrix[0].parameters.framework | Should -Be "netcoreapp2.1"
        $matrix[0].parameters.additionalArguments | Should -Be ""

        $matrix[1].name | Should -Be "windows2019_netcoreapp21_withfoo"
        $matrix[1].parameters.operatingSystem | Should -Be "windows-2019"
        $matrix[1].parameters.framework | Should -Be "netcoreapp2.1"
        $matrix[1].parameters.additionalArguments | Should -Be "--enableFoo"

        $matrix[2].name | Should -Be "ubuntu1804_net461"
        $matrix[2].parameters.framework | Should -Be "net461"
        $matrix[2].parameters.operatingSystem | Should -Be "ubuntu-18.04"
        $matrix[2].parameters.additionalArguments | Should -Be ""

        $matrix[4].name | Should -Be "macOS1015_net461"
        $matrix[4].parameters.framework | Should -Be "net461"
        $matrix[4].parameters.operatingSystem | Should -Be "macOS-10.15"
        $matrix[4].parameters.additionalArguments | Should -Be ""

        $matrix[7].name | Should -Be "windows2019_net50_enableWindowsFoo"
        $matrix[7].parameters.framework | Should -Be "net50"
        $matrix[7].parameters.operatingSystem | Should -Be "windows-2019"
        $matrix[7].parameters.additionalArguments | Should -Be "--enableWindowsFoo"
    }

    It "Should parse a config with an empty base matrix" {
        $matrixConfigForIncludeOnly = @"
{
    "include": [
        {
            "operatingSystem": "windows-2019",
            "framework": "net461"
        }
    ]
}
"@

        $config = GetMatrixConfigFromJson $matrixConfigForIncludeOnly
        [Array]$matrix = GenerateMatrix $config "all"
        $matrix.Length | Should -Be 1
        $matrix[0].name | Should -Be "windows2019_net461"
    }

    It "Should parse a config with an empty include" {
        $matrixConfigForIncludeOnly = @"
{
    "matrix": {
        "operatingSystem": "windows-2019",
        "framework": "net461"
    }
}
"@

        $config = GetMatrixConfigFromJson $matrixConfigForIncludeOnly
        [Array]$matrix = GenerateMatrix $config "all"
        $matrix.Length | Should -Be 1
        $matrix[0].name | Should -Be "windows2019_net461"
    }
}

Describe "Platform Matrix Generation With Object Fields" -Tag "objectfields" {
    BeforeEach {
        $matrixConfigForObject = @"
{
    "matrix": {
        "testObject": {
            "testObjectName1": { "testObject1Value1": "1", "testObject1Value2": "2" },
            "testObjectName2": { "testObject2Value1": "1", "testObject2Value2": "2" }
        },
        "secondTestObject": {
            "secondTestObjectName1": { "secondTestObject1Value1": "1", "secondTestObject1Value2": "2" }
        },
        "testField": [ "footest", "bartest" ]
    },
    "include": [
      {
        "testObjectInclude": {
            "testObjectIncludeName": { "testObjectValue1": "1", "testObjectValue2": "2" }
        },
        "testField": "footest"
      }
    ]
}
"@
        $objectFieldConfig = GetMatrixConfigFromJson $matrixConfigForObject
    }

    It "Should parse dimensions properly" {
        [Array]$dimensions = GetMatrixDimensions $objectFieldConfig.matrixParameters
        $dimensions.Length | Should -Be 3
        $dimensions[0] | Should -Be 2
        $dimensions[1] | Should -Be 1
        $dimensions[2] | Should -Be 2
    }

    It "Should populate a sparse matrix dimensions properly" {
        [Array]$matrix = GenerateMatrix $objectFieldConfig "sparse"
        $matrix.Length | Should -Be 3

        $matrix[0].name | Should -Be "testObjectName1_secondTestObjectName1_footest"
        $matrix[0].parameters.testField | Should -Be "footest"
        $matrix[0].parameters.testObject1Value1 | Should -Be "1"
        $matrix[0].parameters.testObject1Value2 | Should -Be "2"
        $matrix[0].parameters.secondTestObject1Value1 | Should -Be "1"
        $matrix[0].parameters.Count | Should -Be 5

        $matrix[1].name | Should -Be "testObjectName2_secondTestObjectName1_bartest"
        $matrix[1].parameters.testField | Should -Be "bartest"
        $matrix[1].parameters.testObject2Value1 | Should -Be "1"
        $matrix[1].parameters.testObject2Value2 | Should -Be "2"
        $matrix[1].parameters.secondTestObject1Value1 | Should -Be "1"
        $matrix[1].parameters.Count | Should -Be 5

        $matrix[2].name | Should -Be "testObjectIncludeName_footest"
        $matrix[2].parameters.testField | Should -Be "footest"
        $matrix[2].parameters.testObjectValue1 | Should -Be "1"
        $matrix[2].parameters.testObjectValue2 | Should -Be "2"
        $matrix[2].parameters.Count | Should -Be 3
    }

    It "Should splat matrix entries that are objects into key/values" {
        [Array]$matrix = GenerateMatrix $objectFieldConfig "all"
        $matrix.Length | Should -Be 5

        $matrix[0].name | Should -Be "testObjectName1_secondTestObjectName1_footest"
        $matrix[0].parameters.testField | Should -Be "footest"
        $matrix[0].parameters.testObject1Value1 | Should -Be "1"
        $matrix[0].parameters.testObject1Value2 | Should -Be "2"
        $matrix[0].parameters.secondTestObject1Value1 | Should -Be "1"
        $matrix[0].parameters.Count | Should -Be 5

        $matrix[3].name | Should -Be "testObjectName2_secondTestObjectName1_bartest"
        $matrix[3].parameters.testField | Should -Be "bartest"
        $matrix[3].parameters.testObject2Value1 | Should -Be "1"
        $matrix[3].parameters.testObject2Value2 | Should -Be "2"
        $matrix[3].parameters.secondTestObject1Value1 | Should -Be "1"
        $matrix[3].parameters.Count | Should -Be 5

        $matrix[4].name | Should -Be "testObjectIncludeName_footest"
        $matrix[4].parameters.testField | Should -Be "footest"
        $matrix[4].parameters.testObjectValue1 | Should -Be "1"
        $matrix[4].parameters.testObjectValue2 | Should -Be "2"
        $matrix[4].parameters.Count | Should -Be 3
    }
}

Describe "Platform Matrix Display Names" -Tag "displaynames" {
    BeforeEach {
        $matrixConfigForGenerate = @"
{
    "displayNames": {
        "--enableFoo": "withfoo"
    },
    "matrix": {
        "operatingSystem": "ubuntu-18.04",
        "framework": [
          "net461",
          "netcoreapp2.1"
        ],
        "TestNullField": null,
        "TestObjectField": {
            "TestObjectValueName": {
                "foo": "bar",
                "baz": "qux"
            }
        }
    }
}
"@
        $generateConfig = GetMatrixConfigFromJson $matrixConfigForGenerate
    }

    It "Should enforce valid display name format" {
        $generateconfig.displayNamesLookup["net461"] = '123.Some.456.Invalid_format-name$(foo)'
        $generateconfig.displayNamesLookup["netcoreapp2.1"] = (New-Object string[] 150) -join "a"
        $dimensions = GetMatrixDimensions $generateConfig.matrixParameters
        $matrix = GenerateFullMatrix $generateconfig.matrixParameters $generateconfig.displayNamesLookup

        $matrix[0].name | Should -Be "ubuntu1804_123some456invalid_formatnamefoo_TestObjectValueName"

        $matrix[1].name.Length | Should -Be 100
        # The withfoo part of the argument gets cut off at the character limit
        $matrix[1].name | Should -BeLike "ubuntu1804_aaaaaaaaaaaaaaaaa*"
    }

    It "Should generate a display name with null and object values" {
        $matrix = GenerateMatrix $generateConfig "sparse"
        $matrix[0].name | Should -Be "ubuntu1804_net461_TestObjectValueName"
    }
}
