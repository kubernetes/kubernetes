Import-Module Pester


BeforeAll {
    . $PSScriptRoot/../job-matrix-functions.ps1

    function CompareMatrices([Array]$matrix, [Array]$expected) {
        $matrix.Length | Should -Be $expected.Length

        for ($i = 0; $i -lt $matrix.Length; $i++) {
            foreach ($entry in $matrix[$i]) {
                $entry.name | Should -Be $expected[$i].name
                foreach ($param in $entry.parameters.GetEnumerator()) {
                    $param.Value | Should -Be $expected[$i].parameters[$param.Name]
                }
            }
        }
    }
}

Describe "Platform Matrix nonSparse" -Tag "nonsparse" {
    BeforeEach {
        $matrixJson = @'
{
    "matrix": {
        "testField1": [ 1, 2 ],
        "testField2": [ 1, 2, 3 ],
        "testField3": [ 1, 2, 3, 4 ],
    }
}
'@
        $config = GetMatrixConfigFromJson $matrixJson
    }

    It "Should process nonSparse parameters" {
        $parameters, $nonSparse = ProcessNonSparseParameters $config.matrixParameters "testField1","testField3"

        $parameters.Count | Should -Be 1
        $parameters[0].Name | Should -Be "testField2"
        $parameters[0].Value | Should -Be 1,2,3

        $nonSparse.Count | Should -Be 2
        $nonSparse[0].Name | Should -Be "testField1"
        $nonSparse[0].Value | Should -Be 1,2
        $nonSparse[1].Name | Should -Be "testField3"
        $nonSparse[1].Value | Should -Be 1,2,3,4

        $parameters, $nonSparse = ProcessNonSparseParameters $config.matrixParameters "testField3"
        $parameters.Count | Should -Be 2
        ($parameters).Name -match "testField3" | Should -Be $null

        $nonSparse.Count | Should -Be 1
        $nonSparse[0].Name | Should -Be "testField3"
        $nonSparse[0].Value | Should -Be 1,2,3,4
    }

    It "Should ignore nonSparse with all selection" {
        $matrix = GenerateMatrix $config "all" -nonSparseParameters "testField3"
        $matrix.Length | Should -Be 24
    }

    It "Should combine sparse matrix with nonSparse parameters" {
        $matrix = GenerateMatrix $config "sparse" -nonSparseParameters "testField3"
        $matrix.Length | Should -Be 12
    }

    It "Should combine with multiple nonSparse fields" {
        $matrixJson = @'
{
    "matrix": {
        "testField1": [ 1, 2 ],
        "testField2": [ 1, 2 ],
        "testField3": [ 31, 32 ],
        "testField4": [ 41, 42 ]
    }
}
'@
        $config = GetMatrixConfigFromJson $matrixJson

        $matrix = GenerateMatrix $config "all" -nonSparseParameters "testField3","testField4"
        $matrix.Length | Should -Be 16

        $matrix = GenerateMatrix $config "sparse" -nonSparseParameters "testField3","testField4"
        $matrix.Length | Should -Be 8
    }

    It "Should apply nonSparseParameters to an imported matrix" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "TestField1": "test1"
    },
    "exclude": [ { "Baz": "importedBaz" } ]
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "TestField1": "test1", "Foo": "foo1", "Bar": "bar1" },
    "name": "test1_foo1_bar1"
  },
  {
    "parameters": { "TestField1": "test1", "Foo": "foo1", "Bar": "bar2" },
    "name": "test1_foo1_bar2"
  },
  {
    "parameters": { "TestField1": "test1", "Foo": "foo2", "Bar": "bar1" },
    "name": "test1_foo2_bar1"
  },
  {
    "parameters": { "TestField1": "test1", "Foo": "foo2", "Bar": "bar2" },
    "name": "test1_foo2_bar2"
  }
]
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse" -nonSparseParameters "Foo"
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 4
        CompareMatrices $matrix $expected
    }
}

Describe "Platform Matrix Import" -Tag "import" {
    It "Should generate a sparse matrix where the entire base matrix is imported" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json"
    },
    "include": [
        {
            "fooinclude": "fooinclude"
        }
    ]
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "Foo": "foo1", "Bar": "bar1" },
    "name": "foo1_bar1"
  },
  {
    "parameters": { "Foo": "foo2", "Bar": "bar2" },
    "name": "foo2_bar2"
  },
  {
    "parameters": { "Baz": "importedBaz" },
    "name": "importedBazName"
  },
  {
    "parameters": { "fooinclude": "fooinclude" },
    "name": "fooinclude"
  },
]
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse"
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 4
        CompareMatrices $matrix $expected
    }

    It "Should import a matrix and combine with length=1 vectors" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "TestField1": "test1",
        "TestField2": "test2"
    },
    "exclude": [ { "Baz": "importedBaz" } ]
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "TestField1": "test1", "TestField2": "test2", "Foo": "foo1", "Bar": "bar1" },
    "name": "test1_test2_foo1_bar1"
  },
  {
    "parameters": { "TestField1": "test1", "TestField2": "test2", "Foo": "foo2", "Bar": "bar2" },
    "name": "test1_test2_foo2_bar2"
  }
]
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse"
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 2
        CompareMatrices $matrix $expected
    }

    It "Should generate a matrix with nonSparseParameters and an imported sparse matrix" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "testField": [ "test1", "test2" ]
    }
}
'@
        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse" -nonSparseParameters "testField"

        $matrix.Length | Should -Be 6

        $matrix[0].name | Should -Be test1_foo1_bar1
        $matrix[0].parameters.testField | Should -Be "test1"
        $matrix[0].parameters.Foo | Should -Be "foo1"
        $matrix[2].name | Should -Be test1_importedBazName
        $matrix[2].parameters.testField | Should -Be "test1"
        $matrix[2].parameters.Baz | Should -Be "importedBaz"
        $matrix[4].name | Should -Be test2_foo2_bar2
        $matrix[4].parameters.testField | Should -Be "test2"
        $matrix[4].parameters.Foo | Should -Be "foo2"
    }

    It "Should source imported display name lookups" {
        $matrixJson = @'
{
    "displayNames": {
        "test1": "test1DisplayName",
        "importedBaz": "importedBazNameOverride"
    },
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "testField": [ "test1", "test2" ]
    }
}
'@
        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse" -nonSparseParameters "testField"

        $matrix[0].name | Should -Be test1DisplayName_foo1_bar1
        $matrix[2].name | Should -Be test1DisplayName_importedBazNameOverride
    }

    It "Should generate a sparse matrix with an imported sparse matrix" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "testField1": [ "test11", "test12" ],
        "testField2": [ "test21", "test22" ]
    }
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "testField1": "test11", "testField2": "test21", "Foo": "foo1", "Bar": "bar1" },
    "name": "test11_test21_foo1_bar1"
  },
  {
    "parameters": { "testField1": "test11", "testField2": "test21", "Foo": "foo2", "Bar": "bar2" },
    "name": "test11_test21_foo2_bar2"
  },
  {
    "parameters": { "testField1": "test11", "testField2": "test21", "Baz": "importedBaz" },
    "name": "test11_test21_importedBazName"
  },
  {
    "parameters": { "testField1": "test12", "testField2": "test22", "Foo": "foo1", "Bar": "bar1" },
    "name": "test12_test22_foo1_bar1"
  },
  {
    "parameters": { "testField1": "test12", "testField2": "test22", "Foo": "foo2", "Bar": "bar2" },
    "name": "test12_test22_foo2_bar2"
  },
  {
    "parameters": { "testField1": "test12", "testField2": "test22", "Baz": "importedBaz" },
    "name": "test12_test22_importedBazName"
  }
]
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse"
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 6
        CompareMatrices $matrix $expected
    }

    It "Should import a sparse matrix with import, include, and exclude" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "testField": [ "test1", "test2", "test3" ],
    },
    "include": [
      {
        "testImportIncludeName": [ "testInclude1", "testInclude2" ]
      }
    ],
    "exclude": [
      {
        "testField": "test1"
      },
      {
        "testField": "test3",
        "Baz": "importedBaz"
      }
    ]
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "testField": "test2", "Foo": "foo1", "Bar": "bar1" },
    "name": "test2_foo1_bar1"
  },
  {
    "parameters": { "testField": "test2", "Foo": "foo2", "Bar": "bar2" },
    "name": "test2_foo2_bar2"
  },
  {
    "parameters": { "testField": "test2", "Baz": "importedBaz" },
    "name": "test2_importedBazName"
  },
  {
    "parameters": { "testField": "test3", "Foo": "foo1", "Bar": "bar1" },
    "name": "test3_foo1_bar1"
  },
  {
    "parameters": { "testField": "test3", "Foo": "foo2", "Bar": "bar2" },
    "name": "test3_foo2_bar2"
  },
  {
    "parameters": { "testImportIncludeName": "testInclude1" },
    "name": "testInclude1"
  },
  {
    "parameters": { "testImportIncludeName": "testInclude2" },
    "name": "testInclude2"
  }
]
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse"
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 7
        CompareMatrices $matrix $expected
    }

    It "Should not combine matrices with duplicate keys" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "Foo": [ "fooOverride1", "fooOverride2" ],
    }
}
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson
        { GenerateMatrix $importConfig "sparse" } | Should -Throw
    }

}

Describe "Platform Matrix Replace" -Tag "replace" {
    It "Should parse replacement syntax" -TestCases @(
         @{ query = 'foo=bar/baz'; key = '^foo$'; value = '^bar$'; replace = 'baz' },
         @{ query = 'foo=\/p:bar/\/p:baz'; key = '^foo$'; value = '^\/p:bar$'; replace = '/p:baz' },
         @{ query = 'f\=o\/o=\/p:b\=ar/\/p:b\=az'; key = '^f\=o\/o$'; value = '^\/p:b\=ar$'; replace = '/p:b=az' },
         @{ query = 'foo=bar/'; key = '^foo$'; value = '^bar$'; replace = '' },
         @{ query = 'foo=/baz'; key = '^foo$'; value = '^$'; replace = 'baz' }
    ) {
        $parsed = ParseReplacement $query
        $parsed.key | Should -Be $key
        $parsed.value | Should -Be $value
        $parsed.replace | Should -Be $replace
    }

    It "Should fail for invalid replacement syntax" -TestCases @(
        @{ query = '' },
        @{ query = 'asdf' },
        @{ query = 'asdf=foo/bar/baz' },
        @{ query = 'asdf=foo=bar/baz' },
        @{ query = 'asdf=foo' }
    ) {
        { $parsed = ParseReplacement $query } | Should -Throw
        { $parsed = ParseReplacement $query } | Should -Throw
        { $parsed = ParseReplacement $query } | Should -Throw
        { $parsed = ParseReplacement $query } | Should -Throw
        { $parsed = ParseReplacement $query } | Should -Throw
    }
    
    It "Should replace values in a matrix" {
        $matrixJson = @'
{
    "matrix": {
        "Foo": [ "foo1", "foo2" ],
        "Bar": [ "bar1", "bar2" ]
    },
    "include": [ { "Baz": "baz1" } ]
}
'@

        $expectedMatrix = @'
[
  {
    "parameters": { "Foo": "foo1Replaced", "Bar": "bar1" },
    "name": "foo1Replaced_bar1"
  },
  {
    "parameters": { "Foo": "fooDefaultReplaced", "Bar": "bar2" },
    "name": "fooDefaultReplaced_bar2"
  },
  {
    "parameters": { "Baz": "bazReplaced" },
    "name": "bazReplaced"
  }
]
'@

        $replace = @(
            "Foo=foo1/foo1Replaced",
            "Foo=foo.*/fooDefaultReplaced",
            ".*=B.z\d/bazReplaced"
        )

        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix -config $importConfig -selectFromMatrixType "sparse" -replace $replace
        $expected = $expectedMatrix | ConvertFrom-Json -AsHashtable

        $matrix.Length | Should -Be 3
        CompareMatrices $matrix $expected
    }

    It "Should replace values in a matrix with import and nonSparseParameters" {
        $matrixJson = @'
{
    "matrix": {
        "$IMPORT": "./test-import-matrix.json",
        "testField": [ "test1", "test2" ]
    }
}
'@
        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "sparse" -nonSparseParameters "testField" -replace @("testField=test1/testReplaced", "Baz=.*/bazReplaced")

        $matrix.Length | Should -Be 6

        $matrix[0].name | Should -Be testReplaced_foo1_bar1
        $matrix[0].parameters.testField | Should -Be "testReplaced"
        $matrix[0].parameters.Foo | Should -Be "foo1"
        $matrix[2].name | Should -Be testReplaced_bazReplaced
        $matrix[2].parameters.testField | Should -Be "testReplaced"
        $matrix[2].parameters.Baz | Should -Be "bazReplaced"
        $matrix[4].name | Should -Be test2_foo2_bar2
        $matrix[4].parameters.testField | Should -Be "test2"
        $matrix[4].parameters.Foo | Should -Be "foo2"
    }

    It "Should replace values in groupings" {
        $matrixJson = @'
{
  "matrix": {
    "Agent": {
      "ubuntu-1804": { "OSVmImage": "MMSUbuntu18.04", "Pool": "azsdk-pool-mms-ubuntu-1804-general" }
    },
    "JavaTestVersion": [ "1.8", "1.11" ]
  }
}
'@
        $importConfig = GetMatrixConfigFromJson $matrixJson
        $matrix = GenerateMatrix $importConfig "all" -replace @("JavaTestVersion=1.8/2.0", "Pool=.*ubuntu.*/custom-ubuntu-pool")

        $matrix.Length | Should -Be 2
        # Replacements of inner values will preserve the grouping name
        $matrix[0].name | Should -Be "ubuntu1804_20"
        $matrix[0].parameters.JavaTestVersion | Should -Be "2.0"
        $matrix[0].parameters.Pool | Should -Be "custom-ubuntu-pool"
        $matrix[0].parameters.OSVmImage | Should -Be "MMSUbuntu18.04"

        # Make sure non-literal keys still replace under the hood
        $matrix = GenerateMatrix $importConfig "all" -replace ".*=.*ubuntu.*/custom-ubuntu-pool"

        $matrix.Length | Should -Be 2
        $matrix[0].name | Should -Be "ubuntu1804_18"
        $matrix[0].parameters.Pool | Should -Be "custom-ubuntu-pool"
    }

    It "Should replace values and apply regex capture groups" {
        $matrixJson = @'
{
  "matrix": {
    "Foo": [ "foo1", "foo2" ],
    "Bar": [ "bar1", "bar2" ]
  }
}
'@
        $importConfig = GetMatrixConfigFromJson $matrixJson
        $replace = 'Foo=(foo)1/$1ReplacedFoo1', 'B.*=(.*)2/$1ReplacedBar2'
        $matrix = GenerateMatrix $importConfig "sparse" -replace $replace

        $matrix.Length | Should -Be 2
        $matrix[0].name | Should -Be "fooReplacedFoo1_bar1"
        $matrix[0].parameters.Foo | Should -Be "fooReplacedFoo1"

        $matrix[1].name | Should -Be "foo2_barReplacedBar2"
        $matrix[1].parameters.Bar | Should -Be "barReplacedBar2"
    }

    It "Should only fully match a string for replace" {
        $matrixJson = @'
{
    "matrix": {
        "Foo": [ "foo1", "foo2" ],
        "Bar": "bar1"
    }
}
'@

        $importConfig = GetMatrixConfigFromJson $matrixJson

        $replace = @("Foo=foo/shouldNotReplaceFoo", "B=bar1/shouldNotReplaceBar")
        $matrix = GenerateMatrix -config $importConfig -selectFromMatrixType "sparse" -replace $replace

        $matrix.Length | Should -Be 2
        $matrix[0].parameters.Foo | Should -Be "foo1"
        $matrix[0].parameters.Bar | Should -Be "bar1"
        $matrix[1].parameters.Foo | Should -Be "foo2"
        $matrix[1].parameters.Bar | Should -Be "bar1"
    }
}
