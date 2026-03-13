# Sorted Linter

This linter checks if feature gates in Kubernetes code are sorted alphabetically in const and var blocks, as well as
in maps.

## Purpose

In Kubernetes, feature gates should be listed in alphabetical, case-sensitive (upper before any lower case character)
order to reduce the risk of code conflicts and improve readability. This linter enforces this convention by checking
if feature gates are properly sorted in:

1. **Feature gate declarations** in `const` and `var` blocks
2. **Feature gate map keys** in map literals

## How It Works

The linter analyzes Go AST to find:
- `const` and `var` blocks containing feature gate declarations
- Map literals with feature gate keys (e.g., `map[featuregate.Feature]...`)

It extracts feature names, preserves associated comments, and checks alphabetical ordering. If not sorted, it reports
an error with a detailed diff showing the current order versus the expected order.

## Supported Patterns

### Feature Gate Declarations
```go
const (
    FeatureA featuregate.Feature = "FeatureA"
    FeatureB featuregate.Feature = "FeatureB"
)

var (
    MyFeature featuregate.Feature = "MyFeature"
    OtherFeature featuregate.Feature = "OtherFeature"
)
```

### Feature Gate Maps
```go
var DefaultFeatureGate = map[featuregate.Feature]featuregate.VersionedSpecs{
    FeatureA: {...},
    FeatureB: {...},
    genericfeatures.APIServerIdentity: {...}, // selector expressions supported
}
```

**Note**: The linter only works for grouped declarations (`const (...)` or `var (...)`). Individual declarations are not checked:
```go
// These are NOT checked
const FeatureA featuregate.Feature = "FeatureA"
const FeatureB featuregate.Feature = "FeatureB"
```

## Installation

### As a golangci-lint plugin

1. Build the plugin:

```bash
cd hack/tools/golangci-lint/sorted
go build -buildmode=plugin -o sorted.so ./plugin/
```

2. Add the plugin to your `.golangci.yml` configuration:

```yaml
linters:
  settings:
    custom:
      sorted:
        path: /path/to/sorted.so
        description: Checks if feature gates are sorted alphabetically
        original-url: k8s.io/kubernetes/hack/tools/golangci-lint/sorted
        settings:
          debug: false
          files:
            - path/to/additional/file.go
```

### As a standalone tool

```bash
cd hack/tools/golangci-lint/sorted
go run main.go path/to/file.go
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| debug | bool | false | Enable debug logging |
| files | []string | (see below) | Files to check for feature gate sorting. If specified, only these files will be checked |

### Configuration Examples

```yaml
# Enable debug mode
linters:
  settings:
    custom:
      sorted:
        settings:
          debug: true

# Check specific files only
linters:
  settings:
    custom:
      sorted:
        settings:
          files:
            - pkg/features/kube_features.go
            - staging/src/k8s.io/apiserver/pkg/features/kube_features.go
```

## Default Files Checked

When no `files` configuration is provided, the linter checks these Kubernetes files:

- `cmd/kubeadm/app/features/features.go`
- `pkg/features/kube_features.go`
- `staging/src/k8s.io/apiserver/pkg/features/kube_features.go`
- `staging/src/k8s.io/client-go/features/known_features.go`
- `staging/src/k8s.io/controller-manager/pkg/features/kube_features.go`
- `staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go`
- `test/e2e/feature/feature.go`
- `test/e2e/environment/environment.go`

## Usage

### With golangci-lint

```bash
# Run all linters including sorted
golangci-lint run

# Run only sorted
golangci-lint run --enable=sorted --disable-all

# Enable if disabled by default
golangci-lint run -Esorted
```

### Enabling the linter

Custom linters follow standard golangci-lint rules:

- Enabled by default unless `linters.disable-all: true` is set
- Can be explicitly enabled with `linters.enable: [sorted]`
- Can be enabled via command line: `golangci-lint run -Esorted`

## Examples

### ✅ Correctly Sorted

```go
const (
    // Comments are preserved
    FeatureA featuregate.Feature = "FeatureA"
    FeatureB featuregate.Feature = "FeatureB"
    FeatureC featuregate.Feature = "FeatureC"
)

var DefaultSpecs = map[featuregate.Feature]featuregate.VersionedSpecs{
    FeatureA: {...},
    FeatureB: {...},
    genericfeatures.APIServerIdentity: {...},
}
```

### ❌ Incorrectly Sorted (triggers linter error)

```go
const (
    FeatureC featuregate.Feature = "FeatureC"  // Wrong order
    FeatureA featuregate.Feature = "FeatureA"
    FeatureB featuregate.Feature = "FeatureB"
)

var DefaultSpecs = map[featuregate.Feature]featuregate.VersionedSpecs{
    FeatureB: {...},  // Wrong order
    FeatureA: {...},
}
```

## Error Output

When sorting issues are detected, the linter provides a unified diff:

```
not sorted alphabetically:
@@ -1,4 +1,4 @@
 const (
-	FeatureC = value
-	FeatureA = value
 	FeatureB = value
+	FeatureA = value
+	FeatureC = value
 )
```

## Files Checked

By default, this linter checks the following files in the Kubernetes codebase:

- `pkg/features/kube_features.go`
- `staging/src/k8s.io/apiserver/pkg/features/kube_features.go`
- `staging/src/k8s.io/client-go/features/known_features.go`
- `staging/src/k8s.io/controller-manager/pkg/features/kube_features.go`
- `staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go`
- `test/e2e/feature/feature.go`
- `test/e2e/environment/environment.go`

## Integration with CI

This linter is integrated into the Kubernetes CI pipeline to ensure consistent feature gate ordering across the
codebase. Run it locally before submitting pull requests that modify feature gates.

## Troubleshooting

1. **Enable debug mode**: Set `debug: true` in your configuration to see processing details
2. **Check plugin build**: Ensure the plugin is correctly built with `go build -buildmode=plugin`
3. **Verify file paths**: Confirm target files are in the default list or explicitly configured
4. **Test with standalone tool**: Run `go run main.go path/to/file.go` to test specific files

## Implementation Details

- Uses Go's `go/ast` package for parsing
- Preserves comments and formatting context
- Supports both simple identifiers and selector expressions (e.g., `package.FeatureName`)
- Generates unified diffs using `github.com/pmezard/go-difflib`

For more information on custom linters in golangci-lint, refer to the [official documentation](https://golangci-lint.run/contributing/new-linters/).
