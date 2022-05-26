# garif

A GO package to create and manipulate SARIF logs.

SARIF, from _Static Analysis Results Interchange Format_, is a standard JSON-based format for the output of static analysis tools defined and promoted by [OASIS](https://www.oasis-open.org/).

Current supported version of the standard is [SARIF-v2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/csprd01/sarif-v2.1.0-csprd01.html
).

## Usage

The package provides access to every element of the SARIF model, therefore you are free to manipulate it at every detail.

The package also provides constructors functions (`New...`) and decorators methods (`With...`) that simplify the creation of SARIF files for common use cases.

Using these constructors and decorators we can easily create the example SARIF file of the [Microsoft SARIF pages](https://github.com/microsoft/sarif-tutorials/blob/master/docs/1-Introduction.md)


```go
import to `github.com/chavacava/garif`

// ...

rule := garif.NewRule("no-unused-vars").
		WithHelpUri("https://eslint.org/docs/rules/no-unused-vars").
		WithShortDescription("disallow unused variables").
		WithProperties("category", "Variables")

driver := garif.NewDriver("ESLint").
		WithInformationUri("https://eslint.org").
		WithRules(rule)

run := garif.NewRun(NewTool(driver)).
		WithArtifactsURIs("file:///C:/dev/sarif/sarif-tutorials/samples/Introduction/simple-example.js")

run.WithResult(rule.Id, "'x' is assigned a value but never used.", "file:///C:/dev/sarif/sarif-tutorials/samples/Introduction/simple-example.js", 1, 5)

logFile := garif.NewLogFile([]*Run{run}, Version210)

logFile.Write(os.Stdout)
```

## Why this package?
This package was initiated during my works on adding to [`revive`](https://github.com/mgechev/revive) a SARIF output formatter.
I've tried to use [go-sarif](https://github.com/owenrumney/go-sarif) by [Owen Rumney](https://github.com/owenrumney) but it is too focused in the use case of the static analyzer [tfsec](https://tfsec.dev) so I've decided to create a package flexible enough to generate SARIF files in broader cases.

## More information about SARIF
For more information about SARIF, you can visit the [Oasis Open](https://www.oasis-open.org/committees/tc_home.php?wg_abbrev=sarif) site.


## Contributing 
Of course, contributions are welcome!