# Development

Porcupine uses the standard [Go toolchain][golang] for development.

[golang]: https://go.dev/

## Testing

You can run the tests with:

```bash
go test ./...
```

Some of the tests generate visualizations which must be manually inspected. To make the tests print the path to the generated HTML files, run the tests with the `-v` flag.

## Formatting

You can run the Go code formatter with:

```bash
go fmt ./...
```

Additionally, this project uses [Prettier] to format HTML, CSS, and JavaScript files. You can run Prettier (using [npx]) with:

```bash
npx prettier -w '**/*.{html,css,js}'
```

[Prettier]: https://prettier.io/
[npx]: https://docs.npmjs.com/cli/v11/commands/npx

## Static analysis

You can run Go's built-in `vet` tool with:

```bash
go vet ./...
```

This project additionally uses the [staticcheck] tool. You can install it with:

```bash
go install honnef.co/go/tools/cmd/staticcheck@latest
```

You can run staticcheck with:

```bash
staticcheck -f stylish ./...
```

This project uses [XO] to lint JavaScript code. You can run XO with:

```bash
npx xo
```

[staticcheck]: https://staticcheck.dev/
[XO]: https://github.com/xojs/xo

## Continuous integration

Testing and static analysis is [run in CI][ci-test].

[ci-test]: .github/workflows/ci.yml
