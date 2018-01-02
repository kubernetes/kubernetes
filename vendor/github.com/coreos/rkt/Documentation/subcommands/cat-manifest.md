# rkt cat-manifest

For debugging or inspection you may want to extract the PodManifest to stdout.

```
# rkt cat-manifest UUID
{
  "acVersion":"0.8.10",
  "acKind":"PodManifest"
...
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--pretty-print` |  `true` | `true` or `false` | Apply indent to format the output |

## Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
