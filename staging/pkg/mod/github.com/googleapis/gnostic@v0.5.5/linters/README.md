# Linters

This directory contains linters that can be used to check Gnostic models.

Linters are plugins that generate no files but instead return messages in their
responses. Each message can include a level, an identifier, text, and a key
path in an API description associated with that message. Messages are collected
by gnostic and written to a common output file, allowing multiple linter
plugins to be invoked in a single gnostic run.

The following invocation runs the `gnostic-lint-paths` and
`gnostic-lint-descriptions` plugins and writes their messages to a file named
`lint.pb`.

```
% gnostic examples/v2.0/yaml/petstore.yaml --lint-paths --lint-descriptions --messages-out=lint.pb
```

Message files can be displayed using the `report-messages` tool in the `apps`
directory.
