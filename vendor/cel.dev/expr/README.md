# Common Expression Language

The Common Expression Language (CEL) implements common semantics for expression
evaluation, enabling different applications to more easily interoperate.

Key Applications

*   Security policy: organizations have complex infrastructure and need common
    tooling to reason about the system as a whole
*   Protocols: expressions are a useful data type and require interoperability
    across programming languages and platforms.


Guiding philosophy:

1.  Keep it small & fast.
    *   CEL evaluates in linear time, is mutation free, and not Turing-complete.
        This limitation is a feature of the language design, which allows the
        implementation to evaluate orders of magnitude faster than equivalently
        sandboxed JavaScript.
2.  Make it extensible.
    *   CEL is designed to be embedded in applications, and allows for
        extensibility via its context which allows for functions and data to be
        provided by the software that embeds it.
3.  Developer-friendly.
    *   The language is approachable to developers. The initial spec was based
        on the experience of developing Firebase Rules and usability testing
        many prior iterations.
    *   The library itself and accompanying toolings should be easy to adopt by
        teams that seek to integrate CEL into their platforms.

The required components of a system that supports CEL are:

*   The textual representation of an expression as written by a developer. It is
    of similar syntax to expressions in C/C++/Java/JavaScript
*   A representation of the program's abstract syntax tree (AST).
*   A compiler library that converts the textual representation to the binary
    representation. This can be done ahead of time (in the control plane) or
    just before evaluation (in the data plane).
*   A context containing one or more typed variables, often protobuf messages.
    Most use-cases will use `attribute_context.proto`
*   An evaluator library that takes the binary format in the context and
    produces a result, usually a Boolean.

For use cases which require persistence or cross-process communcation, it is
highly recommended to serialize the type-checked expression as a protocol
buffer. The CEL team will maintains canonical protocol buffers for ASTs and
will keep these versions identical and wire-compatible in perpetuity:

*  [CEL canonical](https://github.com/google/cel-spec/tree/master/proto/cel/expr)
*  [CEL v1alpha1](https://github.com/googleapis/googleapis/tree/master/google/api/expr/v1alpha1)


Example of boolean conditions and object construction:

``` c
// Condition
account.balance >= transaction.withdrawal
    || (account.overdraftProtection
    && account.overdraftLimit >= transaction.withdrawal  - account.balance)

// Object construction
common.GeoPoint{ latitude: 10.0, longitude: -5.5 }
```

For more detail, see:

*   [Introduction](doc/intro.md)
*   [Language Definition](doc/langdef.md)

Released under the [Apache License](LICENSE).

Disclaimer: This is not an official Google product.
