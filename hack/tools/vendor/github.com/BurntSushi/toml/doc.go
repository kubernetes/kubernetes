/*
Package toml implements decoding and encoding of TOML files.

This package supports TOML v1.0.0, as listed on https://toml.io

There is also support for delaying decoding with the Primitive type, and
querying the set of keys in a TOML document with the MetaData type.

The github.com/BurntSushi/toml/cmd/tomlv package implements a TOML validator,
and can be used to verify if TOML document is valid. It can also be used to
print the type of each key.
*/
package toml
