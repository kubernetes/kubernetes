# Safe JSON

This repository contains a fork of the `encoding/json` package from Go 1.6.

The following changes were made:

* Object deserialization uses case-sensitive member name matching instead of
  [case-insensitive matching](https://www.ietf.org/mail-archive/web/json/current/msg03763.html).
  This is to avoid differences in the interpretation of JOSE messages between
  go-jose and libraries written in other languages.
* When deserializing a JSON object, we check for duplicate keys and reject the
  input whenever we detect a duplicate. Rather than trying to work with malformed
  data, we prefer to reject it right away.
