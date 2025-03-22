# Kubernetes Validation Tags Documentation

This document lists the supported validation tags and their related information.

## Tags Overview

| Tag | Usage | Args | Description | Scopes |
|-----|-------------|------|-------------|----------|
| [`k8s:eachKey`](#k8seachkey) | k8s:eachKey=\<validation-tag\> | N/A | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:eachVal`](#k8seachval) | k8s:eachVal=\<validation-tag\> | N/A | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:forbidden`](#k8sforbidden) | k8s:forbidden | N/A | Indicates that a field may not be specified. | struct fields |
| [`k8s:immutable`](#k8simmutable) | k8s:immutable | N/A | Indicates that a field may not be updated. | list values, map values, struct fields, type definitions |
| [`k8s:listMapKey`](#k8slistmapkey) | k8s:listMapKey=\<field-json-name\> | N/A | Declares a named sub-field of a list's value-type to be part of the list-map key. | anywhere |
| [`k8s:listType`](#k8slisttype) | k8s:listType=\<type\> | N/A | Declares a list field's semantic type. | anywhere |
| [`k8s:minimum`](#k8sminimum) | k8s:minimum=\<integer\> | N/A | Indicates that a numeric field has a minimum value. | anywhere |
| [`k8s:opaqueType`](#k8sopaquetype) | k8s:opaqueType | N/A | Indicates that any validations declared on the referenced type will be ignored. If a referenced type's package is not included in the generator's current flags, this tag must be set, or code generation will fail (preventing silent mistakes). If the validations should not be ignored, add the type's package to the generator using the --readonly-pkg flag. | struct fields |
| [`k8s:optional`](#k8soptional) | k8s:optional | N/A | Indicates that a field is optional to clients. | struct fields |
| [`k8s:required`](#k8srequired) | k8s:required | N/A | Indicates that a field must be specified by clients. | struct fields |
| [`k8s:subfield`](#k8ssubfield) | k8s:subfield(\<field-json-name\>)=\<validation-tag\> | <field-json-name> | Declares a validation for a subfield of a struct. | anywhere |
| [`k8s:validateError`](#k8svalidateerror) | k8s:validateError=\<string\> | N/A | Always fails code generation (useful for testing). | anywhere |
| [`k8s:validateFalse`](#k8svalidatefalse) | k8s:validateFalse=\<payload\> | N/A | Always fails validation (useful for testing). | anywhere |
| [`k8s:validateTrue`](#k8svalidatetrue) | k8s:validateTrue=\<payload\> | N/A | Always passes validation (useful for testing). | anywhere |

## Tag Details

### k8s:eachKey

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<validation-tag\>** | The tag to evaluate for each value. | None |

### k8s:eachVal

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<validation-tag\>** | The tag to evaluate for each value. | None |

### k8s:forbidden

#### Payloads

null

### k8s:immutable

#### Payloads

null

### k8s:listMapKey

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<field-json-name\>** | The name of the field. | None |

### k8s:listType

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<type\>** | map | atomic | None |

### k8s:minimum

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<integer\>** | This field must be greater than or equal to x. | None |

### k8s:opaqueType

#### Payloads

null

### k8s:optional

#### Payloads

null

### k8s:required

#### Payloads

null

### k8s:subfield

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<validation-tag\>** | The tag to evaluate for the subfield. | None |

### k8s:validateError

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<string\>** | This string will be included in the error message. | None |

### k8s:validateFalse

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<none\>** |  | None |
| **\<quoted-string\>** | The generated code will include this string. | None |
| **\<json-object\>** |  | - `flags`: `<list-of-flag-string>` (values: ShortCircuit, NonError) (default: ``)<br>- `msg`: `<string>` (The generated code will include this string.) (default: ``)<br>- `typeArg`: `<string>` (The type arg in generated code (must be the value-type, not pointer).) (default: ``) |

### k8s:validateTrue

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<none\>** |  | None |
| **\<quoted-string\>** | The generated code will include this string. | None |
| **\<json-object\>** |  | - `flags`: `<list-of-flag-string>` (values: ShortCircuit, NonError) (default: ``)<br>- `msg`: `<string>` (The generated code will include this string.) (default: ``)<br>- `typeArg`: `<string>` (The type arg in generated code (must be the value-type, not pointer).) (default: ``) |

