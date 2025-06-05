# Kubernetes Validation Tags Documentation

This document lists the supported validation tags and their related information.

## Tags Overview

| Tag | Usage | Args | Description | Scopes |
|-----|-------------|------|-------------|----------|
| [`k8s:eachKey`](#k8seachkey) | k8s:eachKey=\<validation-tag\> | N/A | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:eachVal`](#k8seachval) | k8s:eachVal=\<validation-tag\> | N/A | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:enum`](#k8senum) | k8s:enum | N/A | Indicates that a string type is an enum. All const values of this type are considered values in the enum. | type definitions |
| [`k8s:forbidden`](#k8sforbidden) | k8s:forbidden | N/A | Indicates that a field may not be specified. | struct fields |
| [`k8s:format`](#k8sformat) | k8s:format=\<payload\> | N/A | Indicates that a string field has a particular format. | anywhere |
| [`k8s:ifOptionDisabled`](#k8sifoptiondisabled) | k8s:ifOptionDisabled(\<option\>)=\<validation-tag\> | <option> | Declares a validation that only applies when an option is disabled. | anywhere |
| [`k8s:ifOptionEnabled`](#k8sifoptionenabled) | k8s:ifOptionEnabled(\<option\>)=\<validation-tag\> | <option> | Declares a validation that only applies when an option is enabled. | anywhere |
| [`k8s:immutable`](#k8simmutable) | k8s:immutable | N/A | Indicates that a field may not be updated. | list values, map values, struct fields, type definitions |
| [`k8s:listMapKey`](#k8slistmapkey) | k8s:listMapKey=\<field-json-name\> | N/A | Declares a named sub-field of a list's value-type to be part of the list-map key. | anywhere |
| [`k8s:listType`](#k8slisttype) | k8s:listType=\<type\> | N/A | Declares a list field's semantic type. | anywhere |
| [`k8s:maxItems`](#k8smaxitems) | k8s:maxItems=\<non-negative integer\> | N/A | Indicates that a list field has a limit on its size. | list values, map values, struct fields, type definitions |
| [`k8s:maxLength`](#k8smaxlength) | k8s:maxLength=\<non-negative integer\> | N/A | Indicates that a string field has a limit on its length. | anywhere |
| [`k8s:minimum`](#k8sminimum) | k8s:minimum=\<integer\> | N/A | Indicates that a numeric field has a minimum value. | anywhere |
| [`k8s:opaqueType`](#k8sopaquetype) | k8s:opaqueType | N/A | Indicates that any validations declared on the referenced type will be ignored. If a referenced type's package is not included in the generator's current flags, this tag must be set, or code generation will fail (preventing silent mistakes). If the validations should not be ignored, add the type's package to the generator using the --readonly-pkg flag. | struct fields |
| [`k8s:optional`](#k8soptional) | k8s:optional | N/A | Indicates that a field is optional to clients. | struct fields |
| [`k8s:required`](#k8srequired) | k8s:required | N/A | Indicates that a field must be specified by clients. | struct fields |
| [`k8s:subfield`](#k8ssubfield) | k8s:subfield(\<field-json-name\>)=\<validation-tag\> | <field-json-name> | Declares a validation for a subfield of a struct. | anywhere |
| [`k8s:unionDiscriminator`](#k8suniondiscriminator) | k8s:unionDiscriminator(\<string\>) | <string> | Indicates that this field is the discriminator for a union. | struct fields |
| [`k8s:unionMember`](#k8sunionmember) | k8s:unionMember(\<string\>, \<string\>) | <string>,<string> | Indicates that this field is a member of a union. | struct fields |
| [`k8s:validateError`](#k8svalidateerror) | k8s:validateError=\<string\> | N/A | Always fails code generation (useful for testing). | anywhere |
| [`k8s:validateFalse`](#k8svalidatefalse) | k8s:validateFalse(\<comma-separated-list-of-flag-string\>, \<string\>)=\<payload\> | <comma-separated-list-of-flag-string>,<string> | Always fails validation (useful for testing). | anywhere |
| [`k8s:validateTrue`](#k8svalidatetrue) | k8s:validateTrue(\<comma-separated-list-of-flag-string\>, \<string\>)=\<payload\> | <comma-separated-list-of-flag-string>,<string> | Always passes validation (useful for testing). | anywhere |

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

### k8s:enum

#### Payloads

null

### k8s:forbidden

#### Payloads

null

### k8s:format

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **k8s-ip-sloppy** | This field holds an IPv4 or IPv6 address value. IPv4 octets may have leading zeros. | None |
| **dns-label** | This field holds a DNS label value. | None |

### k8s:ifOptionDisabled

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<validation-tag\>** | This validation tag will be evaluated only if the validation option is disabled. | None |

### k8s:ifOptionEnabled

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<validation-tag\>** | This validation tag will be evaluated only if the validation option is enabled. | None |

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
| **\<type\>** | atomic | map | set | None |

### k8s:maxItems

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<non-negative integer\>** | This field must be no more than X items long. | None |

### k8s:maxLength

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<non-negative integer\>** | This field must be no more than X characters long. | None |

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

### k8s:unionDiscriminator

#### Payloads

null

### k8s:unionMember

#### Payloads

null

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
| **\<string\>** | The generated code will include this string. | None |

### k8s:validateTrue

#### Payloads

| Description | Docs | Schema |
|-------------|------|---------|
| **\<none\>** |  | None |
| **\<string\>** | The generated code will include this string. | None |

