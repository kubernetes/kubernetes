# Kubernetes Validation Tags Documentation

This document lists the supported validation tags and their related information.

## Tags Overview

| Tag | Usage | Description | Scopes |
|-----|-------------|-------------|----------|
| [`k8s:eachKey`](#k8seachkey) | k8s:eachKey=\<validation-tag\> | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:eachVal`](#k8seachval) | k8s:eachVal=\<validation-tag\> | Declares a validation for each value in a map or list. | anywhere |
| [`k8s:forbidden`](#k8sforbidden) | k8s:forbidden | Indicates that a field may not be specified. | struct fields |
| [`k8s:immutable`](#k8simmutable) | k8s:immutable | Indicates that a field may not be updated. | list values, map values, struct fields, type definitions |
| [`k8s:listMapKey`](#k8slistmapkey) | k8s:listMapKey=\<field-json-name\> | Declares a named sub-field of a list's value-type to be part of the list-map key. | anywhere |
| [`k8s:listType`](#k8slisttype) | k8s:listType=\<type\> | Declares a list field's semantic type. | anywhere |
| [`k8s:minimum`](#k8sminimum) | k8s:minimum=\<integer\> | Indicates that a numeric field has a minimum value. | anywhere |
| [`k8s:opaqueType`](#k8sopaquetype) | k8s:opaqueType | Indicates that any validations declared on the referenced type will be ignored. If a referenced type's package is not included in the generator's current flags, this tag must be set, or code generation will fail (preventing silent mistakes). If the validations should not be ignored, add the type's package to the generator using the --readonly-pkg flag. | struct fields |
| [`k8s:optional`](#k8soptional) | k8s:optional | Indicates that a field is optional to clients. | struct fields |
| [`k8s:required`](#k8srequired) | k8s:required | Indicates that a field must be specified by clients. | struct fields |
| [`k8s:subfield`](#k8ssubfield) | k8s:subfield(\<field-json-name\>)=\<validation-tag\> | Declares a validation for a subfield of a struct. | anywhere |
| [`k8s:validateError`](#k8svalidateerror) | k8s:validateError=\<string\> | Always fails code generation (useful for testing). | anywhere |
| [`k8s:validateFalse`](#k8svalidatefalse) | k8s:validateFalse(\<comma-separated-list-of-flag-string\>, \<string\>)=\<payload\> | Always fails validation (useful for testing). | anywhere |
| [`k8s:validateTrue`](#k8svalidatetrue) | k8s:validateTrue(\<comma-separated-list-of-flag-string\>, \<string\>)=\<payload\> | Always passes validation (useful for testing). | anywhere |

## Tag Details

### k8s:eachKey

#### Payloads

**Type:** tag | **Required:** true

| Description | Docs |
|-------------|------|
| \<validation-tag\> | The tag to evaluate for each value. |

### k8s:eachVal

#### Payloads

**Type:** tag | **Required:** true

| Description | Docs |
|-------------|------|
| \<validation-tag\> | The tag to evaluate for each value. |

### k8s:forbidden

### k8s:immutable

### k8s:listMapKey

#### Payloads

**Type:** string | **Required:** true

| Description | Docs |
|-------------|------|
| \<field-json-name\> | The name of the field. |

### k8s:listType

#### Payloads

**Type:** string | **Required:** true

| Description | Docs |
|-------------|------|
| \<type\> | map | atomic |

### k8s:minimum

#### Payloads

**Type:** int | **Required:** true

| Description | Docs |
|-------------|------|
| \<integer\> | This field must be greater than or equal to x. |

### k8s:opaqueType

### k8s:optional

### k8s:required

### k8s:subfield

The named subfield must be a direct field of the struct, or of an embedded struct.

#### Args

| Name | Description | Type | Required | Default | Docs |
|------|-------------|------|----------|---------|------|
| N/A | \<field-json-name\> | string | Yes | N/A | N/A |

#### Payloads

**Type:** tag | **Required:** true

| Description | Docs |
|-------------|------|
| \<validation-tag\> | The tag to evaluate for the subfield. |

### k8s:validateError

#### Payloads

**Type:** string | **Required:** false

| Description | Docs |
|-------------|------|
| \<string\> | This string will be included in the error message. |

### k8s:validateFalse

#### Args

| Name | Description | Type | Required | Default | Docs |
|------|-------------|------|----------|---------|------|
| flags | \<comma-separated-list-of-flag-string\> | string | No | N/A | values: ShortCircuit, NonError |
| typeArg | \<string\> | string | No | N/A | The type arg in generated code (must be the value-type, not pointer). |

#### Payloads

**Type:** string | **Required:** false

| Description | Docs |
|-------------|------|
| \<none\> | N/A |
| \<string\> | The generated code will include this string. |

### k8s:validateTrue

#### Args

| Name | Description | Type | Required | Default | Docs |
|------|-------------|------|----------|---------|------|
| flags | \<comma-separated-list-of-flag-string\> | string | No | N/A | values: ShortCircuit, NonError |
| typeArg | \<string\> | string | No | N/A | The type arg in generated code (must be the value-type, not pointer). |

#### Payloads

**Type:** string | **Required:** false

| Description | Docs |
|-------------|------|
| \<none\> | N/A |
| \<string\> | The generated code will include this string. |

