## Types

Several schema types are shared between the different parts of the specification and are defined below.

### AC Identifier Type

An AC Identifier Type is restricted to lowercase URI unreserved characters defined in [RFC3986](http://tools.ietf.org/html/rfc3986#section-2.3)
An AC Identifier Type cannot be an empty string and must begin and end with an alphanumeric character.
An AC Identifier Type will match the following [RE2](https://code.google.com/p/re2/wiki/Syntax) regular expression: `^[a-z0-9]+([-._~/][a-z0-9]+)*$`

Examples:

* database
* example.com/database
* example.com/ourapp
* example.com/~user/app_v1
* sub-domain.example.com/org/product/release

The AC Identifier Type is used as the primary key for a number of fields in the schemas that are in the global namespace, such as image names, image label keys and isolator names.
The schema validator will ensure that the keys conform to these constraints.

### AC Name Type

An AC Name Type is restricted to numeric and lowercase characters accepted by the DNS [RFC1123](http://tools.ietf.org/html/rfc1123#page-13) plus "-".
An AC Name Type cannot be an empty string and must begin and end with an alphanumeric character.
An AC Name Type will match the following [RE2](https://code.google.com/p/re2/wiki/Syntax) regular expression: `^[a-z0-9]+([-][a-z0-9]+)*$`

Examples:

* database
* product-database
* product-database-release

The AC Name Type is used as the primary key for a number of fields in the schemas below.
The schema validator will ensure that the keys conform to these constraints.


### AC Kind Type

An AC Kind cannot be an empty string and must be alphanumeric characters.
An AC Kind value matching defined kinds, will have defined compatibility.
There is no expected compatibility with undefined AC Kinds.

Defined Kinds:

* `ImageManifest`
* `PodManifest`


### AC Version Type

The App Container specification aims to follow semantic versioning and retain forward and backwards compatibility within major versions.
For example, if an implementation is compliant against version 1.0.1 of the spec, it is compatible with the complete 1.x series.

The version of the App Container specification and associated tooling is recorded in [VERSION](https://github.com/appc/spec/blob/master/VERSION), and is otherwise denoted in the [release version](https://github.com/appc/spec/releases) or git version control tag. 

An AC Version must reference a tagged version of the App Container specification, not exceeding the version of its greatest compliance.
An AC Version for [Image Manifest](aci.md#image-manifest-schema) and [Pod Manifest](pods.md#pod-manifest-schema) schemas must be compatible on major AC version series.
An AC Version cannot be an empty string and must be in [SemVer v2.0.0](http://semver.org/spec/v2.0.0.html) format.


### Image ID Type

An Image ID Type must be a string of the format "hash-value", where "hash" is the hash algorithm used and "value" is the hex encoded string of the digest.
Currently the only permitted hash algorithm is `sha512`.


### Isolator Type

An Isolator Type must be a JSON object with two required fields: "name" and "value".
"name" must be a string restricted to [AC Identifier](#ac-identifier-type) formatting.
"value" may be an arbitrary JSON value.


### Timestamps Type

Timestamps will be formatted to [RFC3339](https://tools.ietf.org/html/rfc3339).

Specifically including the "T" between the date and time components, per the `date-time` format of [RFC3339 Section 5.6](https://tools.ietf.org/html/rfc3339#section-5.6).
An example of this with shell script is:

```bash
$ date --rfc-3339=ns | tr " " "T"
2015-05-18T13:49:28.351729952-04:00
```

