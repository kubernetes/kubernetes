# Changelog

## [2.7.1](https://github.com/googleapis/gax-go/compare/v2.7.0...v2.7.1) (2023-03-06)


### Bug Fixes

* **v2/apierror:** return Unknown GRPCStatus when err source is HTTP ([#260](https://github.com/googleapis/gax-go/issues/260)) ([043b734](https://github.com/googleapis/gax-go/commit/043b73437a240a91229207fb3ee52a9935a36f23)), refs [#254](https://github.com/googleapis/gax-go/issues/254)

## [2.7.0](https://github.com/googleapis/gax-go/compare/v2.6.0...v2.7.0) (2022-11-02)


### Features

* update google.golang.org/api to latest ([#240](https://github.com/googleapis/gax-go/issues/240)) ([f690a02](https://github.com/googleapis/gax-go/commit/f690a02c806a2903bdee943ede3a58e3a331ebd6))
* **v2/apierror:** add apierror.FromWrappingError ([#238](https://github.com/googleapis/gax-go/issues/238)) ([9dbd96d](https://github.com/googleapis/gax-go/commit/9dbd96d59b9d54ceb7c025513aa8c1a9d727382f))

## [2.6.0](https://github.com/googleapis/gax-go/compare/v2.5.1...v2.6.0) (2022-10-13)


### Features

* **v2:** copy DetermineContentType functionality ([#230](https://github.com/googleapis/gax-go/issues/230)) ([2c52a70](https://github.com/googleapis/gax-go/commit/2c52a70bae965397f740ed27d46aabe89ff249b3))

## [2.5.1](https://github.com/googleapis/gax-go/compare/v2.5.0...v2.5.1) (2022-08-04)


### Bug Fixes

* **v2:** resolve bad genproto pseudoversion in go.mod ([#218](https://github.com/googleapis/gax-go/issues/218)) ([1379b27](https://github.com/googleapis/gax-go/commit/1379b27e9846d959f7e1163b9ef298b3c92c8d23))

## [2.5.0](https://github.com/googleapis/gax-go/compare/v2.4.0...v2.5.0) (2022-08-04)


### Features

* add ExtractProtoMessage to apierror ([#213](https://github.com/googleapis/gax-go/issues/213)) ([a6ce70c](https://github.com/googleapis/gax-go/commit/a6ce70c725c890533a9de6272d3b5ba2e336d6bb))

## [2.4.0](https://github.com/googleapis/gax-go/compare/v2.3.0...v2.4.0) (2022-05-09)


### Features

* **v2:** add OnHTTPCodes CallOption ([#188](https://github.com/googleapis/gax-go/issues/188)) ([ba7c534](https://github.com/googleapis/gax-go/commit/ba7c5348363ab6c33e1cee3c03c0be68a46ca07c))


### Bug Fixes

* **v2/apierror:** use errors.As in FromError ([#189](https://github.com/googleapis/gax-go/issues/189)) ([f30f05b](https://github.com/googleapis/gax-go/commit/f30f05be583828f4c09cca4091333ea88ff8d79e))


### Miscellaneous Chores

* **v2:** bump release-please processing ([#192](https://github.com/googleapis/gax-go/issues/192)) ([56172f9](https://github.com/googleapis/gax-go/commit/56172f971d1141d7687edaac053ad3470af76719))
