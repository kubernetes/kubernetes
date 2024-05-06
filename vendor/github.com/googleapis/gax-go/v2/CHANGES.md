# Changelog

## [2.11.0](https://github.com/googleapis/gax-go/compare/v2.10.0...v2.11.0) (2023-06-13)


### Features

* **v2:** add GoVersion package variable ([#283](https://github.com/googleapis/gax-go/issues/283)) ([26553cc](https://github.com/googleapis/gax-go/commit/26553ccadb4016b189881f52e6c253b68bb3e3d5))


### Bug Fixes

* **v2:** handle space in non-devel go version ([#288](https://github.com/googleapis/gax-go/issues/288)) ([fd7bca0](https://github.com/googleapis/gax-go/commit/fd7bca029a1c5e63def8f0a5fd1ec3f725d92f75))

## [2.10.0](https://github.com/googleapis/gax-go/compare/v2.9.1...v2.10.0) (2023-05-30)


### Features

* update dependencies ([#280](https://github.com/googleapis/gax-go/issues/280)) ([4514281](https://github.com/googleapis/gax-go/commit/4514281058590f3637c36bfd49baa65c4d3cfb21))

## [2.9.1](https://github.com/googleapis/gax-go/compare/v2.9.0...v2.9.1) (2023-05-23)


### Bug Fixes

* **v2:** drop cloud lro test dep ([#276](https://github.com/googleapis/gax-go/issues/276)) ([c67eeba](https://github.com/googleapis/gax-go/commit/c67eeba0f10a3294b1d93c1b8fbe40211a55ae5f)), refs [#270](https://github.com/googleapis/gax-go/issues/270)

## [2.9.0](https://github.com/googleapis/gax-go/compare/v2.8.0...v2.9.0) (2023-05-22)


### Features

* **apierror:** add method to return HTTP status code conditionally ([#274](https://github.com/googleapis/gax-go/issues/274)) ([5874431](https://github.com/googleapis/gax-go/commit/587443169acd10f7f86d1989dc8aaf189e645e98)), refs [#229](https://github.com/googleapis/gax-go/issues/229)


### Documentation

* add ref to usage with clients ([#272](https://github.com/googleapis/gax-go/issues/272)) ([ea4d72d](https://github.com/googleapis/gax-go/commit/ea4d72d514beba4de450868b5fb028601a29164e)), refs [#228](https://github.com/googleapis/gax-go/issues/228)

## [2.8.0](https://github.com/googleapis/gax-go/compare/v2.7.1...v2.8.0) (2023-03-15)


### Features

* **v2:** add WithTimeout option ([#259](https://github.com/googleapis/gax-go/issues/259)) ([9a8da43](https://github.com/googleapis/gax-go/commit/9a8da43693002448b1e8758023699387481866d1))

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
