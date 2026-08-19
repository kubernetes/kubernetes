# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [2.266.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.265.0-alpha.0...v2.266.0-alpha.0) (2026-08-19)


### ⚠ BREAKING CHANGES

* **glue-alpha:** `DataQualityRulesetProps.rulesetDqdl: string` is replaced by
`dqdl: Dqdl`. Build it with `Dqdl.fromString('Rules = [ ... ]')`.
* **glue-alpha:** `s3Encryption`, `cloudWatchEncryption`, and `jobBookmarksEncryption` are no longer object literals. Use `S3Encryption.s3Managed()` / `S3Encryption.kms(key?)`, `CloudWatchEncryption.kms(key?)`, and `JobBookmarksEncryption.clientSideKms(key?)`. The `CloudWatchEncryptionMode` and `JobBookmarksEncryptionMode` enums are removed (their mode is now implicit); `S3EncryptionMode` is retained.
* **glue-alpha:** this is a corrective breaking change. Apps that leaned on the bug, and did things like `InputFormat x = OutputFormat.AVRO;` will get a compilation error in other jsii languages. The intended usage, on the other hand, was broken before and works now.
* **glue-alpha:** `workerType` and `numberOfWorkers` are no longer top-level job props. For Spark jobs, pass them together via `workerConfiguration: { workerType, numberOfWorkers }`. `PythonShellJob` no longer accepts them (it is sized by `maxCapacity`). `RayJob` no longer accepts `workerType` (it is fixed to `Z.2X`).
* **glue-alpha:** `SparkJobProps.enableMetrics` removed, which will cause a compilation error for any app using it. But there is no behavior change, since this is a dead prop.
* **glue-alpha:** a differing/tokenized `has_encrypted_data` supplied via `parameters` now throws.

### Features

* **glue-alpha:** add warning for maxRetries when job run queuing is enabled ([#38575](https://github.com/aws/aws-cdk/issues/38575)) ([09ae11e](https://github.com/aws/aws-cdk/commit/09ae11e81d44bd58059ac337d3c418254a6955a7))
* **glue-alpha:** model SecurityConfiguration encryption as factory subtypes ([#38586](https://github.com/aws/aws-cdk/issues/38586)) ([e37e7a6](https://github.com/aws/aws-cdk/commit/e37e7a6ea3d525ca0b4048093416eac059272cdf))
* **glue-alpha:** new `hasEncryptedData` property ([#38511](https://github.com/aws/aws-cdk/issues/38511)) ([c977e36](https://github.com/aws/aws-cdk/commit/c977e36c3aa43c0be3d043aef80775d9748477a6))
* **glue-alpha:** pair workerType and numberOfWorkers into a required workerConfiguration ([#38576](https://github.com/aws/aws-cdk/issues/38576)) ([5f3b1b6](https://github.com/aws/aws-cdk/commit/5f3b1b6f3aa700d73fbc10c956ae53809af99047))
* **glue-alpha:** wrap DataQualityRuleset DQDL in a typed value object ([#38587](https://github.com/aws/aws-cdk/issues/38587)) ([fd01268](https://github.com/aws/aws-cdk/commit/fd012680b1cff1905ad114082fc612ad7fb8992e))


### Bug Fixes

* **glue-alpha:** correct OutputFormat.AVRO/ORC type and tidy GA-hygiene items ([#38593](https://github.com/aws/aws-cdk/issues/38593)) ([35aefaf](https://github.com/aws/aws-cdk/commit/35aefaf5a72b89eda630de8cae38a1ed9c4c95ce))
* **glue-alpha:** omit comments from struct type strings ([#38008](https://github.com/aws/aws-cdk/issues/38008)) ([5f16a86](https://github.com/aws/aws-cdk/commit/5f16a86dc8edefdb223260f8202acd272d43e643)), closes [#26935](https://github.com/aws/aws-cdk/issues/26935)
* **mediapackagev2-alpha:** add GetChannel grant on MediaPackageV2 ([#38582](https://github.com/aws/aws-cdk/issues/38582)) ([74d922c](https://github.com/aws/aws-cdk/commit/74d922c4bf22fde05465f61c5dd19291b9db608f)), closes [#38581](https://github.com/aws/aws-cdk/issues/38581)

## [2.265.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.264.0-alpha.0...v2.265.0-alpha.0) (2026-08-13)


### ⚠ BREAKING CHANGES

* **glue-alpha:** PySparkFlexEtlJob and ScalaSparkFlexEtlJob now default to `GlueVersion.V5_0` instead of `V3_0`. Set `glueVersion` explicitly to keep the previous behavior.
* **glue-alpha:** removal policy of existing `Database` resources will change to `RETAIN`.

### Features

* **glue-alpha:** default Flex jobs to Glue 5.0 ([#38543](https://github.com/aws/aws-cdk/issues/38543)) ([36d4976](https://github.com/aws/aws-cdk/commit/36d4976e9d6575debcf68b5e666a5920d53f69ab))
* **glue-alpha:** removal policy is RETAIN for `Database` by default ([#38535](https://github.com/aws/aws-cdk/issues/38535)) ([9669928](https://github.com/aws/aws-cdk/commit/9669928f7837798e3e7169bea55c3907ce16f440))
* **glue-alpha:** warn on plaintext secrets  ([#38538](https://github.com/aws/aws-cdk/issues/38538)) ([10bf0b5](https://github.com/aws/aws-cdk/commit/10bf0b596a06399271712987b0857df513b69ad5))


### Bug Fixes

* **glue-alpha:** warn when S3Table grants cover a shared bucket ([#38542](https://github.com/aws/aws-cdk/issues/38542)) ([d81d2d6](https://github.com/aws/aws-cdk/commit/d81d2d63613b5118ae3c7cc4c82b26aac2fd69f4))
* **mediaconnect-alpha:** update validation to allow adding an underscore to name ([#38539](https://github.com/aws/aws-cdk/issues/38539)) ([7c85994](https://github.com/aws/aws-cdk/commit/7c85994cd1ee59204341ad7510124eafabe1eb6f))

## [2.264.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.263.0-alpha.0...v2.264.0-alpha.0) (2026-08-10)


### ⚠ BREAKING CHANGES

* **glue-alpha:** `IDatabase.catalogArn` and `IDatabase.catalogId` were removed in factor of a type
safe `ICatalog`, which has  `catalogArn` and `catalogId`. Consumers and implementations were updated
accordingly.

### Features

* **glue-alpha:** new `Catalog` L2 ([#38443](https://github.com/aws/aws-cdk/issues/38443)) ([6a8ba8e](https://github.com/aws/aws-cdk/commit/6a8ba8e6597dd73ab2e92ac2dbb688006bbb481b))
* **glue-alpha:** strengthen data encryption for `S3Table` ([#38501](https://github.com/aws/aws-cdk/issues/38501)) ([eb81d5e](https://github.com/aws/aws-cdk/commit/eb81d5e68c429d33417a8609ddc1204e39102687)), closes [/docs.aws.amazon.com/securityhub/latest/userguide/s3-controls.html#s3-5](https://github.com/aws//docs.aws.amazon.com/securityhub/latest/userguide/s3-controls.html/issues/s3-5)


### Bug Fixes

* **glue-alpha:** enable key rotation for security configuration encryption ([#38512](https://github.com/aws/aws-cdk/issues/38512)) ([18560e0](https://github.com/aws/aws-cdk/commit/18560e001cffa882ac498b363fc3edc047fea118))
* **mediaconnect-alpha:** addOutput options and simplified VPC interface referencing ([#38515](https://github.com/aws/aws-cdk/issues/38515)) ([3c71466](https://github.com/aws/aws-cdk/commit/3c7146693547b05e048ac93f60bf6ace845270fc)), closes [#38517](https://github.com/aws/aws-cdk/issues/38517)

## [2.263.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.262.2-alpha.0...v2.263.0-alpha.0) (2026-07-31)

### ⚠ BREAKING CHANGES

* **mediaconnect:** `removalPolicy` prop removed from `FlowProps`, `GatewayProps`, and `BridgeProps`. These resources now follow CloudFormation's default deletion behaviour (Delete).

### Bug Fixes

* **glue-alpha:** race condition when creating mutiple indices ([#38016](https://github.com/aws/aws-cdk/issues/38016)) ([18b1947](https://github.com/aws/aws-cdk/commit/18b1947a9db0e695ec85888faf24939261ffbe30)), closes [#24813](https://github.com/aws/aws-cdk/issues/24813) [#34](https://github.com/aws/aws-cdk/issues/34) [#34](https://github.com/aws/aws-cdk/issues/34)
* **mediaconnect:** jsdoc typo ([#38434](https://github.com/aws/aws-cdk/issues/38434)) ([7fe6d3d](https://github.com/aws/aws-cdk/commit/7fe6d3df03720b381bb989f289855557f2cd47ac))
* **mediaconnect:** remove default removal policy ([#38437](https://github.com/aws/aws-cdk/issues/38437)) ([238193d](https://github.com/aws/aws-cdk/commit/238193de50910ac1da8fb9f31f01776863998989))

## [2.262.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.262.1-alpha.0...v2.262.2-alpha.0) (2026-07-29)

## [2.262.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.262.0-alpha.0...v2.262.1-alpha.0) (2026-07-23)

## [2.262.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.261.0-alpha.0...v2.262.0-alpha.0) (2026-07-22)


### Bug Fixes

* **redshift-alpha:** quote identifiers and escape literals containing special characters ([#38269](https://github.com/aws/aws-cdk/issues/38269)) ([3cb6844](https://github.com/aws/aws-cdk/commit/3cb684493ba4a86c8f96b3c57bb1df3541db87b7))

## [2.261.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.260.0-alpha.0...v2.261.0-alpha.0) (2026-07-02)


### Features

* **metrics-facade-alpha:** generate metrics facades ([#37334](https://github.com/aws/aws-cdk/issues/37334)) ([aa9beb0](https://github.com/aws/aws-cdk/commit/aa9beb08e6740c16a682a93921959f2574d7ca99))

## [2.260.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.259.0-alpha.0...v2.260.0-alpha.0) (2026-06-16)

## [2.259.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.258.1-alpha.0...v2.259.0-alpha.0) (2026-06-11)

## [2.258.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.258.0-alpha.0...v2.258.1-alpha.0) (2026-06-08)

## [2.258.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.257.0-alpha.0...v2.258.0-alpha.0) (2026-06-04)


### Features

* **integ-tests-alpha:** add option to set the provider log level ([#38005](https://github.com/aws/aws-cdk/issues/38005)) ([c634a79](https://github.com/aws/aws-cdk/commit/c634a795de080df21b86fad191d05dfde884eb4e))


### Bug Fixes

* **custom-resource-handlers:** deterministic asset hashes for generated lambdas ([#37634](https://github.com/aws/aws-cdk/issues/37634)) ([6c3d5bc](https://github.com/aws/aws-cdk/commit/6c3d5bc36e19b8834319578fbd8525615a47a4b9)), closes [#34307](https://github.com/aws/aws-cdk/issues/34307)
* **glue-alpha:** deprecate Ray Jobs ([#38055](https://github.com/aws/aws-cdk/issues/38055)) ([3fa428b](https://github.com/aws/aws-cdk/commit/3fa428b9b24d286841940cfff5200d16817196bf))
* **glue-alpha:** restore notifyDelayAfter to PySpark and Scala Spark ETL jobs ([#37815](https://github.com/aws/aws-cdk/issues/37815)) ([05be88a](https://github.com/aws/aws-cdk/commit/05be88aa324635108076bdc648a4ee940d22a386)), closes [#33839](https://github.com/aws/aws-cdk/issues/33839)
* **integ-tests-alpha:** assertion failures print too much unnecessary information ([#37974](https://github.com/aws/aws-cdk/issues/37974)) ([bc0de1d](https://github.com/aws/aws-cdk/commit/bc0de1dacc59b95a2e89f9c3ec96589e98b35ed2))
* **mediapackagev2-alpha:** cdnAuth on OriginEndpoint now generates the required policy ([#38013](https://github.com/aws/aws-cdk/issues/38013)) ([1d56b46](https://github.com/aws/aws-cdk/commit/1d56b46abd477f188021d32d77551be1377765d0))

## [2.257.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.256.1-alpha.0...v2.257.0-alpha.0) (2026-05-21)

## [2.256.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.256.0-alpha.0...v2.256.1-alpha.0) (2026-05-20)

## [2.256.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.255.0-alpha.0...v2.256.0-alpha.0) (2026-05-19)

## [2.255.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.254.0-alpha.0...v2.255.0-alpha.0) (2026-05-18)

### Features

* **bedrock-agentcore-alpha:** Graduation of the library to stable. The **Policy** submodule is the only submodule that remains in alpha. All other constructs have graduated to stable in `aws-cdk-lib/aws-bedrockagentcore` and we recommend migrating to the stable versions  ([#37876](https://github.com/aws/aws-cdk/issues/37876)) ([00cf601](https://github.com/aws/aws-cdk/commit/00cf6015f30755653de7a541f79c944d4a68f423))


## [2.254.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.253.1-alpha.0...v2.254.0-alpha.0) (2026-05-13)


### Features

* **bedrock-agentcore-alpha:** add tags support to Evaluator and OnlineEvaluationConfig ([#37804](https://github.com/aws/aws-cdk/issues/37804)) ([adbf88f](https://github.com/aws/aws-cdk/commit/adbf88faeb4d2b762563389aea160cfda496f200))
* **bedrock-agentcore-alpha:** add identity L2 constructs ([#37610](https://github.com/aws/aws-cdk/issues/37610)) ([67c3af2](https://github.com/aws/aws-cdk/commit/67c3af260cedb8e610dff36f829afb36114dd93a))
* **mediapackagev2-alpha:** add OAC integration between CloudFront and MediaPackageV2 ([#37701](https://github.com/aws/aws-cdk/issues/37701)) ([654f59c](https://github.com/aws/aws-cdk/commit/654f59c559f6eb2b5240e7885372ac82ea05a996))


### Bug Fixes

* **bedrock-agentcore-alpha:** fix cedar policy bug ([#37782](https://github.com/aws/aws-cdk/issues/37782)) ([e678d5c](https://github.com/aws/aws-cdk/commit/e678d5cc251f00a0a67bc308b562a3fddca6a73c)), closes [#37828](https://github.com/aws/aws-cdk/issues/37828)
* **custom-resource-handlers:** deployment fails when parameter already exists ([#37852](https://github.com/aws/aws-cdk/issues/37852)) ([025c38c](https://github.com/aws/aws-cdk/commit/025c38ca989a234189b411df011a560a003107b3))

## [2.253.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.253.0-alpha.0...v2.253.1-alpha.0) (2026-05-08)

## [2.253.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.252.0-alpha.0...v2.253.0-alpha.0) (2026-05-06)


### Features

* **bedrock-agentcore-alpha:** add OnlineEvaluationConfig and Evaluator L2 constructs ([#37615](https://github.com/aws/aws-cdk/issues/37615)) ([c13de04](https://github.com/aws/aws-cdk/commit/c13de04223e32272b0c6c6dd4e2fca8e300fafa8)), closes [#37614](https://github.com/aws/aws-cdk/issues/37614)
* **glue-alpha:** add extraPythonFiles support to PythonShellJob ([#37130](https://github.com/aws/aws-cdk/issues/37130)) ([c9c6f9c](https://github.com/aws/aws-cdk/commit/c9c6f9c1b7c12722d18a45bf8a02c09672f8720d)), closes [#34448](https://github.com/aws/aws-cdk/issues/34448)


### Bug Fixes

* **bedrock-agentcore-alpha:** self-managed memory strategy validation throws on unresolved tokens ([#37691](https://github.com/aws/aws-cdk/issues/37691)) ([7956537](https://github.com/aws/aws-cdk/commit/79565376cdb642d821625fe10ae5916e7d2e64fe)), closes [#37197](https://github.com/aws/aws-cdk/issues/37197)

## [2.252.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.251.0-alpha.0...v2.252.0-alpha.0) (2026-04-29)

## [2.251.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.250.0-alpha.0...v2.251.0-alpha.0) (2026-04-24)


### Features

* **bedrock-agentcore-alpha:** add L2 constructs for policy and policy engine  ([#37238](https://github.com/aws/aws-cdk/issues/37238)) ([1e89e7e](https://github.com/aws/aws-cdk/commit/1e89e7e921a9946cb9c23f967c6b7a33a6048de4))
* **bedrock-agentcore-alpha:** add observability configuration for Runtime ([#36689](https://github.com/aws/aws-cdk/issues/36689)) ([34b43aa](https://github.com/aws/aws-cdk/commit/34b43aabe2c3a946ba286812b402ce946222d820)), closes [#36596](https://github.com/aws/aws-cdk/issues/36596)
* **bedrock-agentcore-alpha:** support No Authorization for AgentCore Gateway ([#36610](https://github.com/aws/aws-cdk/issues/36610)) ([f20bd8e](https://github.com/aws/aws-cdk/commit/f20bd8e43700877f7166cdac3cd994876963bc67))
* **dsql-alpha:** initial L2 construct ([#34599](https://github.com/aws/aws-cdk/issues/34599)) ([be1a458](https://github.com/aws/aws-cdk/commit/be1a45861a5138b6e397cf076e39dfe0a18d4e99)), closes [#34593](https://github.com/aws/aws-cdk/issues/34593)

## [2.250.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.249.0-alpha.0...v2.250.0-alpha.0) (2026-04-14)

## [2.249.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.248.0-alpha.0...v2.249.0-alpha.0) (2026-04-10)

## [2.248.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.247.0-alpha.0...v2.248.0-alpha.0) (2026-04-02)

## [2.247.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.246.0-alpha.0...v2.247.0-alpha.0) (2026-04-02)


### Features

* **mediapackagev2-alpha:** new L2 construct ([#37279](https://github.com/aws/aws-cdk/issues/37279)) ([7debfb9](https://github.com/aws/aws-cdk/commit/7debfb9c5e807fac5df6e9e0ea3097d72325ffbc))

## [2.246.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.245.0-alpha.0...v2.246.0-alpha.0) (2026-03-31)

## [2.245.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.244.0-alpha.0...v2.245.0-alpha.0) (2026-03-27)


### Features

* **s3tables-alpha:** add support for partition spec, sort order, and table properties ([#36811](https://github.com/aws/aws-cdk/issues/36811)) ([2696cd1](https://github.com/aws/aws-cdk/commit/2696cd16e8e2edc8d40f1443b9c87eb6171e5d1f))
* **s3tables-alpha:** add metrics configuration support for TableBucket ([#37275](https://github.com/aws/aws-cdk/issues/37275)) ([e8786f5](https://github.com/aws/aws-cdk/commit/e8786f5d782d906971f933a2d6d432309d5384d7))
* **s3tables-alpha:** implement ITaggableV2 on TableBucket and Table L2 constructs ([#37277](https://github.com/aws/aws-cdk/issues/37277)) ([69c8944](https://github.com/aws/aws-cdk/commit/69c8944ea3f4abf0f4218af2fc42c8e862e8cad3)), closes [#33054](https://github.com/aws/aws-cdk/issues/33054)

## [2.244.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.243.0-alpha.0...v2.244.0-alpha.0) (2026-03-19)


### Bug Fixes

* **kinesisanalytics-flink-alpha:** mark deprecated flink runtimes as deprecated ([#37155](https://github.com/aws/aws-cdk/issues/37155)) ([0a89447](https://github.com/aws/aws-cdk/commit/0a894472650bb1a2c41050ae2b00581fb937c924))

## [2.243.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.242.0-alpha.0...v2.243.0-alpha.0) (2026-03-11)

## [2.242.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.241.0-alpha.0...v2.242.0-alpha.0) (2026-03-10)


### Features

* **mixins-preview:** allow passing resource objects into properties in CFN Property mixins ([#37148](https://github.com/aws/aws-cdk/issues/37148)) ([f238629](https://github.com/aws/aws-cdk/commit/f2386291a50961660135b6d13b576a3744fa5ecf))
* **mixins-preview:** generate EventBridge pattern for all events ([#37081](https://github.com/aws/aws-cdk/issues/37081)) ([f30e836](https://github.com/aws/aws-cdk/commit/f30e8360112c724ce386f26d7d2bf10d6a58e479))
* **mixins-preview:** support custom merge strategies via IMergeStrategy ([#37170](https://github.com/aws/aws-cdk/issues/37170)) ([0dec011](https://github.com/aws/aws-cdk/commit/0dec0113c45f5808e2afd45ac5be1d044e577a4b))

## [2.241.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.240.0-alpha.0...v2.241.0-alpha.0) (2026-03-02)


### Features

* **mixins-preview:** add `recordFields` and `outputFormat` to Vended Logs Mixin ([#37042](https://github.com/aws/aws-cdk/issues/37042)) ([dd94c31](https://github.com/aws/aws-cdk/commit/dd94c312ae77cd9b51cbf6d544c85a2af6a7cdc8))
* **mixins-preview:** cross account delivery destinations ([#36827](https://github.com/aws/aws-cdk/issues/36827)) ([a759eb6](https://github.com/aws/aws-cdk/commit/a759eb69d560ff039d09d62e91627bb267a664e5))

## [2.240.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.239.0-alpha.0...v2.240.0-alpha.0) (2026-02-23)

## [2.239.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.238.0-alpha.0...v2.239.0-alpha.0) (2026-02-19)

### ⚠ BREAKING CHANGES

redshift-alpha: update default node type from `DC2_LARGE` to `RA3_LARGE`

### Features

* **bedrock-agentcore-alpha:** add `fromCodeAsset` method to create runtime artifact with local code assets ([#36472](https://github.com/aws/aws-cdk/issues/36472)) ([c5a87e6](https://github.com/aws/aws-cdk/commit/c5a87e6682a3668de849d4c7a0913fdee3eb170c)), closes [#36473](https://github.com/aws/aws-cdk/issues/36473)
* **bedrock-agentcore-alpha:** added new target type (api gateway) in agentcore gateway target. ([#36841](https://github.com/aws/aws-cdk/issues/36841)) ([0842754](https://github.com/aws/aws-cdk/commit/0842754ec56412a9b22f2e92f5aea7c86129ec52)), closes [#36817](https://github.com/aws/aws-cdk/issues/36817)
* **mixins-preview:** add ECS ClusterSettingsMixin ([#36796](https://github.com/aws/aws-cdk/issues/36796)) ([b8ab5be](https://github.com/aws/aws-cdk/commit/b8ab5be8f2e0733433a55dd48b26e7f56f6e0393))
* **mixins-preview:** add s3 bucket mixin for publicAccessBlock ([#36905](https://github.com/aws/aws-cdk/issues/36905)) ([feed4b2](https://github.com/aws/aws-cdk/commit/feed4b2690bd481e464dd3ececa4cba0997a03db))
* **mixins-preview:** send Vended Logs to pre-created DeliveryDestination using `toDestination()` ([#36896](https://github.com/aws/aws-cdk/issues/36896)) ([48f1fe6](https://github.com/aws/aws-cdk/commit/48f1fe6aa86473a25ffdcf53cfecb5e1169b54db))


### Bug Fixes

* **redshift-alpha:** update default node type from `DC2_LARGE` to `RA3_LARGE` ([#36516](https://github.com/aws/aws-cdk/issues/36516)) ([ea19e5c](https://github.com/aws/aws-cdk/commit/ea19e5cde2e64d5cdcdfa3af41764542e77e221c)), closes [#36416](https://github.com/aws/aws-cdk/issues/36416)

## [2.238.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.237.1-alpha.0...v2.238.0-alpha.0) (2026-02-09)


### Features

* **eks-v2-alpha:** add support for bootstrapSelfManagedAddons ([#36740](https://github.com/aws/aws-cdk/issues/36740)) ([1ffe38d](https://github.com/aws/aws-cdk/commit/1ffe38dc950a096cb5e1c1ee20f2c49899dc0a23))
* **eks-v2-alpha:** add support for EKS hybrid nodes ([#36749](https://github.com/aws/aws-cdk/issues/36749)) ([48ace56](https://github.com/aws/aws-cdk/commit/48ace56d82537630fc339cb41962473a97375aea))


### Bug Fixes

* **eks-v2-alpha:** ensure kubectl provider and handler functions use the same vpc configuration  ([#36735](https://github.com/aws/aws-cdk/issues/36735)) ([4e02f08](https://github.com/aws/aws-cdk/commit/4e02f0896069105dae83c46f19f1b346a546ad57)), closes [#34878](https://github.com/aws/aws-cdk/issues/34878) [#34877](https://github.com/aws/aws-cdk/issues/34877)
* **ivs-alpha:** add region constraints to integration tests ([#36851](https://github.com/aws/aws-cdk/issues/36851)) ([d55fec4](https://github.com/aws/aws-cdk/commit/d55fec42357410b8263b814b931daf5dccc5c5e3))
* **mixins-preview:** apply mixins in order ([#36847](https://github.com/aws/aws-cdk/issues/36847)) ([726060c](https://github.com/aws/aws-cdk/commit/726060c0ea9f57de4c6e13c1f50c330e4fc2608e))
* **mixins-preview:** apply mixins in order in `MixinApplicator` ([#36877](https://github.com/aws/aws-cdk/issues/36877)) ([09db1c9](https://github.com/aws/aws-cdk/commit/09db1c99710c9f8e91774e767de93fff1a0d2650)), closes [#36847](https://github.com/aws/aws-cdk/issues/36847)

## [2.237.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.237.0-alpha.0...v2.237.1-alpha.0) (2026-02-03)

## [2.237.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.236.0-alpha.0...v2.237.0-alpha.0) (2026-02-02)


### Features

* **bedrock-agentcore-alpha:** add support for custom claims and scopes to runtime/gateway authorizers ([#36810](https://github.com/aws/aws-cdk/issues/36810)) ([a3abcd0](https://github.com/aws/aws-cdk/commit/a3abcd010648e83bed273dff8f581291b5b3c902))
* **eks-v2-alpha:** pass additional helm chart values to aws-load-balancer-controller ([#36754](https://github.com/aws/aws-cdk/issues/36754)) ([cf61814](https://github.com/aws/aws-cdk/commit/cf61814ac58448ddd70682e45c07dd8ca58c4fd1)), closes [/github.com/kubernetes-sigs/aws-load-balancer-controller/blob/main/helm/aws-load-balancer-controller/values.yaml#L199](https://github.com/aws//github.com/kubernetes-sigs/aws-load-balancer-controller/blob/main/helm/aws-load-balancer-controller/values.yaml/issues/L199)
* **mixins-preview:** align Mixins API with latest RFC proposal ([#36825](https://github.com/aws/aws-cdk/issues/36825)) ([82c2fdb](https://github.com/aws/aws-cdk/commit/82c2fdb246557fa4804e2dc88ce16c28db52956c))
* **mixins-preview:** handle destination bucket with KMS keys ([#36776](https://github.com/aws/aws-cdk/issues/36776)) ([950401f](https://github.com/aws/aws-cdk/commit/950401f405751a7634927af0d7667c97ddddd73d))


### Bug Fixes

* **bedrock-agentcore-alpha:** construct ID collision when multiple schemas are set ([#36565](https://github.com/aws/aws-cdk/issues/36565)) ([9ebfb62](https://github.com/aws/aws-cdk/commit/9ebfb62d6c6599bee2bf477cdc6b4b6da0a4030a)), closes [#36559](https://github.com/aws/aws-cdk/issues/36559)

## [2.236.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.235.1-alpha.0...v2.236.0-alpha.0) (2026-01-23)


### Features

* **bedrock-agentcore-alpha:** added episodic memory strategy ([#36591](https://github.com/aws/aws-cdk/issues/36591)) ([21dcfc6](https://github.com/aws/aws-cdk/commit/21dcfc6807a3876e2275bdac6f1e4f7564a66100))
* **bedrock-agentcore-alpha:** added gateway interceptors ([#36604](https://github.com/aws/aws-cdk/issues/36604)) ([ba8aa48](https://github.com/aws/aws-cdk/commit/ba8aa48a33b1e008194d6b6b13d10c41019f56b4))
* **bedrock-agentcore-alpha:** make physical name properties optional for AgentCore resources ([#36354](https://github.com/aws/aws-cdk/issues/36354)) ([5137d81](https://github.com/aws/aws-cdk/commit/5137d811a92eb63f52d2bfa0713a660f5476839e)), closes [#36341](https://github.com/aws/aws-cdk/issues/36341)
* **mixins-preview:** expose `BucketPolicyStatementsMixin` publicly ([#36771](https://github.com/aws/aws-cdk/issues/36771)) ([458156d](https://github.com/aws/aws-cdk/commit/458156dd43ced89c893687415d7c2a2fce141653))
* **sagemaker:** add containerStartupHealthCheckTimeoutInSeconds support for EndpointConfig ([#35626](https://github.com/aws/aws-cdk/issues/35626)) ([47d707a](https://github.com/aws/aws-cdk/commit/47d707aac809fda8ec5302bf927380e8060d380a)), closes [#35566](https://github.com/aws/aws-cdk/issues/35566)

### Bug Fixes

* **eks-v2-alpha:** ensure kubectl provider access entry is depended upon by downstream resources ([#36734](https://github.com/aws/aws-cdk/issues/36734)) ([e104f45](https://github.com/aws/aws-cdk/commit/e104f45654177e87e2fb46510f77d02fcf20c499)), closes [#34898](https://github.com/aws/aws-cdk/issues/34898) [#34897](https://github.com/aws/aws-cdk/issues/34897)

## [2.235.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.235.0-alpha.0...v2.235.1-alpha.0) (2026-01-19)

## [2.235.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.234.1-alpha.0...v2.235.0-alpha.0) (2026-01-15)


### ⚠ BREAKING CHANGES

* **bedrock-agentcore-alpha:** The User Pool Client will be replaced and new Resource Server and Domain resources will be added for existing Gateway stacks using the default Cognito authorizer.

### Checklist
- [x] My code adheres to the [CONTRIBUTING GUIDE](https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md) and [DESIGN GUIDELINES](https://github.com/aws/aws-cdk/blob/main/docs/DESIGN_GUIDELINES.md)

### Bug Fixes

* **bedrock-agentcore-alpha:** default Cognito User Pool for AgentCore Gateway is not set up for M2M authentication. ([#36323](https://github.com/aws/aws-cdk/issues/36323)) ([5a5605a](https://github.com/aws/aws-cdk/commit/5a5605aafdba676ea1d73edd3ebdbfaf7dfe668d))

## [2.234.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.234.0-alpha.0...v2.234.1-alpha.0) (2026-01-08)

## [2.234.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.233.0-alpha.0...v2.234.0-alpha.0) (2026-01-08)


### Features

* **msk-alpha:** support express broker for Kafka v3.9 ([#36450](https://github.com/aws/aws-cdk/issues/36450)) ([afcc953](https://github.com/aws/aws-cdk/commit/afcc95362cbf1dff56f2a3d96f37915dc877d01b))


### Bug Fixes

* **elasticache-alpha:** deployment fails when serverlessCacheName or userGroupId is not specified ([#36459](https://github.com/aws/aws-cdk/issues/36459)) ([b3f62f7](https://github.com/aws/aws-cdk/commit/b3f62f7acd935176d540d6c4e227a4c660fc7481)), closes [#36458](https://github.com/aws/aws-cdk/issues/36458)
* **elasticache-alpha:** security group for `ServerlessCache` does not use default endpoint port ([#35738](https://github.com/aws/aws-cdk/issues/35738)) ([79d91ad](https://github.com/aws/aws-cdk/commit/79d91ad156452540525710b1c5049904bcbfc053))

## [2.233.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.232.2-alpha.0...v2.233.0-alpha.0) (2025-12-18)


### ⚠ BREAKING CHANGES

* **bedrock-agentcore-alpha:** Runtime constructs will no longer automatically include lifecycleConfiguration with default values when not explicitly specified by users.
* **elasticache-alpha:** The `engine` property in `NoPasswordUserProps` has been removed.

### Bug Fixes

* **bedrock-agentcore-alpha:** runtime construct incorrectly forces default lifecycleConfiguration values ([#36379](https://github.com/aws/aws-cdk/issues/36379)) ([7954354](https://github.com/aws/aws-cdk/commit/795435459e06e90aa9818bc99967930b125754bf)), closes [#36376](https://github.com/aws/aws-cdk/issues/36376)
* **elasticache-alpha:** the default engine for NoPasswordUser contradict in the docs ([#35920](https://github.com/aws/aws-cdk/issues/35920)) ([495fa37](https://github.com/aws/aws-cdk/commit/495fa3707a71f025f55bf37fdd13017554b89a32)), closes [#35847](https://github.com/aws/aws-cdk/issues/35847)
* **mixins-preview:** improving delivery source and delivery destination creation ([#36314](https://github.com/aws/aws-cdk/issues/36314)) ([86092ab](https://github.com/aws/aws-cdk/commit/86092ab023681bcf81678ad881b990bccc457737))

## [2.232.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.232.1-alpha.0...v2.232.2-alpha.0) (2025-12-12)

## [2.232.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.232.0-alpha.0...v2.232.1-alpha.0) (2025-12-05)

## [2.232.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.231.0-alpha.0...v2.232.0-alpha.0) (2025-12-04)


### Bug Fixes

* **bedrock-agentcore-alpha:** use static construct ID for asset-based runtime artifacts ([#36241](https://github.com/aws/aws-cdk/issues/36241)) ([e2bdddd](https://github.com/aws/aws-cdk/commit/e2bdddd82f3e04f1cb2aebb187c580563bae453f)), closes [#35968](https://github.com/aws/aws-cdk/issues/35968)
* **mixins-preview:** service exports are different then in `aws-cdk-lib` ([#36201](https://github.com/aws/aws-cdk/issues/36201)) ([5858006](https://github.com/aws/aws-cdk/commit/585800660b65a3a87d2b358054c7b5e162faabcf)), closes [#36210](https://github.com/aws/aws-cdk/issues/36210)
* **mixins-preview:** strongly-typed ConstructSelector interface ([#36266](https://github.com/aws/aws-cdk/issues/36266)) ([1d2f473](https://github.com/aws/aws-cdk/commit/1d2f4730cc4358d35198980957e4fc01a21e9daf))

## [2.231.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.230.0-alpha.0...v2.231.0-alpha.0) (2025-12-01)


### Features

* **glue-alpha:** support Glue Version 5.1 ([#36223](https://github.com/aws/aws-cdk/issues/36223)) ([b956492](https://github.com/aws/aws-cdk/commit/b9564923ee1136e7b680115a2983ad317b0c30fb))
* **imagebuilder-alpha:** add support for Image Construct ([#36154](https://github.com/aws/aws-cdk/issues/36154)) ([eee3ae6](https://github.com/aws/aws-cdk/commit/eee3ae6a55ff632012c19c384bb11e91b820bcc3)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)

## [2.230.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.229.1-alpha.0...v2.230.0-alpha.0) (2025-11-26)


### Features

* **bedrock-agentcore-alpha:** update resources on grantInvokeXXX for runtime ([#35864](https://github.com/aws/aws-cdk/issues/35864)) ([5dad62f](https://github.com/aws/aws-cdk/commit/5dad62f94c1a9f02c5fda1df7aa105e14630beb3))
* **imagebuilder-alpha:** add support for Image Pipeline Construct ([#36153](https://github.com/aws/aws-cdk/issues/36153)) ([d8c324a](https://github.com/aws/aws-cdk/commit/d8c324ae31db5997f9511e3f43affceea57d86c3)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)
* **imagebuilder-alpha:** add support for Lifecycle Policy Construct ([#36152](https://github.com/aws/aws-cdk/issues/36152)) ([7e31eb6](https://github.com/aws/aws-cdk/commit/7e31eb6a310344634fefa0651bf55c7493fc09bd)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)
* **mixins-preview:** adds LogDelivery Mixins for 47 resources ([#36158](https://github.com/aws/aws-cdk/issues/36158)) ([6607ce9](https://github.com/aws/aws-cdk/commit/6607ce95256acda86107fa30b1f000a9dc189df7))
* **mixins-preview:** vended log deliveries ([#36138](https://github.com/aws/aws-cdk/issues/36138)) ([69442a8](https://github.com/aws/aws-cdk/commit/69442a88b16b39d485bf7ec57d520599c564a612))
* **mixins-preview:** helpers to generate EventBridge event patterns for 26 services ([#36121](https://github.com/aws/aws-cdk/issues/36121)) ([073185d](https://github.com/aws/aws-cdk/commit/073185d46e5d8a95915579013d1ef8fe47e6343c))


### Bug Fixes

* **mixins-preview:** `AutoDeleteObjects` mixin fails with cannot find file error ([#36188](https://github.com/aws/aws-cdk/issues/36188)) ([3ef337d](https://github.com/aws/aws-cdk/commit/3ef337dba5d9f6f13ca8faa84376bd9177a7ecbd)), closes [aws-cdk/mixins-preview/lib/custom-resource-handlers/aws-s3/auto-delete-objects-provider.ts#L21](https://github.com/aws-cdk/mixins-preview/lib/custom-resource-handlers/aws-s3/auto-delete-objects-provider.ts/issues/L21)
* **mixins-preview:** `ResourcePolicy with this name already exists` error when setting up `LogDelivery` ([#36195](https://github.com/aws/aws-cdk/issues/36195)) ([f9aa31d](https://github.com/aws/aws-cdk/commit/f9aa31d021400aee8e172b3b0c949cd2d00473f7))
* **mixins-preview:** cannot use string literal types for `S3LogsDeliveryProps.permissionsVersion` ([#36197](https://github.com/aws/aws-cdk/issues/36197)) ([cc491df](https://github.com/aws/aws-cdk/commit/cc491df13f6a4e62e0233b90dcc636efa34e32d7))

## [2.229.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.229.0-alpha.0...v2.229.1-alpha.0) (2025-11-25)

## [2.229.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.228.0-alpha.0...v2.229.0-alpha.0) (2025-11-24)


### Features

* **imagebuilder-alpha:** add support for Container Recipe Construct ([#36091](https://github.com/aws/aws-cdk/issues/36091)) ([875e0e7](https://github.com/aws/aws-cdk/commit/875e0e72bcc791eac4ea93d223941bc3d4ba99e5)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)
* **imagebuilder-alpha:** add support for Image Recipe Construct ([#36092](https://github.com/aws/aws-cdk/issues/36092)) ([4361f8b](https://github.com/aws/aws-cdk/commit/4361f8bd0fca4b6a69b69a9ca69f178b9b220db2)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)
* **imagebuilder-alpha:** add support for Workflow Construct ([#36007](https://github.com/aws/aws-cdk/issues/36007)) ([616d32a](https://github.com/aws/aws-cdk/commit/616d32a0fde55abb203c7396801cf3adab8b5006)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)
* **mixins-preview:** developer preview of CDK Mixins ([#36136](https://github.com/aws/aws-cdk/issues/36136)) ([0c6ee1d](https://github.com/aws/aws-cdk/commit/0c6ee1d2dc2b7474d767b947558369b3d67122d9))


### Bug Fixes

* **bedrock-agentcore-alpha:** empty submodule accidentally exposed and runtime validation fix ([#36148](https://github.com/aws/aws-cdk/issues/36148)) ([72d3e6f](https://github.com/aws/aws-cdk/commit/72d3e6f40528092e99fe2390e0d0101e1075e8af))

## [2.228.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.227.0-alpha.0...v2.228.0-alpha.0) (2025-11-24)

## [2.227.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.226.0-alpha.0...v2.227.0-alpha.0) (2025-11-20)


### Features

* **bedrock-agentcore-alpha:** agentcore gateway L2 construct ([#35771](https://github.com/aws/aws-cdk/issues/35771)) ([07c4a0d](https://github.com/aws/aws-cdk/commit/07c4a0dfd4f26519f433ec3dc19b4c294ae8d56e))
* **imagebuilder-alpha:** add support for Component Construct ([#36107](https://github.com/aws/aws-cdk/issues/36107)) ([93a76e4](https://github.com/aws/aws-cdk/commit/93a76e481e79bb7e0df1aabcb158cc9b064345bf)), closes [#36006](https://github.com/aws/aws-cdk/issues/36006) [#36104](https://github.com/aws/aws-cdk/issues/36104)
* **imagebuilder-alpha:** add support for Distribution Configuration Construct ([#36108](https://github.com/aws/aws-cdk/issues/36108)) ([6051039](https://github.com/aws/aws-cdk/commit/605103939894a785062422f04ee31f5460b18d6f)), closes [#36005](https://github.com/aws/aws-cdk/issues/36005)


### Bug Fixes

* **bedrock-agentcore-alpha:** fix unexpected validation error when properties are Token  ([#35978](https://github.com/aws/aws-cdk/issues/35978)) ([084b736](https://github.com/aws/aws-cdk/commit/084b736f80959ee17a28c2d9c355b0dcf1faa393))


## [2.226.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.225.0-alpha.0...v2.226.0-alpha.0) (2025-11-20)

## [2.225.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.224.0-alpha.0...v2.225.0-alpha.0) (2025-11-17)

## [2.224.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.223.0-alpha.0...v2.224.0-alpha.0) (2025-11-13)

### Features

* **imagebuilder-alpha:** add support for EC2 Image Builder L2 Constructs - Infrastructure Configuration ([#35882](https://github.com/aws/aws-cdk/issues/35882)) ([db1d964](https://github.com/aws/aws-cdk/commit/db1d964b7775e0609c137b12051bac535094f8cd)), closes [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789) [aws/aws-cdk-rfcs#789](https://github.com/aws/aws-cdk-rfcs/issues/789)

## [2.223.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.222.0-alpha.0...v2.223.0-alpha.0) (2025-11-10)

## [2.222.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.221.1-alpha.0...v2.222.0-alpha.0) (2025-11-04)


### Features

* **eks-v2-alpha:** eks-v2-alpha is now in developer preview ([#35801](https://github.com/aws/aws-cdk/issues/35801)) ([32afc0f](https://github.com/aws/aws-cdk/commit/32afc0ff940394da737714e599ddc3c25ff738e3))


### Bug Fixes

* **bedrock-alpha:** apply permission dependency to existing and non-existing roles ([#35123](https://github.com/aws/aws-cdk/issues/35123)) ([b39ccf3](https://github.com/aws/aws-cdk/commit/b39ccf3a874401c2a0a7ae0806f1be02b9b75d5e)), closes [#35120](https://github.com/aws/aws-cdk/issues/35120)
* **eks-v2-alpha:** remove hyphen from Go package name ([#35927](https://github.com/aws/aws-cdk/issues/35927)) ([2cdfc8a](https://github.com/aws/aws-cdk/commit/2cdfc8a909ce3752833e46dd2ed0106fee0e785a))

## [2.221.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.221.0-alpha.0...v2.221.1-alpha.0) (2025-10-29)

## [2.221.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.220.0-alpha.0...v2.221.0-alpha.0) (2025-10-24)


### Features

* **msk-alpha:** support Kafka 4.1  ([#35759](https://github.com/aws/aws-cdk/issues/35759)) ([67539de](https://github.com/aws/aws-cdk/commit/67539de15ee70a61629fa8dfc7f7b42b187a82e0))


### Bug Fixes

* **elasticache-alpha:** cannot import Redis 7 serverless cache ([#35629](https://github.com/aws/aws-cdk/issues/35629)) ([2bde1a0](https://github.com/aws/aws-cdk/commit/2bde1a02ecb1fc3439249eb1bb756bf335ec3553))

## [2.220.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.219.0-alpha.0...v2.220.0-alpha.0) (2025-10-14)


### Bug Fixes

* **amplify-alpha:** handle empty customResponseHeaders array ([#35700](https://github.com/aws/aws-cdk/issues/35700)) ([57f9068](https://github.com/aws/aws-cdk/commit/57f90687d56ebe2cdc3fd5cff1dc35329c502ee4)), closes [#35693](https://github.com/aws/aws-cdk/issues/35693)
## [2.219.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.218.0-alpha.0...v2.219.0-alpha.0) (2025-10-01)

## [2.218.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.217.0-alpha.0...v2.218.0-alpha.0) (2025-09-29)
* **elasticache-alpha:** implement Serverless ElastiCache L2 Construct ([#35424](https://github.com/aws/aws-cdk/issues/35424)) ([0e08c8c](https://github.com/aws/aws-cdk/commit/0e08c8ccc17f45a9023433c17c22a951b23cad43))
## [2.217.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.216.0-alpha.0...v2.217.0-alpha.0) (2025-09-25)

## [2.216.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.215.0-alpha.0...v2.216.0-alpha.0) (2025-09-22)

## [2.215.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.214.0-alpha.0...v2.215.0-alpha.0) (2025-09-15)


### Bug Fixes

* **bedrock-alpha:** added missing validation when prompt uses default variant ([#35366](https://github.com/aws/aws-cdk/issues/35366)) ([cbd271e](https://github.com/aws/aws-cdk/commit/cbd271e36b68adbbaa1c87f95e7b99eaf3817264))

## [2.214.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.213.0-alpha.0...v2.214.0-alpha.0) (2025-09-02)


### Features

* **bedrock-alpha:** add guardrails l2 construct ([#34765](https://github.com/aws/aws-cdk/issues/34765)) ([b4fdb2b](https://github.com/aws/aws-cdk/commit/b4fdb2b494abf41eccab90cca0d0f64d38009fbe))
* **eks-v2-alpha:** support eks with k8s 1.33 ([#34713](https://github.com/aws/aws-cdk/issues/34713)) ([c24565e](https://github.com/aws/aws-cdk/commit/c24565e015d692e9a9c5dc9703372ec51b04be6b)), closes [#34520](https://github.com/aws/aws-cdk/issues/34520) [#34911](https://github.com/aws/aws-cdk/issues/34911)
* **s3tables-alpha:** add TablePolicy support to L2 construct library ([#35223](https://github.com/aws/aws-cdk/issues/35223)) ([2741dfb](https://github.com/aws/aws-cdk/commit/2741dfbf69bbfa3e4353b4b3cbb4423fddc53226)), closes [#33054](https://github.com/aws/aws-cdk/issues/33054)

## [2.213.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.212.0-alpha.0...v2.213.0-alpha.0) (2025-08-27)


### Features

* **s3tables-alpha:** add TablePolicy support to L2 construct library ([#35223](https://github.com/aws/aws-cdk/issues/35223)) ([a4aad78](https://github.com/aws/aws-cdk/commit/a4aad78a45fb776f2c2c71cb7818b4f8cbeaadd0)), closes [#33054](https://github.com/aws/aws-cdk/issues/33054)

## [2.212.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.211.0-alpha.0...v2.212.0-alpha.0) (2025-08-20)

## [2.211.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.210.0-alpha.0...v2.211.0-alpha.0) (2025-08-12)

## [2.210.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.209.1-alpha.0...v2.210.0-alpha.0) (2025-08-06)


### Features

* **glue-alpha:** add optional metrics control for cost optimization ([#35154](https://github.com/aws/aws-cdk/issues/35154)) ([6e24133](https://github.com/aws/aws-cdk/commit/6e24133d26dc2cde2cbefa8736495bfc423c5e56)), closes [#35149](https://github.com/aws/aws-cdk/issues/35149)


## [2.209.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.209.0-alpha.0...v2.209.1-alpha.0) (2025-08-06)


### Bug Fixes

* **eks-v2-alpha:** fix helm commands not running ecr public login ([#35162](https://github.com/aws/aws-cdk/issues/35162)) ([6c2a8b8](https://github.com/aws/aws-cdk/commit/6c2a8b8fd54991ce2c8557578eeb8199b8af38ac))

## [2.209.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.208.0-alpha.0...v2.209.0-alpha.0) (2025-08-05)

## [2.208.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.207.0-alpha.0...v2.208.0-alpha.0) (2025-07-29)


### Features

* **glue:** throw ValidationError instead of untyped errors ([#35084](https://github.com/aws/aws-cdk/issues/35084)) ([1e20df6](https://github.com/aws/aws-cdk/commit/1e20df640dfe1ddfd082d459fc9ff5e063b1a95c))

## [2.207.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.206.0-alpha.0...v2.207.0-alpha.0) (2025-07-24)

## [2.206.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.205.0-alpha.0...v2.206.0-alpha.0) (2025-07-16)

## [2.205.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.204.0-alpha.0...v2.205.0-alpha.0) (2025-07-15)

## [2.204.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.203.1-alpha.0...v2.204.0-alpha.0) (2025-07-04)


### Features

* **lambda-python-alpha:** support uv ([#33880](https://github.com/aws/aws-cdk/issues/33880)) ([ac6f136](https://github.com/aws/aws-cdk/commit/ac6f13695ba84d5358b69c0763c2f611af54e826)), closes [#31238](https://github.com/aws/aws-cdk/issues/31238) [#32413](https://github.com/aws/aws-cdk/issues/32413)

## [2.203.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.203.0-alpha.0...v2.203.1-alpha.0) (2025-07-02)

## [2.203.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.202.0-alpha.0...v2.203.0-alpha.0) (2025-07-01)


### Features

* **ec2:** support for client route enforcement for client VPN endpoint ([#34405](https://github.com/aws/aws-cdk/issues/34405)) ([063f4e7](https://github.com/aws/aws-cdk/commit/063f4e79d7416c52b622450222c5439e893ca74c))


### Bug Fixes

* **ec2:** don't use inferenceAccellerators in the constructors if not needed ([#34618](https://github.com/aws/aws-cdk/issues/34618)) ([054c6c5](https://github.com/aws/aws-cdk/commit/054c6c53982b8ba33ca31af6752b1662ed5752b8)), closes [#33505](https://github.com/aws/aws-cdk/issues/33505) [/github.com/aws/aws-cdk/issues/33505#issuecomment-2770818825](https://github.com/aws//github.com/aws/aws-cdk/issues/33505/issues/issuecomment-2770818825) [#34610](https://github.com/aws/aws-cdk/issues/34610)

## [2.202.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.201.0-alpha.0...v2.202.0-alpha.0) (2025-06-20)


### Features

* **amplify:** add compute role support for Amplify branches ([#34708](https://github.com/aws/aws-cdk/issues/34708)) ([817a21a](https://github.com/aws/aws-cdk/commit/817a21a782da035f7878facf8cc4d938250b70e0))
* **amplify-alpha:** support custom response headers in monorepo structures ([#31771](https://github.com/aws/aws-cdk/issues/31771)) ([aea1372](https://github.com/aws/aws-cdk/commit/aea1372ab7bc68c489cea5ee5e499233755910e8)), closes [#31758](https://github.com/aws/aws-cdk/issues/31758)

## [2.201.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.200.2-alpha.0...v2.201.0-alpha.0) (2025-06-12)


### Features

* **amplify:** add skew protection feature for Amplify branches ([#34656](https://github.com/aws/aws-cdk/issues/34656)) ([8808998](https://github.com/aws/aws-cdk/commit/8808998d0ba6e5e22e36888c10474f9788a77454))
* **bedrock:** adding bedrock l2 construct features ([#34065](https://github.com/aws/aws-cdk/issues/34065)) ([a028927](https://github.com/aws/aws-cdk/commit/a0289271aa9990f85c120d0549b878bd3c6ea484))
* **ec2:** add interface VPC endpoint for DSQL ([#34487](https://github.com/aws/aws-cdk/issues/34487)) ([3e53c2c](https://github.com/aws/aws-cdk/commit/3e53c2c0449692e639779c4315a34bca3b778cf4))
* **ec2:** support Firehose `IDeliveryStream` as flow log destination ([#34596](https://github.com/aws/aws-cdk/issues/34596)) ([cdfe6e7](https://github.com/aws/aws-cdk/commit/cdfe6e7e2946e80b3dffe9b23fd30659c0b88b99)), closes [#33883](https://github.com/aws/aws-cdk/issues/33883) [#33757](https://github.com/aws/aws-cdk/issues/33757)
* **ec2:** volume initialization rate for EBS volume ([#34570](https://github.com/aws/aws-cdk/issues/34570)) ([527d483](https://github.com/aws/aws-cdk/commit/527d483a670042caed1ba8366de81f38dde12a51))


### Reverts

* **ec2:** support Firehose `IDeliveryStream` as flow log destination ([#34665](https://github.com/aws/aws-cdk/issues/34665)) ([b77bd0e](https://github.com/aws/aws-cdk/commit/b77bd0e0da635098ca951c7f8b030bcb3cb387dd)), closes [aws/aws-cdk#34596](https://github.com/aws/aws-cdk/issues/34596)

## [2.200.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.200.1-alpha.0...v2.200.2-alpha.0) (2025-06-11)

## [2.200.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.200.0-alpha.0...v2.200.1-alpha.0) (2025-06-03)

## [2.200.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.199.0-alpha.0...v2.200.0-alpha.0) (2025-06-02)

### Features
* **msk:** support Kafka 4.0 ([#34501](https://github.com/aws/aws-cdk/issues/34501)) ([aa40e38](https://github.com/aws/aws-cdk/commit/aa40e382ddd585f027febcc994a8e32491bf37c1))

## [2.199.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.198.0-alpha.0...v2.199.0-alpha.0) (2025-05-27)


## [2.198.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.197.0-alpha.0...v2.198.0-alpha.0) (2025-05-22)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **amplify:** A compute role is created when `platform` is `Platform.WEB_COMPUTE` or `Platform.WEB_DYNAMIC`.

### Features

* **amplify:** add compute role support for Amplify app ([#33962](https://github.com/aws/aws-cdk/issues/33962)) ([7490b92](https://github.com/aws/aws-cdk/commit/7490b921aa2c464e32ad27064fa4b84571d5ba1a)), closes [#33882](https://github.com/aws/aws-cdk/issues/33882)
* **ec2:** add i7i instance class ([#34453](https://github.com/aws/aws-cdk/issues/34453)) ([3fe8ab4](https://github.com/aws/aws-cdk/commit/3fe8ab4bb100e23042f3d55d4a52bfc8cdbf495a))

## [2.197.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.196.1-alpha.0...v2.197.0-alpha.0) (2025-05-20)

## [2.196.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.196.0-alpha.0...v2.196.1-alpha.0) (2025-05-19)

## [2.196.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.195.0-alpha.0...v2.196.0-alpha.0) (2025-05-15)


### Features

* **msk:** support Kafka versions 3.9.x and 3.9.x Kraft ([#34213](https://github.com/aws/aws-cdk/issues/34213)) ([a1226db](https://github.com/aws/aws-cdk/commit/a1226db3164f885ab1bbf13a18697831cfde74d0))
* **pipes-targets:** add SNS ([#34159](https://github.com/aws/aws-cdk/issues/34159)) ([2f846b3](https://github.com/aws/aws-cdk/commit/2f846b395cc5061363bd6def946a04740ac0139b))
* **s3tables:** server-side encryption by customer managed KMS key ([#34229](https://github.com/aws/aws-cdk/issues/34229)) ([488f0db](https://github.com/aws/aws-cdk/commit/488f0db714c20fcaf5dbdf682277a70c6a938d3f))


### Bug Fixes

* **ec2:**  dual-stack vpc without private subnets creates EgressOnlyInternetGateway (under feature flag) ([#34437](https://github.com/aws/aws-cdk/issues/34437)) ([35e818b](https://github.com/aws/aws-cdk/commit/35e818b4f86638b5fe6074705511d1eee16266d2)), closes [#30981](https://github.com/aws/aws-cdk/issues/30981)
* **ec2-alpha:** fix resource id references and tags for migration behind feature flag ([#34377](https://github.com/aws/aws-cdk/issues/34377)) ([aa73534](https://github.com/aws/aws-cdk/commit/aa735341a8e95224a14241b5e1c5c5ba71de5022))

## [2.195.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.194.0-alpha.0...v2.195.0-alpha.0) (2025-05-07)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **iot:** By default, `deviceDertificateAgeCheck` is automatically enabled.

### Features

* **iot:** device certificate age check audit configuration ([#33816](https://github.com/aws/aws-cdk/issues/33816)) ([9ad383d](https://github.com/aws/aws-cdk/commit/9ad383d5300c5d5f5a9d2552fbd541436570a404))
* **location:** support L2 API Key Construct ([#32733](https://github.com/aws/aws-cdk/issues/32733)) ([d867878](https://github.com/aws/aws-cdk/commit/d86787889dd49dce220324d141bf48e1bfa8fc3b)), closes [#30684](https://github.com/aws/aws-cdk/issues/30684)


### Bug Fixes

* **amplify-alpha:** example code for adding a custom rule is wrong  ([#34353](https://github.com/aws/aws-cdk/issues/34353)) ([8ab2606](https://github.com/aws/aws-cdk/commit/8ab2606b7b8de068a70dfaf02c5d96651ef5d286)), closes [#34351](https://github.com/aws/aws-cdk/issues/34351)

## [2.194.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.193.0-alpha.0...v2.194.0-alpha.0) (2025-05-01)

## [2.193.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.192.0-alpha.0...v2.193.0-alpha.0) (2025-04-30)


### Features

* **pipes-targets-alpha:** support Amazon Data Firehose target ([#33860](https://github.com/aws/aws-cdk/issues/33860)) ([ebf1ea2](https://github.com/aws/aws-cdk/commit/ebf1ea2a57ec7876fffbe16eddac6b409ae79074))

## [2.192.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.191.0-alpha.0...v2.192.0-alpha.0) (2025-04-24)


### Features

* **applicationsignals-alpha:** introduce Application Signals L2 constructs ([#32931](https://github.com/aws/aws-cdk/issues/32931)) ([e7a6e14](https://github.com/aws/aws-cdk/commit/e7a6e14d7c3ddfbff8b1fd3f583abeefeed1258a))

## [2.191.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.190.0-alpha.0...v2.191.0-alpha.0) (2025-04-22)


### Features

* **location:** throw ValidationError instead of untyped errors ([#34174](https://github.com/aws/aws-cdk/issues/34174)) ([2ecf14a](https://github.com/aws/aws-cdk/commit/2ecf14a0c3e5a988532975536980d81589ea448e))
* **msk:** throw ValidationError instead of untyped errors ([#34214](https://github.com/aws/aws-cdk/issues/34214)) ([02cb5a4](https://github.com/aws/aws-cdk/commit/02cb5a4284e9aad2f8cc4fc8fcb2c1aebe8f92be))

## [2.190.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.189.1-alpha.0...v2.190.0-alpha.0) (2025-04-16)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **ec2-alpha:** The logical ID for the NAT Gateway, defined using the `addNatGateways` method, will be changed, resulting in the NAT Gateway being recreated. Additionally, the domain for the Elastic IP (EIP) will be set to `vpc`, which will also trigger its recreation in the account.

### Features

* **ec2:** enabling features for ipv6 and dualstack support with corresponding unit tests ([#33898](https://github.com/aws/aws-cdk/issues/33898)) ([47a65db](https://github.com/aws/aws-cdk/commit/47a65dbf00ce2a866be2546dcca5be818db70824)), closes [#3873](https://github.com/aws/aws-cdk/issues/3873) [#33493](https://github.com/aws/aws-cdk/issues/33493) [#33493](https://github.com/aws/aws-cdk/issues/33493)
* **ec2:** throw ValidationErrors instead of untyped Errors ([#34127](https://github.com/aws/aws-cdk/issues/34127)) ([93313dd](https://github.com/aws/aws-cdk/commit/93313dded1d719691689c6fb59d7a7a77bb7dade))
* **neptune-alpha:** add engine versions up to v1.4.5.0  ([#33989](https://github.com/aws/aws-cdk/issues/33989)) ([07f1d0a](https://github.com/aws/aws-cdk/commit/07f1d0a9d381fb9bceab0f836a2f1eb7977610c7)), closes [#33807](https://github.com/aws/aws-cdk/issues/33807)


### Bug Fixes

* **ec2-alpha:** add multiple NATGW to the VPC using addNatGateway method ([#34094](https://github.com/aws/aws-cdk/issues/34094)) ([ccd8de7](https://github.com/aws/aws-cdk/commit/ccd8de71c02068e43d36e2445dbb5e51f4aa695b))
* **ec2-alpha:** update default config for Subnet's `assignIpv6AddressOnCreation` ([#34116](https://github.com/aws/aws-cdk/issues/34116)) ([dff2798](https://github.com/aws/aws-cdk/commit/dff279800edd9688fa5de04766ae2667472fe861))

## [2.189.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.189.0-alpha.0...v2.189.1-alpha.0) (2025-04-14)

## [2.189.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.188.0-alpha.0...v2.189.0-alpha.0) (2025-04-09)


### Features

* **ec2-alpha:** implement mapPublicIpOnLaunch prop in SubnetV2 ([#34057](https://github.com/aws/aws-cdk/issues/34057)) ([836c5cf](https://github.com/aws/aws-cdk/commit/836c5cf3e4c627f817e4dc8ed2af28a5bba54792)), closes [#32159](https://github.com/aws/aws-cdk/issues/32159)


### Bug Fixes

* **amplify:** unable to re-run integ test due to missing `status` field in `customRule` ([#33973](https://github.com/aws/aws-cdk/issues/33973)) ([6638c08](https://github.com/aws/aws-cdk/commit/6638c08d56afe7ecc4f23cff4cf334b887001e5e)), closes [#33962](https://github.com/aws/aws-cdk/issues/33962)

## [2.188.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.187.0-alpha.0...v2.188.0-alpha.0) (2025-04-03)


### Features

* **ec2:** add mailmanager vpc endpoints ([#33996](https://github.com/aws/aws-cdk/issues/33996)) ([7ee77d7](https://github.com/aws/aws-cdk/commit/7ee77d71df569d21c280866976109333e3266132))
* **eks-v2-alpha:** add new nodegroup ami type ([#34025](https://github.com/aws/aws-cdk/issues/34025)) ([864a7c6](https://github.com/aws/aws-cdk/commit/864a7c6f6811777971d1349e7552567604167f02))


### Bug Fixes

* **ec2-alpha:** addInternetGW handles shared route table for subnets ([#33824](https://github.com/aws/aws-cdk/issues/33824)) ([3154d01](https://github.com/aws/aws-cdk/commit/3154d016ba31455f2d57ff5d90ee7b394c25e88f)), closes [#33672](https://github.com/aws/aws-cdk/issues/33672)

## [2.187.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.186.0-alpha.0...v2.187.0-alpha.0) (2025-03-31)


### Features

* **apprunner:** throw ValidationError instead of untyped errors ([#33914](https://github.com/aws/aws-cdk/issues/33914)) ([38f89af](https://github.com/aws/aws-cdk/commit/38f89afe2ffdf67b0918e38f861166bdb0f8738f))
* **ec2:** adding `placementGroup` to `LaunchTemplateProps` and `LaunchTemplate` ([#33726](https://github.com/aws/aws-cdk/issues/33726)) ([e5f71db](https://github.com/aws/aws-cdk/commit/e5f71db53ce985172e565eb9da5692d77ab7b268)), closes [#33721](https://github.com/aws/aws-cdk/issues/33721)
* **ec2:** support the new `SupportedRegions` property for `AWS::EC2::VPCEndpointService` ([#33959](https://github.com/aws/aws-cdk/issues/33959)) ([0c77cb6](https://github.com/aws/aws-cdk/commit/0c77cb627e1e7e729205624a9603331f5442af8e))
* **iot:** backfill enum values in iot module ([#33969](https://github.com/aws/aws-cdk/issues/33969)) ([2a8a8a3](https://github.com/aws/aws-cdk/commit/2a8a8a36ed872f7f3de4b24fd7d9c874a3da9dbf))

## [2.186.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.185.0-alpha.0...v2.186.0-alpha.0) (2025-03-26)


### Features

* **ec2:** backfill missing enums for ec2 ([#33821](https://github.com/aws/aws-cdk/issues/33821)) ([ae3fd67](https://github.com/aws/aws-cdk/commit/ae3fd67d3e153187d2e6fa53df9ec78080fe71d0)), closes [/docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-vpc.html#cfn-ec2](https://github.com/aws//docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-vpc.html/issues/cfn-ec2) [/docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-vpcendpoint.html#cfn-ec2](https://github.com/aws//docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-vpcendpoint.html/issues/cfn-ec2)
* **ec2:** support `PrefixList.fromLookup()` ([#33619](https://github.com/aws/aws-cdk/issues/33619)) ([b6a15f3](https://github.com/aws/aws-cdk/commit/b6a15f384e79eb0020f46ffeea20507f07380a97)), closes [aws/aws-cdk#33606](https://github.com/aws/aws-cdk/issues/33606) [aws/aws-cdk#15115](https://github.com/aws/aws-cdk/issues/15115)
* **ec2:** support AWS::EC2::VPCEndpointService SupportedIpAddressTypes property ([#33877](https://github.com/aws/aws-cdk/issues/33877)) ([ed5df9c](https://github.com/aws/aws-cdk/commit/ed5df9cac46dd862ec67751f5d0e6a53f81e8d0a))


### Bug Fixes

* **eks-v2-alpha:** prevent IAM role creation when node pools are empty ([#33894](https://github.com/aws/aws-cdk/issues/33894)) ([55bf451](https://github.com/aws/aws-cdk/commit/55bf451c48da33ce2ecda1c17cccdedea4e3527f)), closes [#33771](https://github.com/aws/aws-cdk/issues/33771)

## [2.185.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.184.1-alpha.0...v2.185.0-alpha.0) (2025-03-19)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **scheduler-targets-alpha:** The class `KinesisDataFirehosePutRecord` has been renamed to `FirehosePutRecord`.

### Bug Fixes

* **scheduler-targets-alpha:** rename `KinesisDataFirehosePutRecord` to `FirehosePutRecord` ([#33758](https://github.com/aws/aws-cdk/issues/33758)) ([e6f5bc8](https://github.com/aws/aws-cdk/commit/e6f5bc8915081a74a83e4055ccbaa11987ba943c)), closes [#33757](https://github.com/aws/aws-cdk/issues/33757) [#33798](https://github.com/aws/aws-cdk/issues/33798)

## [2.184.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.184.0-alpha.0...v2.184.1-alpha.0) (2025-03-14)

## [2.184.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.183.0-alpha.0...v2.184.0-alpha.0) (2025-03-13)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **glue-alpha:** Updated casing of `workflow.addconditionalTrigger` to `workflow.addConditionalTrigger`.

### Bug Fixes

* **glue-alpha:** inconsistent workflow addconditionalTrigger casing ([#33752](https://github.com/aws/aws-cdk/issues/33752)) ([4886a3e](https://github.com/aws/aws-cdk/commit/4886a3e503b22f3dfadca908501a2cb208c2ebee)), closes [#33751](https://github.com/aws/aws-cdk/issues/33751) [#33751](https://github.com/aws/aws-cdk/issues/33751)

## [2.183.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.182.0-alpha.0...v2.183.0-alpha.0) (2025-03-11)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **scheduler-targets-alpha:** The `InspectorStartAssessmentRun` target's constructor now accepts `IAssessmentTemplate` instead of `CfnAssessmentTemplate` as its parameter type. To migrate existing code, use the `AssessmentTemplate.fromCfnAssessmentTemplate()` method to convert your `CfnAssessmentTemplate` instances to `IAssessmentTemplate`.

### Features

* **kinesisanalytics-flink-alpha** backfill missing enums for kinesisanalytics-flink-alpha ([#33632](https://github.com/aws/aws-cdk/pull/33632)) ([b55199a](https://github.com/aws/aws-cdk/pull/33740/commits/b55199a782582348408fb75123c533977b38326d))
* **kinesisfirehose-destinations-alpha** backfill missing enums for kinesisfirehose-destinations-alpha ([#33633](https://github.com/aws/aws-cdk/pull/33633)) ([6ed7a45](https://github.com/aws/aws-cdk/pull/33740/commits/6ed7a452e261b0033b44d0b2b61b18466d6e6b48))


### Bug Fixes

* **scheduler-alpha:** deprecate `Group` in favour of `ScheduleGroup` ([#33678](https://github.com/aws/aws-cdk/issues/33678)) ([4d8eae9](https://github.com/aws/aws-cdk/commit/4d8eae9da577a94114602df261c98b65aa616956))
* **scheduler-targets-alpha:** update inspector target to use IAssessmentTemplate instead of CfnAssessmentTemplate ([#33682](https://github.com/aws/aws-cdk/issues/33682)) ([50ba3ef](https://github.com/aws/aws-cdk/commit/50ba3efabca81a3c57ce34654f8ec1002deace6f))

## [2.182.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.181.1-alpha.0...v2.182.0-alpha.0) (2025-03-04)


### Features

* **pipes-alpha:** support for customer-managed KMS keys to encrypt pipe data ([#33546](https://github.com/aws/aws-cdk/issues/33546)) ([dd0d62f](https://github.com/aws/aws-cdk/commit/dd0d62f84da06e2cafbe7a8bac80899d86b6f153)), closes [#31453](https://github.com/aws/aws-cdk/issues/31453)


### Bug Fixes

* **cognito-identitypool-alpha:** prevent stacks from not deploying correctly ([#33609](https://github.com/aws/aws-cdk/issues/33609)) ([e220bc8](https://github.com/aws/aws-cdk/commit/e220bc8ca9b75bcbb4bb7447703f32737b47fc77)), closes [#33510](https://github.com/aws/aws-cdk/issues/33510)
* **eks-v2-alpha:** can't delete fargate cluster ([#33573](https://github.com/aws/aws-cdk/issues/33573)) ([4ada313](https://github.com/aws/aws-cdk/commit/4ada3132e73e8f6b299548003d46e68f9db353a5)), closes [#33347](https://github.com/aws/aws-cdk/issues/33347)
* **scheduler-targets:** update kinesis firehose imports ([#33615](https://github.com/aws/aws-cdk/issues/33615)) ([1df1a78](https://github.com/aws/aws-cdk/commit/1df1a784ca4d4ed8c724f0a8840137724fb46ca9))

## [2.181.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.181.0-alpha.0...v2.181.1-alpha.0) (2025-02-27)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **cognito-identitypool-alpha:** Any `IdentityPool` resources deployed in versions `>=2.179.0` will now fail to deploy. You will need to delete the `IdentityPoolRoleAttachment` from your stack via the console before redeploying.

### Bug Fixes

* **cognito-identitypool-alpha:** prevent stacks from not deploying correctly ([#33609](https://github.com/aws/aws-cdk/issues/33609)) ([a1e2afe](https://github.com/aws/aws-cdk/commit/a1e2afe67cc907fa278503ebc886aa3b5bf97887)), closes [#33510](https://github.com/aws/aws-cdk/issues/33510)

## [2.181.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.180.0-alpha.0...v2.181.0-alpha.0) (2025-02-25)


### Features

* **eks-v2-alpha:** eks auto mode support ([#33373](https://github.com/aws/aws-cdk/issues/33373)) ([e03d112](https://github.com/aws/aws-cdk/commit/e03d1126231cc99a69c83d1f0146aa36b4d2b8ef))
* **msk:** support for Kafka version 3.8.x and add deprecated labels to legacy versions ([#33553](https://github.com/aws/aws-cdk/issues/33553)) ([29623d1](https://github.com/aws/aws-cdk/commit/29623d1c1f27f5733dec9b6ae0dc3efb53e28fe5))
* **redshift-alpha:** maintenance track name ([#33552](https://github.com/aws/aws-cdk/issues/33552)) ([9ac3084](https://github.com/aws/aws-cdk/commit/9ac30842b304e2a9814caf02c21a331736c258ce))

## [2.180.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.179.0-alpha.0...v2.180.0-alpha.0) (2025-02-21)


### Features

* **glue-alpha:** include extra jars parameter in pyspark jobs ([#33238](https://github.com/aws/aws-cdk/issues/33238)) ([be3bce3](https://github.com/aws/aws-cdk/commit/be3bce385c18af99b1fb3d0940a2cbec4f51e8d3)), closes [#33225](https://github.com/aws/aws-cdk/issues/33225)

## [2.179.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.178.2-alpha.0...v2.179.0-alpha.0) (2025-02-17)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **cognito-identitypool-alpha:** The `IdentityPoolRoleAttachment` construct and `IdentityPool.addRoleMappings()` function will no longer exist. This is to disambiguate that only one role attachment can exist per Identity Pool. If you are using the `IdentityPool` construct, this change will trigger a redeployment. If you need to add role mappings, please do so when the `IdentityPool` is created.

### Features

* **eks-v2-alpha:** support eks with k8s 1.32 ([#33344](https://github.com/aws/aws-cdk/issues/33344)) ([7175a04](https://github.com/aws/aws-cdk/commit/7175a042b66426864fae5199715f7076f4a95335))
* **ivs:** support Multitrack Video ([#33370](https://github.com/aws/aws-cdk/issues/33370)) ([29d1945](https://github.com/aws/aws-cdk/commit/29d194513fb9da44c9f2c1b28484381d7fb9feb3))


### Bug Fixes

* **cognito-identitypool-alpha:** remove `RoleAttachment` construct ([#33305](https://github.com/aws/aws-cdk/issues/33305)) ([9449f9c](https://github.com/aws/aws-cdk/commit/9449f9c805b045bab11a27b21924470672e804fe)), closes [#23449](https://github.com/aws/aws-cdk/issues/23449)
* **integ-tests:** http flattenResponse ([#30361](https://github.com/aws/aws-cdk/issues/30361)) ([4744ee5](https://github.com/aws/aws-cdk/commit/4744ee578116add497e5314a30629939925b015c)), closes [#30327](https://github.com/aws/aws-cdk/issues/30327)
* **msk:** allow both sasl/scram and iam auth ([#31743](https://github.com/aws/aws-cdk/issues/31743)) ([fbcb732](https://github.com/aws/aws-cdk/commit/fbcb732ed5224d2ad5a8218229762efa13db689e)), closes [/github.com/aws/aws-cdk/pull/14647#issuecomment-2129517358](https://github.com/aws//github.com/aws/aws-cdk/pull/14647/issues/issuecomment-2129517358) [#32779](https://github.com/aws/aws-cdk/issues/32779)

## [2.178.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.178.1-alpha.0...v2.178.2-alpha.0) (2025-02-12)

## [2.178.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.178.0-alpha.0...v2.178.1-alpha.0) (2025-02-06)

## [2.178.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.177.0-alpha.0...v2.178.0-alpha.0) (2025-02-05)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **ec2-alpha:** `operatingRegion` property under IPAM class is now renamed to `operatingRegions`.

### Features

* **ec2-alpha:**  `ec2-alpha` module is now in Developer Preview ([#33230](https://github.com/aws/aws-cdk/issues/33230)) ([a06f91a](https://github.com/aws/aws-cdk/commit/a06f91a79ff979b76f84dde4dec6f4d5db33e75f))
* **ec2-alpha:** add Transit Gateway L2 ([#32956](https://github.com/aws/aws-cdk/issues/32956)) ([af44791](https://github.com/aws/aws-cdk/commit/af44791ab55188bad67bb4130f9139335708a2e5))
* **eks-v2:** new eks v2 alpha module ([#33215](https://github.com/aws/aws-cdk/issues/33215)) ([ccc9f5e](https://github.com/aws/aws-cdk/commit/ccc9f5e213ed4001e31aaab9bb26ae472d511b77))
* **msk:** support ServerlessCluster ([#32780](https://github.com/aws/aws-cdk/issues/32780)) ([86ce155](https://github.com/aws/aws-cdk/commit/86ce1550abf9809ca9c16a4842b7445f0cce34e3)), closes [#28709](https://github.com/aws/aws-cdk/issues/28709)


### Bug Fixes

* **ec2-alpha:** readme updates, new unit tests, logic update ([#33086](https://github.com/aws/aws-cdk/issues/33086)) ([bcb7f9b](https://github.com/aws/aws-cdk/commit/bcb7f9bb16d6a3df55ba8d49594031d18296ca6c)), closes [#30762](https://github.com/aws/aws-cdk/issues/30762)

## [2.177.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.176.0-alpha.0...v2.177.0-alpha.0) (2025-01-24)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **glue-alpha:** Developers must refactor their existing Job
instantiation method calls to choose the right job type and language,
and use the new constants static values to define the associated Job
configuration settings. See the RFC and/or new README for examples.

### Description of how you validated changes

Increased unit test coverage to > 90%, consulted with Glue service team
on best practices and sane defaults, updated integration tests.

### Features

* **amplify-alpha:** throw `ValidationError` instead of untyped errors ([#33141](https://github.com/aws/aws-cdk/issues/33141)) ([a7cd9eb](https://github.com/aws/aws-cdk/commit/a7cd9ebc55f8fd70a469aea7dcf1c16919475982)), closes [#32569](https://github.com/aws/aws-cdk/issues/32569)


### Bug Fixes

* **custom-resource-handlers:** do not allow unauthorized connection for iam OIDC connection (under feature flag) ([#32921](https://github.com/aws/aws-cdk/issues/32921)) ([3e4f377](https://github.com/aws/aws-cdk/commit/3e4f3773bfa48b75bf0adc7d53d46bbec7714a9e)), closes [#32920](https://github.com/aws/aws-cdk/issues/32920)


### Code Refactoring

* **glue-alpha:**  Refactored glue-alpha L2 CDK construct RFC 0497 ([#32521](https://github.com/aws/aws-cdk/issues/32521)) ([1a18dc9](https://github.com/aws/aws-cdk/commit/1a18dc951a3946430231b685bd3584f62055127c))

## [2.176.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.175.1-alpha.0...v2.176.0-alpha.0) (2025-01-15)


### Features

* **scheduler-targets:** add support for universal target ([#32341](https://github.com/aws/aws-cdk/issues/32341)) ([021e6d6](https://github.com/aws/aws-cdk/commit/021e6d6ca6c01ecece485f7a51620fd42e632f0b)), closes [#32328](https://github.com/aws/aws-cdk/issues/32328)


### Bug Fixes

* **msk:** clusterName validation in Cluster class is incorrect ([#32792](https://github.com/aws/aws-cdk/issues/32792)) ([41ddd46](https://github.com/aws/aws-cdk/commit/41ddd46dc17e0afd551cce2737ecc11ed343de04)), closes [/github.com/aws/aws-cdk/pull/32505#discussion_r1891027876](https://github.com/aws//github.com/aws/aws-cdk/pull/32505/issues/discussion_r1891027876)

## [2.175.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.175.0-alpha.0...v2.175.1-alpha.0) (2025-01-10)

## [2.175.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.174.1-alpha.0...v2.175.0-alpha.0) (2025-01-09)


### Features

* **s3objectlambda:** open s3 access point arn ([#32661](https://github.com/aws/aws-cdk/issues/32661)) ([0486b9c](https://github.com/aws/aws-cdk/commit/0486b9c5e2b4286499a9d3f87a0db7c95741fb6b)), closes [#31950](https://github.com/aws/aws-cdk/issues/31950)


### Bug Fixes

* **apprunner:** the Service class does not implement IService ([#32771](https://github.com/aws/aws-cdk/issues/32771)) ([3d56efa](https://github.com/aws/aws-cdk/commit/3d56efa20ef92761ed22f12e4f651856b6889be3)), closes [#32745](https://github.com/aws/aws-cdk/issues/32745)
* **integ-runner:** `ENOENT` no such file or directory 'recommended-feature-flags.json' ([#32750](https://github.com/aws/aws-cdk/issues/32750)) ([f809b94](https://github.com/aws/aws-cdk/commit/f809b94d9952b8203221e73e177d2615c21248a8))

## [2.174.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.174.0-alpha.0...v2.174.1-alpha.0) (2025-01-07)

## [2.174.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.173.4-alpha.0...v2.174.0-alpha.0) (2025-01-04)


### Features

* **glue:** support AWS Glue 5.0 ([#32467](https://github.com/aws/aws-cdk/issues/32467)) ([ca01a25](https://github.com/aws/aws-cdk/commit/ca01a25d1328685fc9a362a1e2cbe7738389956c))
* **ec2:** add c8g and m8g instance classes ([#32528](https://github.com/aws/aws-cdk/issues/32528)) ([a81eec6](https://github.com/aws/aws-cdk/commit/a81eec66fecfc6dd27f3a4328888bd3f0c97b159)), closes [#32522](https://github.com/aws/aws-cdk/issues/32522)
* **msk-alpha:** new KafkaVersions `3_7_X` and `3_7_X_KRAFT` ([#32515](https://github.com/aws/aws-cdk/issues/32515)) ([cbacf4d](https://github.com/aws/aws-cdk/commit/cbacf4d9adde9a06fc78387b37ba1f97b2de2493))


### Bug Fixes

* **ec2-alpha:** do not use string comparison in `rangesOverlap` ([#32269](https://github.com/aws/aws-cdk/issues/32269)) ([87e21d6](https://github.com/aws/aws-cdk/commit/87e21d625af86873716734dd5568940d41096c45)), closes [#32145](https://github.com/aws/aws-cdk/issues/32145) [#32145](https://github.com/aws/aws-cdk/issues/32145)
* **redshift-alpha:** extract tableName from custom resource functions ([#32452](https://github.com/aws/aws-cdk/issues/32452)) ([283edd6](https://github.com/aws/aws-cdk/commit/283edd6601ac54cf868213d68edb7c76b3a45223)), closes [PR#24308](https://github.com/aws/PR/issues/24308)
* **redshift-alpha:** use same role for database-query singleton function ([#32363](https://github.com/aws/aws-cdk/issues/32363)) ([db950b3](https://github.com/aws/aws-cdk/commit/db950b30271e5776915936f91fa975b121a6e62c)), closes [#32089](https://github.com/aws/aws-cdk/issues/32089)

## [2.173.4-alpha.0](https://github.com/aws/aws-cdk/compare/v2.173.3-alpha.0...v2.173.4-alpha.0) (2024-12-27)

## [2.173.3-alpha.0](https://github.com/aws/aws-cdk/compare/v2.173.2-alpha.0...v2.173.3-alpha.0) (2024-12-26)

## [2.173.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.173.1-alpha.0...v2.173.2-alpha.0) (2024-12-17)

## [2.173.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.173.0-alpha.0...v2.173.1-alpha.0) (2024-12-14)

## [2.173.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.172.0-alpha.0...v2.173.0-alpha.0) (2024-12-11)


### Features

* **redshift-alpha:** add support for RA3.large node type ([#31637](https://github.com/aws/aws-cdk/issues/31637)) ([ce0e09f](https://github.com/aws/aws-cdk/commit/ce0e09fea17c78d40026df114796bc89ad365d18)), closes [#31634](https://github.com/aws/aws-cdk/issues/31634)

## [2.172.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.171.1-alpha.0...v2.172.0-alpha.0) (2024-12-06)


### Features

* **ec2:** default BastionHostLinux to use Amazon Linux 2023 (under feature flag) ([#31996](https://github.com/aws/aws-cdk/issues/31996)) ([bf77e51](https://github.com/aws/aws-cdk/commit/bf77e51c90e3da972c464430d579695163160a13)), closes [#29493](https://github.com/aws/aws-cdk/issues/29493) [#29493](https://github.com/aws/aws-cdk/issues/29493)
* **ec2:** instance support passing IAM instance profile ([#32073](https://github.com/aws/aws-cdk/issues/32073)) ([cf89d0f](https://github.com/aws/aws-cdk/commit/cf89d0f67f6d03bdeec38a4ffb48d3cda59db7cc)), closes [#8348](https://github.com/aws/aws-cdk/issues/8348)
* **neptune:** auto minor version upgrade for an instance ([#31988](https://github.com/aws/aws-cdk/issues/31988)) ([d95db49](https://github.com/aws/aws-cdk/commit/d95db491f7c1fd11dd42299f99d40fd94b0d642f))
* **pipes:** add LogDestination implementation ([#31672](https://github.com/aws/aws-cdk/issues/31672)) ([af5345e](https://github.com/aws/aws-cdk/commit/af5345e9ed2528bde2af6cd4b2428654b096eb93)), closes [#31671](https://github.com/aws/aws-cdk/issues/31671)
* **pipes-targets:** add API Gateway ([#31954](https://github.com/aws/aws-cdk/issues/31954)) ([c77536f](https://github.com/aws/aws-cdk/commit/c77536f8999e221c8d6dae5742f484a04b05bac5))
* **redshift:** execute resource action ([#31995](https://github.com/aws/aws-cdk/issues/31995)) ([40835a0](https://github.com/aws/aws-cdk/commit/40835a01536509daefa44e5e4cad5d8829d8dd1c))


### Bug Fixes

* **scheduler-targets-alpha:** incorrect validation of maximumEventAge ([#32284](https://github.com/aws/aws-cdk/issues/32284)) ([2eebc59](https://github.com/aws/aws-cdk/commit/2eebc5913966f0266efbad65c3f137c07c75270b))

## [2.171.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.171.0-alpha.0...v2.171.1-alpha.0) (2024-11-27)

## [2.171.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.170.0-alpha.0...v2.171.0-alpha.0) (2024-11-25)

## [2.170.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.169.0-alpha.0...v2.170.0-alpha.0) (2024-11-22)

## [2.169.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.168.0-alpha.0...v2.169.0-alpha.0) (2024-11-21)


### Features

* **location:** support Map ([#30648](https://github.com/aws/aws-cdk/issues/30648)) ([98b801c](https://github.com/aws/aws-cdk/commit/98b801cae9c7a957469ef61121878d81a09f4cfe)), closes [#30647](https://github.com/aws/aws-cdk/issues/30647)
* **scheduler:** `scheduler` and `scheduler-targets` modules are now in Developer Preview ([#32207](https://github.com/aws/aws-cdk/issues/32207)) ([8776832](https://github.com/aws/aws-cdk/commit/877683233f1af9674cd2e715411eed6ebc4e3e11)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)


### Bug Fixes

* **location:** underscores are not allowed in the name ([#32046](https://github.com/aws/aws-cdk/issues/32046)) ([f6ad9c9](https://github.com/aws/aws-cdk/commit/f6ad9c99db902064ab62e236fec3d7fbfcca828a))

## [2.168.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.167.2-alpha.0...v2.168.0-alpha.0) (2024-11-20)


### Bug Fixes

* **scheduler-targets-alpha:** imported target resources as schedule target throws synth error ([#32105](https://github.com/aws/aws-cdk/issues/32105)) ([1a8306f](https://github.com/aws/aws-cdk/commit/1a8306fab7d746fb66019979e8f353e17499cfbf)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785) [/github.com/aws/aws-cdk/pull/29615#issuecomment-2417858246](https://github.com/aws//github.com/aws/aws-cdk/pull/29615/issues/issuecomment-2417858246)
* **scheduler-targets-alpha:** kinesis data firehose target uses l1 instead of l2 ([#32150](https://github.com/aws/aws-cdk/issues/32150)) ([11384f0](https://github.com/aws/aws-cdk/commit/11384f0718947aebe519e346ffe31429289a9a63)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)
* **scheduler-targets-alpha:** scope down permissions for sqs and kinesis stream targets ([#32122](https://github.com/aws/aws-cdk/issues/32122)) ([6bb142e](https://github.com/aws/aws-cdk/commit/6bb142e805fdd754755cc54c31c0e6e7970be7f9)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)

## [2.167.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.167.1-alpha.0...v2.167.2-alpha.0) (2024-11-18)

## [2.167.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.167.0-alpha.0...v2.167.1-alpha.0) (2024-11-14)

## [2.167.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.166.0-alpha.0...v2.167.0-alpha.0) (2024-11-13)


### Features

* **ivs:** support recording configuration for channel ([#31899](https://github.com/aws/aws-cdk/issues/31899)) ([8a3734d](https://github.com/aws/aws-cdk/commit/8a3734d25ce36460b6fee583a7e2049b17f79d87)), closes [#31780](https://github.com/aws/aws-cdk/issues/31780)
* **redshift:** relocating a cluster ([#31993](https://github.com/aws/aws-cdk/issues/31993)) ([b763d86](https://github.com/aws/aws-cdk/commit/b763d866d660f72bf70c1cf37dadd58769642746))


### Bug Fixes

* **scheduler-targets-alpha:** add dlq policy to execution role instead of queue policy ([#32032](https://github.com/aws/aws-cdk/issues/32032)) ([b953b2a](https://github.com/aws/aws-cdk/commit/b953b2a3f01a1e75baf6426bbff5f63e49d3e626)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)

## [2.166.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.165.0-alpha.0...v2.166.0-alpha.0) (2024-11-06)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **scheduler-targets-alpha:** Schedule Target will reuse role if target is re-used across schedules. This change triggered replacement of existing roles for Schedule as logical ID of the roles are changed.

### Features

* **glue-alpha:** add job run queuing to Glue job ([#31830](https://github.com/aws/aws-cdk/issues/31830)) ([5fca268](https://github.com/aws/aws-cdk/commit/5fca268e455c1ae7c424a4dec01c0c08bec3c16c)), closes [#31826](https://github.com/aws/aws-cdk/issues/31826)


### Bug Fixes

* **scheduler-targets-alpha:** create a role per target instead of singleton schedule target role ([#31895](https://github.com/aws/aws-cdk/issues/31895)) ([aee1b30](https://github.com/aws/aws-cdk/commit/aee1b30adabebe1712720d0d7d27ed4704ac9719)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)

## [2.165.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.164.1-alpha.0...v2.165.0-alpha.0) (2024-10-31)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **ec2-alpha:** The new `VpcCidrBlock` L2 construct replaces `CfnVPCCidrBlock`. This change alters the logical ID of `AWS::EC2::VPCCidrBlock` resources in CloudFormation templates. Existing deployments will see errors like `CIDR range conflicts with x.xx.xx.xx/xx with association ID vpc-cidr-assoc-ABCD`. To resolve this, you must recreate your existing stacks to use the new module.

### Features

* **apprunner:** support vpc ingress connection ([#30623](https://github.com/aws/aws-cdk/issues/30623)) ([048e753](https://github.com/aws/aws-cdk/commit/048e7538dd71d69c2add28ff454b6d9c69b4f256)), closes [#22850](https://github.com/aws/aws-cdk/issues/22850)
* **ec2-alpha:** adding imports for SubnetV2 and VpcV2 ([#31765](https://github.com/aws/aws-cdk/issues/31765)) ([d108a80](https://github.com/aws/aws-cdk/commit/d108a8003e463301acea15076501cd815b0eda4a))
* **location:** support Tracker and TrackerConsumer ([#31268](https://github.com/aws/aws-cdk/issues/31268)) ([046f041](https://github.com/aws/aws-cdk/commit/046f0418a3de08a59c940a7a3d93148cb5f0659b)), closes [#30712](https://github.com/aws/aws-cdk/issues/30712)
* **pipes-enrichments:** support API Gateway enrichment ([#31794](https://github.com/aws/aws-cdk/issues/31794)) ([09052c2](https://github.com/aws/aws-cdk/commit/09052c2060c410028896fd54e76a857b2141c8a4)), closes [#29384](https://github.com/aws/aws-cdk/issues/29384)
* **pipes-targets:** add SageMaker ([#30696](https://github.com/aws/aws-cdk/issues/30696)) ([a5fdf57](https://github.com/aws/aws-cdk/commit/a5fdf570beb1456b1307276f56d90fd1ba0b46d8))
* **redshift-alpha:** query execution timeout setting during table creation ([#31818](https://github.com/aws/aws-cdk/issues/31818)) ([40f07ae](https://github.com/aws/aws-cdk/commit/40f07ae330d074cfa7861e24a0427da7ec427f68)), closes [#31329](https://github.com/aws/aws-cdk/issues/31329)
* **kinesisfirehose-alpha:** kinesis firehose and kinesis firehose destinations modules are now in Developer Preview ([#31952](https://github.com/aws/aws-cdk/pull/31952))

### Bug Fixes

* **location:** remove base class from PlaceIndex class ([#31287](https://github.com/aws/aws-cdk/issues/31287)) ([bc67866](https://github.com/aws/aws-cdk/commit/bc67866f579c401556d427eb150bcd118d69bd17)), closes [#30711](https://github.com/aws/aws-cdk/issues/30711) [#30682](https://github.com/aws/aws-cdk/issues/30682)
* **scheduler-alpha:** scheduler input always get transformed to string with extra double quotes ([#31894](https://github.com/aws/aws-cdk/issues/31894)) ([186b8ab](https://github.com/aws/aws-cdk/commit/186b8abfab8452b31cba13b56998242f63c43159))
* **scheduler-alpha:** too many KMS permissions granted ([#31923](https://github.com/aws/aws-cdk/issues/31923)) ([06678a3](https://github.com/aws/aws-cdk/commit/06678a39e029582af14c8b021f946b9ce9cac9be)), closes [#31785](https://github.com/aws/aws-cdk/issues/31785)

## [2.164.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.164.0-alpha.0...v2.164.1-alpha.0) (2024-10-25)

## [2.164.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.163.1-alpha.0...v2.164.0-alpha.0) (2024-10-24)


### Features

* **iot:** scheduled audit ([#31776](https://github.com/aws/aws-cdk/issues/31776)) ([366b492](https://github.com/aws/aws-cdk/commit/366b4927c50168113dd4057f6255ab6c76278135)), closes [#31779](https://github.com/aws/aws-cdk/issues/31779)


### Bug Fixes

* **ec2:** allow NAT instance to associate public IP ([#31812](https://github.com/aws/aws-cdk/issues/31812)) ([e96b4ce](https://github.com/aws/aws-cdk/commit/e96b4ce4ae64076e4c2e688c649c69fb15a624d6)), closes [#31711](https://github.com/aws/aws-cdk/issues/31711)
* **scheduler-targets-alpha:** imported lambda function as schedule target throws synth error ([#31837](https://github.com/aws/aws-cdk/issues/31837)) ([d1d179f](https://github.com/aws/aws-cdk/commit/d1d179f617f83bbb3bf44d3cc629be8eed0d4e2b)), closes [#29284](https://github.com/aws/aws-cdk/issues/29284)

## [2.163.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.163.0-alpha.0...v2.163.1-alpha.0) (2024-10-22)

## [2.163.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.162.1-alpha.0...v2.163.0-alpha.0) (2024-10-21)


### Features

* **ec2:** disable api termination ([#30620](https://github.com/aws/aws-cdk/issues/30620)) ([108737d](https://github.com/aws/aws-cdk/commit/108737d613e2a2da20a53fe92a4dac2b43d21044))
* **kinesisfirehose-alpha:** refactor sourceStream property to support multiple types of sources ([#31723](https://github.com/aws/aws-cdk/issues/31723)) ([0260046](https://github.com/aws/aws-cdk/commit/026004682f25d324b5f82b8d0ed92820c55233c1))
* **pipes-enrichments:** support API destination enrichment ([#31312](https://github.com/aws/aws-cdk/issues/31312)) ([1557793](https://github.com/aws/aws-cdk/commit/1557793f696da77ab592e81165dbbb5c0886e7e2)), closes [#29383](https://github.com/aws/aws-cdk/issues/29383)
* **pipes-targets:** add CloudWatch Logs ([#30665](https://github.com/aws/aws-cdk/issues/30665)) ([893769e](https://github.com/aws/aws-cdk/commit/893769ed22818a6c31ec1bdd58d458f50ba28c48))


### Bug Fixes

* **ec2:** exposed userDataCausesReplacement in BastionHostLinuxProps ([#31416](https://github.com/aws/aws-cdk/issues/31416)) ([029c298](https://github.com/aws/aws-cdk/commit/029c298db9875214eb16b88689b13f5e244b5ea4)), closes [#31348](https://github.com/aws/aws-cdk/issues/31348)
* **scheduler-alpha:** remove `targetOverrides` prop from Schedule ([#31799](https://github.com/aws/aws-cdk/issues/31799)) ([be4154b](https://github.com/aws/aws-cdk/commit/be4154b3a2bba28700b8476dfb26af54da0bdc6f))

## [2.162.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.162.0-alpha.0...v2.162.1-alpha.0) (2024-10-11)

## [2.162.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.161.1-alpha.0...v2.162.0-alpha.0) (2024-10-10)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **kinesisfirehose-alpha:** replaced `destinations` property with `destination` (singular) and changed the type from array of Destinations to a single Destination. Old behaviour would only allow an array with a single Destination to be passed in anyway.

### Features

* **iot-alpha:** support for account audit configuration ([#31661](https://github.com/aws/aws-cdk/issues/31661)) ([fc19571](https://github.com/aws/aws-cdk/commit/fc19571c6392e905ff03998d0e8bc4e3b01399f4)), closes [#31663](https://github.com/aws/aws-cdk/issues/31663)
* **pipes-targets:** add EventBridge ([#30654](https://github.com/aws/aws-cdk/issues/30654)) ([842f49a](https://github.com/aws/aws-cdk/commit/842f49a224ceadb1ef973dc3048ee3ac52d8f118))


### Bug Fixes

* **cli-lib:** cannot bootstrap specific environment ([#31713](https://github.com/aws/aws-cdk/issues/31713)) ([fec4bb1](https://github.com/aws/aws-cdk/commit/fec4bb1c26db54bbb151bd05239e1fc1be5de657))


### Miscellaneous Chores

* **kinesisfirehose-alpha:** replace`destinations` property with `destination` and change type from array to single IDestination ([#31630](https://github.com/aws/aws-cdk/issues/31630)) ([1e2cff1](https://github.com/aws/aws-cdk/commit/1e2cff19eec234e1d1f7f501230cba01b220a09b))

## [2.161.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.161.0-alpha.0...v2.161.1-alpha.0) (2024-10-05)

## [2.161.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.160.0-alpha.0...v2.161.0-alpha.0) (2024-10-03)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **kinesisfirehose-destinations:** the `logging` and `logGroup` properties in `DestinationLoggingProps` have been removed and replaced with a single optional property `loggingConfig` which accepts a class of type `LoggingConfig`.

#### Details
Combine the `logging` and `logGroup` properties into a single new optional property called `loggingConfig` which accepts a class of type `LoggingConfig`.

`LoggingConfig` is an abstract class which can be instantiated through either an instance of `EnableLogging` or `DisableLogging` which can be used in the following 3 ways:

```ts
import * as logs from 'aws-cdk-lib/aws-logs';

const logGroup = new logs.LogGroup(this, 'Log Group');
declare const bucket: s3.Bucket;

// 1. Enable logging with no parameters - a log group will be created for you
const destinationWithLogging = new destinations.S3Bucket(bucket, {
  loggingConfig: new destinations.EnableLogging(),
});

// 2. Enable a logging and pass in a logGroup to be used
const destinationWithLoggingAndMyLogGroup = new destinations.S3Bucket(bucket, {
  loggingConfig: new destinations.EnableLogging(logGroup),
});

// 3. Disable logging (does not accept any parameters so it is now impossible to provide a logGroup in this case)
const destinationWithoutLogging = new destinations.S3Bucket(bucket, {
  loggingConfig: new destinations.DisableLogging(),
});

```

### Description of how you validated changes
unit + integ test

### Checklist
- [x] My code adheres to the [CONTRIBUTING GUIDE](https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md) and [DESIGN GUIDELINES](https://github.com/aws/aws-cdk/blob/main/docs/DESIGN_GUIDELINES.md)

### Features

* **ec2:** add interface endpoint dynamodb ([#30162](https://github.com/aws/aws-cdk/issues/30162)) ([182804a](https://github.com/aws/aws-cdk/commit/182804a3b3116924e2f7a8e50a22e2e7d99c71ae)), closes [#29547](https://github.com/aws/aws-cdk/issues/29547)
* **pipes-sources:** add Kinesis and DynamoDB ([#29476](https://github.com/aws/aws-cdk/issues/29476)) ([00c2efb](https://github.com/aws/aws-cdk/commit/00c2efb323fdba21191c69e7f970e2cd78c37d68)), closes [#29378](https://github.com/aws/aws-cdk/issues/29378) [#29377](https://github.com/aws/aws-cdk/issues/29377)
* **pipes-targets:** add API destination ([#30756](https://github.com/aws/aws-cdk/issues/30756)) ([5e08c98](https://github.com/aws/aws-cdk/commit/5e08c981dd2a309c84abc01a0c8b358e55b5cc4c)), closes [/github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-events-targets/lib/api-gateway.ts#L11-L32](https://github.com/aws//github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-events-targets/lib/api-gateway.ts/issues/L11-L32)
* **pipes-targets:** add Kinesis ([#30656](https://github.com/aws/aws-cdk/issues/30656)) ([d0c99d8](https://github.com/aws/aws-cdk/commit/d0c99d85e0bd85beea78ce65f843d319abd493ce))
* **redshift:** supports excludeCharacters settings for DatabaseSecret ([#30563](https://github.com/aws/aws-cdk/issues/30563)) ([a1c46cf](https://github.com/aws/aws-cdk/commit/a1c46cfc5eefa58640324420a3dc15b32c37e7dd)), closes [#26847](https://github.com/aws/aws-cdk/issues/26847)


### Bug Fixes

* **custom-resource-handlers:** better fallback for require failures ([#31571](https://github.com/aws/aws-cdk/issues/31571)) ([00cdbcb](https://github.com/aws/aws-cdk/commit/00cdbcba93baa9a605f62ae8c18c0880ec15aea2)), closes [#30067](https://github.com/aws/aws-cdk/issues/30067)


### Miscellaneous Chores

* **kinesisfirehose-destinations:** refactor logging to combine logGroup and logging properties into loggingConfig ([#31488](https://github.com/aws/aws-cdk/issues/31488)) ([c4bda64](https://github.com/aws/aws-cdk/commit/c4bda6409cea78dbfa51fb6437f61fb13d0d0abb))

## [2.160.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.159.1-alpha.0...v2.160.0-alpha.0) (2024-09-24)


### Features

* **kinesisanalytics-flink:** support Apache Flink 1.20 ([#31349](https://github.com/aws/aws-cdk/issues/31349)) ([b3b9aa8](https://github.com/aws/aws-cdk/commit/b3b9aa8a4584e178808f5babe0583749d1a87da5))


### Bug Fixes

* **cognito-identitypool-alpha:** cannot configure roleMappings with imported userPool and client ([#30421](https://github.com/aws/aws-cdk/issues/30421)) ([0fdd6a9](https://github.com/aws/aws-cdk/commit/0fdd6a92ac7196e5498969be2743019f825b9262)), closes [#30304](https://github.com/aws/aws-cdk/issues/30304) [/github.com/aws/aws-cdk/blob/c3003ab41f0efc763f39eb2cab490c8a005e146b/packages/aws-cdk-lib/aws-cognito/lib/user-pool.ts#L902](https://github.com/aws//github.com/aws/aws-cdk/blob/c3003ab41f0efc763f39eb2cab490c8a005e146b/packages/aws-cdk-lib/aws-cognito/lib/user-pool.ts/issues/L902)
* **ec2:** instance resourceSignalTimeout overwrites initOptions.timeout ([#31446](https://github.com/aws/aws-cdk/issues/31446)) ([a29bf19](https://github.com/aws/aws-cdk/commit/a29bf19be1e17c13b85f6edd45c382c1f0d89702)), closes [#30052](https://github.com/aws/aws-cdk/issues/30052)

## [2.159.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.159.0-alpha.0...v2.159.1-alpha.0) (2024-09-19)

## [2.159.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.158.0-alpha.0...v2.159.0-alpha.0) (2024-09-18)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **kinesisfirehose-alpha:** `encryptionKey` property is removed and `encryption` property type has changed from the `StreamEncryption` enum to the `StreamEncryption` class.

To pass in a KMS key for the customer managed key case, use `StreamEncryption.customerManagedKey(key)`

#### Details
Replaced `encryption` and `encryptionKey` properties with a single property `encryption` of type `StreamEncryption` and is used by calling one of the 3 methods:
```ts
SreamEncryption.unencrypted()
StreamEncryption.awsOwnedKey()
StreamEncryption.customerManagedKey(key?: IKey)
```

This makes it so it's not longer possible to pass in a key when the encryption type is AWS owned or unencrypted. The `key` is an optional parameter in `StreamEncryption.customerManagedKey(key?: IKey)` so following the previous behaviour, if a key is provided it will be used, otherwise a key will be created for the user.
### Description of how you validated changes

Generated templates do not change so behaviour remains the same.

Updated integ/unit tests.

### Checklist
- [x] My code adheres to the [CONTRIBUTING GUIDE](https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md) and [DESIGN GUIDELINES](https://github.com/aws/aws-cdk/blob/main/docs/DESIGN_GUIDELINES.md)

### Features

* **ivs:** support RTMP ingest for IVS channel ([#31380](https://github.com/aws/aws-cdk/issues/31380)) ([a907a7e](https://github.com/aws/aws-cdk/commit/a907a7eb0a54f51b6e77ff57cac278de9574eee2))


### Bug Fixes

* **ec2:** fixing vpc endpoint pattern for ecr and ecr docker ([#31434](https://github.com/aws/aws-cdk/issues/31434)) ([95c49ab](https://github.com/aws/aws-cdk/commit/95c49abdfa4ad77a0c0fcb82a230778dcc2ea59a))


### Miscellaneous Chores

* **kinesisfirehose-alpha:** refactor encryption property to combine encryptionKey ([#31430](https://github.com/aws/aws-cdk/issues/31430)) ([8e92185](https://github.com/aws/aws-cdk/commit/8e9218525b606d72b2dfe55933fa1c515d26d386))

## [2.158.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.157.0-alpha.0...v2.158.0-alpha.0) (2024-09-11)


### Features

* **amplify:** support cache configuration for app ([#31381](https://github.com/aws/aws-cdk/issues/31381)) ([b7bd041](https://github.com/aws/aws-cdk/commit/b7bd0416dc1f809eac22b9dbc49a1947e1cfc262))
* **iot:** configure IoT Logging ([#31352](https://github.com/aws/aws-cdk/issues/31352)) ([6348717](https://github.com/aws/aws-cdk/commit/63487174736236e329488f4d06b29110d910a031)), closes [#31357](https://github.com/aws/aws-cdk/issues/31357)

## [2.157.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.156.0-alpha.0...v2.157.0-alpha.0) (2024-09-09)

## [2.156.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.155.0-alpha.0...v2.156.0-alpha.0) (2024-09-05)


### Features

* **location:** support RouteCalculator ([#30682](https://github.com/aws/aws-cdk/issues/30682)) ([574d383](https://github.com/aws/aws-cdk/commit/574d383ef0a5b78564bbba76e5a14a049129c042)), closes [#30681](https://github.com/aws/aws-cdk/issues/30681)
* **neptune-alpha:** specify port for the cluster ([#31137](https://github.com/aws/aws-cdk/issues/31137)) ([130b62b](https://github.com/aws/aws-cdk/commit/130b62b7904a10145b810837323451263ec7d66f)), closes [#31074](https://github.com/aws/aws-cdk/issues/31074)
* **scheduler:** validate schedule name length ([#31200](https://github.com/aws/aws-cdk/issues/31200)) ([d0f9688](https://github.com/aws/aws-cdk/commit/d0f9688423d1e12c00aebb7a9ff3cecd68d3bb3c))


### Bug Fixes

* **scheduler:** the value of the description property is not reflected to the resource. ([#31276](https://github.com/aws/aws-cdk/issues/31276)) ([a3332b6](https://github.com/aws/aws-cdk/commit/a3332b61514f13fa3c5aef87041cb2d7f81a1e92)), closes [#31269](https://github.com/aws/aws-cdk/issues/31269)

## [2.155.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.154.1-alpha.0...v2.155.0-alpha.0) (2024-08-29)


### Features

* **ec2:** `ipv6AddressCount` property for an instance ([#31076](https://github.com/aws/aws-cdk/issues/31076)) ([e3e5e1c](https://github.com/aws/aws-cdk/commit/e3e5e1c5a29a5da8759c2e821ef108eade8b87ea)), closes [#31075](https://github.com/aws/aws-cdk/issues/31075)
* **ec2:** support throughput on LaunchTemplate EBS volumes ([#30716](https://github.com/aws/aws-cdk/issues/30716)) ([6ed0bed](https://github.com/aws/aws-cdk/commit/6ed0bed61088765b9f0aa736d7daea4cbea05b3d)), closes [#24341](https://github.com/aws/aws-cdk/issues/24341) [#22441](https://github.com/aws/aws-cdk/issues/22441)
* **location:** support GeofenceCollection ([#30711](https://github.com/aws/aws-cdk/issues/30711)) ([04d73ac](https://github.com/aws/aws-cdk/commit/04d73acc5e95568bd410a714ef00e7edb192681a)), closes [#30710](https://github.com/aws/aws-cdk/issues/30710)

## [2.154.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.154.0-alpha.0...v2.154.1-alpha.0) (2024-08-23)

## [2.154.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.153.0-alpha.0...v2.154.0-alpha.0) (2024-08-22)


### Features

* **amplify:** support custom certificate ([#30791](https://github.com/aws/aws-cdk/issues/30791)) ([8d76778](https://github.com/aws/aws-cdk/commit/8d767786fe88d0ed60104ea6f48176e8981dd0fa)), closes [#30594](https://github.com/aws/aws-cdk/issues/30594)
* **ec2:** security group lookup via filters ([#30625](https://github.com/aws/aws-cdk/issues/30625)) ([abc78bf](https://github.com/aws/aws-cdk/commit/abc78bfa613453185e59d5d9a17e5c5cfb5437b3)), closes [#30331](https://github.com/aws/aws-cdk/issues/30331)
* **sagemaker:** network isolation for a model ([#30657](https://github.com/aws/aws-cdk/issues/30657)) ([f1af7fc](https://github.com/aws/aws-cdk/commit/f1af7fc66b5ca5b3a4780ed695045d2da65df3ba))


### Bug Fixes

* **cli-lib-alpha:** bootstrap fails with "bad argument name" error for trust and trustForLookup ([#31159](https://github.com/aws/aws-cdk/issues/31159)) ([b11ca4a](https://github.com/aws/aws-cdk/commit/b11ca4ae27753e33f5c44baedf1529489c4e081a)), closes [#30404](https://github.com/aws/aws-cdk/issues/30404)
* **cognito-identitypool-alpha:** validation error if provided id is a token ([#30882](https://github.com/aws/aws-cdk/issues/30882)) ([ad1b797](https://github.com/aws/aws-cdk/commit/ad1b7977768430da0ce262103e8a91f0e632ffe2)), closes [#29780](https://github.com/aws/aws-cdk/issues/29780) [#28184](https://github.com/aws/aws-cdk/issues/28184)
* **ec2:** prevent deduplication of init command args ([#30821](https://github.com/aws/aws-cdk/issues/30821)) ([1e7c690](https://github.com/aws/aws-cdk/commit/1e7c690f5ec404d7c620dc54692999fee67b3eaf)), closes [#26221](https://github.com/aws/aws-cdk/issues/26221)

## [2.153.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.152.0-alpha.0...v2.153.0-alpha.0) (2024-08-19)

## [2.152.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.151.1-alpha.0...v2.152.0-alpha.0) (2024-08-14)

## [2.151.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.151.0-alpha.0...v2.151.1-alpha.0) (2024-08-14)

## [2.151.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.150.0-alpha.0...v2.151.0-alpha.0) (2024-08-01)


### Features

* **kinesisanalytics-flink:** add support for Flink 1.19 ([#30723](https://github.com/aws/aws-cdk/issues/30723)) ([c185194](https://github.com/aws/aws-cdk/commit/c185194e2bc5bb538672df51482e1bfb134698b5)), closes [/docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-kinesisanalyticsv2-application.html#aws-resource-kinesisanalyticsv2](https://github.com/aws//docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-kinesisanalyticsv2-application.html/issues/aws-resource-kinesisanalyticsv2)

## [2.150.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.149.0-alpha.0...v2.150.0-alpha.0) (2024-07-22)

## [2.149.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.148.1-alpha.0...v2.149.0-alpha.0) (2024-07-12)

## [2.148.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.148.0-alpha.0...v2.148.1-alpha.0) (2024-07-11)

## [2.148.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.147.3-alpha.0...v2.148.0-alpha.0) (2024-07-05)


### Features

* **apprunner:** add AutoScalingConfiguration for AppRunner Service ([#30358](https://github.com/aws/aws-cdk/issues/30358)) ([a598508](https://github.com/aws/aws-cdk/commit/a598508e2a86c5932cbbbc4249cdc228bf2cbf39)), closes [#30353](https://github.com/aws/aws-cdk/issues/30353)
* **apprunner:** add ObservabilityConfiguration for AppRunner Service ([#30359](https://github.com/aws/aws-cdk/issues/30359)) ([9e9cc27](https://github.com/aws/aws-cdk/commit/9e9cc277d55774b0b2e8887676ddb328eb662d74)), closes [#22985](https://github.com/aws/aws-cdk/issues/22985)
* **pipes-enrichments:** add Step Functions enrichment eventbridge pipes ([#30495](https://github.com/aws/aws-cdk/issues/30495)) ([8b495f9](https://github.com/aws/aws-cdk/commit/8b495f9ec157c0b00674715f62b1bbcabf2096ac)), closes [#29385](https://github.com/aws/aws-cdk/issues/29385)
* **pipes-targets:** add lambda function ([#30271](https://github.com/aws/aws-cdk/issues/30271)) ([f594fae](https://github.com/aws/aws-cdk/commit/f594fae922ea19c087c56405046098e5683d2c70)), closes [#30270](https://github.com/aws/aws-cdk/issues/30270)


### Bug Fixes

* **apprunner:** auto deployment fails after new container image  pushed due to lack of a permission ([#30630](https://github.com/aws/aws-cdk/issues/30630)) ([cce10b1](https://github.com/aws/aws-cdk/commit/cce10b1c1f0f764e39e734ad7d6c288b7ffb9da3)), closes [#26640](https://github.com/aws/aws-cdk/issues/26640) [/github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-ecr/lib/repository.ts#L385](https://github.com/aws//github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-ecr/lib/repository.ts/issues/L385) [40aws-cdk/aws-apprunner-alpha/lib/service.ts#L1303](https://github.com/40aws-cdk/aws-apprunner-alpha/lib/service.ts/issues/L1303) [40aws-cdk/aws-apprunner-alpha/lib/service.ts#L1368](https://github.com/40aws-cdk/aws-apprunner-alpha/lib/service.ts/issues/L1368)

## [2.147.3-alpha.0](https://github.com/aws/aws-cdk/compare/v2.147.2-alpha.0...v2.147.3-alpha.0) (2024-07-01)

## [2.147.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.147.1-alpha.0...v2.147.2-alpha.0) (2024-06-27)

## [2.147.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.147.0-alpha.0...v2.147.1-alpha.0) (2024-06-21)

## [2.147.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.146.0-alpha.0...v2.147.0-alpha.0) (2024-06-20)


### Bug Fixes

* **custom-resource-handlers:** don't recursively process s3 bucket objects ([#30209](https://github.com/aws/aws-cdk/issues/30209)) ([25835e4](https://github.com/aws/aws-cdk/commit/25835e415bcaa7900c78914b0993605b6d6d4b68)), closes [#30573](https://github.com/aws/aws-cdk/issues/30573) [/github.com/aws/aws-cdk/pull/30209#issuecomment-2118991218](https://github.com/aws//github.com/aws/aws-cdk/pull/30209/issues/issuecomment-2118991218)

## [2.146.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.145.0-alpha.0...v2.146.0-alpha.0) (2024-06-13)


### Features

* **apprunner:** add ipAddressType property to the Service class ([#30351](https://github.com/aws/aws-cdk/issues/30351)) ([665396f](https://github.com/aws/aws-cdk/commit/665396fa8485ab642c27acf30df85f2b023acde4))

## [2.145.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.144.0-alpha.0...v2.145.0-alpha.0) (2024-06-07)

## [2.144.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.143.1-alpha.0...v2.144.0-alpha.0) (2024-05-31)


### Features

* **apprunner:** add kmsKey property for the AppRunner Service class ([#30352](https://github.com/aws/aws-cdk/issues/30352)) ([0c1aeb6](https://github.com/aws/aws-cdk/commit/0c1aeb6d6d9b9d72624a394fff45e65dfb94b733)), closes [#30365](https://github.com/aws/aws-cdk/issues/30365)
* **ivs-alpha:** support advanced channel type ([#30086](https://github.com/aws/aws-cdk/issues/30086)) ([544e54a](https://github.com/aws/aws-cdk/commit/544e54abb0985e69bb615fd3c9c47e4067bb17d8)), closes [#30075](https://github.com/aws/aws-cdk/issues/30075)
* **neptune:** add copyTagsToSnapshot property to the DatabaseCluster Construct ([#30092](https://github.com/aws/aws-cdk/issues/30092)) ([ba8edb3](https://github.com/aws/aws-cdk/commit/ba8edb3e16e3b7c72bb2a38bb3318969e0e9f054)), closes [#30087](https://github.com/aws/aws-cdk/issues/30087)

## [2.143.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.143.0-alpha.0...v2.143.1-alpha.0) (2024-05-30)

## [2.143.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.142.1-alpha.0...v2.143.0-alpha.0) (2024-05-23)

## [2.142.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.142.0-alpha.0...v2.142.1-alpha.0) (2024-05-17)

## [2.142.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.141.0-alpha.0...v2.142.0-alpha.0) (2024-05-15)


### Features

* **pipes-targets:** add step function target ([#29987](https://github.com/aws/aws-cdk/issues/29987)) ([b0975e4](https://github.com/aws/aws-cdk/commit/b0975e410a404d07952e01303af01224ccfad864)), closes [#29665](https://github.com/aws/aws-cdk/issues/29665) [#29665](https://github.com/aws/aws-cdk/issues/29665)
* **redshift:** multi AZ cluster ([#29976](https://github.com/aws/aws-cdk/issues/29976)) ([a53517c](https://github.com/aws/aws-cdk/commit/a53517c6772332cc2a15c9b38e964a933e9c8355))

## [2.141.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.140.0-alpha.0...v2.141.0-alpha.0) (2024-05-08)

## [2.140.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.139.1-alpha.0...v2.140.0-alpha.0) (2024-05-02)

## [2.139.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.139.0-alpha.0...v2.139.1-alpha.0) (2024-04-29)

## [2.139.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.138.0-alpha.0...v2.139.0-alpha.0) (2024-04-24)

## [2.138.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.137.0-alpha.0...v2.138.0-alpha.0) (2024-04-18)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **cognito-identitypool-alpha:** The argument of `IdentityPoolProviderUrl.userPool()` has been changed from `url: string` to `userPool: UserPool, userPoolClient: UserPoolClient`. If you want to specify custom identifier string, use `IdentityPoolProviderUrl.custom()` instead.

### Bug Fixes

* **cognito-identitypool-alpha:** inconvenient IdentityPoolProviderUrl.userPool() ([#29025](https://github.com/aws/aws-cdk/issues/29025)) ([90a7734](https://github.com/aws/aws-cdk/commit/90a773407753b4ff868dede4442bf20243dfeaec))

## [2.137.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.136.1-alpha.0...v2.137.0-alpha.0) (2024-04-10)


### Bug Fixes

* **integ-tests:** `httpApiCall.expect` with resolved URL ([#29705](https://github.com/aws/aws-cdk/issues/29705)) ([49b4aa1](https://github.com/aws/aws-cdk/commit/49b4aa1e22062a3183dce7092e740af49fa951bf)), closes [#29700](https://github.com/aws/aws-cdk/issues/29700) [#29701](https://github.com/aws/aws-cdk/issues/29701)

## [2.136.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.136.0-alpha.0...v2.136.1-alpha.0) (2024-04-09)

## [2.136.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.135.0-alpha.0...v2.136.0-alpha.0) (2024-04-06)

## [2.135.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.134.0-alpha.0...v2.135.0-alpha.0) (2024-04-01)

## [2.134.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.133.0-alpha.0...v2.134.0-alpha.0) (2024-03-26)


### Features

* **kinesisanalytics-flink:** add support for Flink 1.18 ([#29554](https://github.com/aws/aws-cdk/issues/29554)) ([8fd8ee8](https://github.com/aws/aws-cdk/commit/8fd8ee8e7e5a6e047e5110f084dff61906bde160)), closes [/docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-kinesisanalyticsv2-application.html#aws-resource-kinesisanalyticsv2](https://github.com/aws//docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-kinesisanalyticsv2-application.html/issues/aws-resource-kinesisanalyticsv2)


### Bug Fixes

* **glue:** s3 path specified in --spark-event-logs-path needs to end with slash ([#29357](https://github.com/aws/aws-cdk/issues/29357)) ([4ff3565](https://github.com/aws/aws-cdk/commit/4ff3565a9d7b0298bf884822fecabdd3cff643aa)), closes [#29356](https://github.com/aws/aws-cdk/issues/29356)

## [2.133.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.132.1-alpha.0...v2.133.0-alpha.0) (2024-03-14)

## [2.132.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.132.0-alpha.0...v2.132.1-alpha.0) (2024-03-12)

## [2.132.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.131.0-alpha.0...v2.132.0-alpha.0) (2024-03-08)


### Bug Fixes

* **glue:** `PythonRayExecutableProps` has innaccurate properties ([#28625](https://github.com/aws/aws-cdk/issues/28625)) ([7994733](https://github.com/aws/aws-cdk/commit/79947337d59539a03a2d7d2849043aa9405268d8)), closes [#28570](https://github.com/aws/aws-cdk/issues/28570)

## [2.131.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.130.0-alpha.0...v2.131.0-alpha.0) (2024-03-01)

## [2.130.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.129.0-alpha.0...v2.130.0-alpha.0) (2024-02-23)

## [2.129.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.128.0-alpha.0...v2.129.0-alpha.0) (2024-02-21)


### Bug Fixes

* **appconfig:** scope generated alarm role policy to '*' for composite alarm support ([#29171](https://github.com/aws/aws-cdk/issues/29171)) ([c17879d](https://github.com/aws/aws-cdk/commit/c17879dd8dab526695e69c6faf8345292634ba24))

## [2.128.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.127.0-alpha.0...v2.128.0-alpha.0) (2024-02-14)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **app-staging-synthesizer-alpha:** `stagingBucketEncryption` property is now required. For existing apps, specify `BucketEncryption.KMS` to retain existing behavior. For new apps, choose the bucket encryption that makes most sense for your use case. `BucketEncryption.S3_MANAGED` is available and is intended to be the default when this module is stabilized.

### Features

* **app-staging-synthesizer-alpha:** require passing `stagingBucketEncryption` and note that we intend to default to `S3_MANAGED` in the future ([#28978](https://github.com/aws/aws-cdk/issues/28978)) ([fc8b955](https://github.com/aws/aws-cdk/commit/fc8b95528f97704588329abaa33cf45b07273b9e)), closes [#28815](https://github.com/aws/aws-cdk/issues/28815) [#28903](https://github.com/aws/aws-cdk/issues/28903) [/github.com/aws/aws-cdk/pull/28978#issuecomment-1930007176](https://github.com/aws//github.com/aws/aws-cdk/pull/28978/issues/issuecomment-1930007176)
* **pipes-enrichments:** new EventBridge Pipes enrichments alpha module ([#29063](https://github.com/aws/aws-cdk/issues/29063)) ([5a54ec5](https://github.com/aws/aws-cdk/commit/5a54ec5923eaf580b32184a84caed118945e3f6d))
* **pipes-targets:** new EventBridge Pipes targets alpha module  ([#29057](https://github.com/aws/aws-cdk/issues/29057)) ([9419f54](https://github.com/aws/aws-cdk/commit/9419f547569ef7ec4c8b9de19498cb10b3d6c30c))
* **scheduler-targets-alpha:** `SageMakerStartPipelineExecution` Target ([#28927](https://github.com/aws/aws-cdk/issues/28927)) ([db260b0](https://github.com/aws/aws-cdk/commit/db260b091b6328b36cd8a541a5ab55048839daa6)), closes [#27457](https://github.com/aws/aws-cdk/issues/27457)


### Bug Fixes

* **appconfig:** deployment recreated on every cdk deployment ([#28782](https://github.com/aws/aws-cdk/issues/28782)) ([a21731c](https://github.com/aws/aws-cdk/commit/a21731c9b1f34525433fbee668ebb2321debf5b8))

## [2.127.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.126.0-alpha.0...v2.127.0-alpha.0) (2024-02-09)


### Features

* **pipes-sources:** new EventBridge Pipes sources alpha module ([#29048](https://github.com/aws/aws-cdk/issues/29048)) ([2e53568](https://github.com/aws/aws-cdk/commit/2e53568af8b3939413e2fe7b6d668a006b4a02d8))


### Bug Fixes

* **appconfig:** deprecate deploy method ([#29021](https://github.com/aws/aws-cdk/issues/29021)) ([9675bcd](https://github.com/aws/aws-cdk/commit/9675bcdd22dda93258b2e2bfcd24b9ef5990e704))
* **integ-tests:** cannot use v3 package name in an awsApiCall ([#28895](https://github.com/aws/aws-cdk/issues/28895)) ([5035080](https://github.com/aws/aws-cdk/commit/5035080ecc7e9e6029478496169344d5eb4b3300)), closes [/github.com/aws/aws-cdk/pull/27313/files#diff-3ab65cbf843775673ff370c9c90deceba5f0ead8a3e016e0c2f243d27bf84609](https://github.com/aws//github.com/aws/aws-cdk/pull/27313/files/issues/diff-3ab65cbf843775673ff370c9c90deceba5f0ead8a3e016e0c2f243d27bf84609) [#28844](https://github.com/aws/aws-cdk/issues/28844)

## [2.126.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.125.0-alpha.0...v2.126.0-alpha.0) (2024-02-02)

## [2.125.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.124.0-alpha.0...v2.125.0-alpha.0) (2024-01-31)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **integ-runner:** Parsing of the cli input caused arguments passed after the first instance `--language <language>` to be interpreted as a language as well. This prevented passing a test name after providing cli options. To overcome this limitation, `integ-runner` now requires an explicit `--language` option for each language you want to include: `integ-runner --language javascript --language python`. This was already documented that way and always the intended way to use this feature.
* **neptune-alpha:** Corrected LogRetention IDs for DatabaseCluster. Previously, regardless of the log type, the string ‘objectObject’ was always included, but after the correction, the log type is now included.

### Features

* **app-staging-synthesizer-alpha:** encryption type for staging bucket ([#28903](https://github.com/aws/aws-cdk/issues/28903)) ([69f4b8d](https://github.com/aws/aws-cdk/commit/69f4b8d6560a293059b9e527be1fac6d1c70c971)), closes [#28815](https://github.com/aws/aws-cdk/issues/28815)
* **pipes:** EventBridge Pipes alpha module ([#28388](https://github.com/aws/aws-cdk/issues/28388)) ([2d9106b](https://github.com/aws/aws-cdk/commit/2d9106b7bd764a5bca33b7b5e3f3faffde726757)), closes [#23495](https://github.com/aws/aws-cdk/issues/23495)


### Bug Fixes

* **integ-runner:** cannot pass test name after `--language` ([#28922](https://github.com/aws/aws-cdk/issues/28922)) ([f9fbbb4](https://github.com/aws/aws-cdk/commit/f9fbbb4bd3a3ff74c3a2b34ffa333ac4fa57de2b))
* **neptune-alpha:** multiple `cloudwatchLogsExports` cannot be set when configuring log retention ([#28643](https://github.com/aws/aws-cdk/issues/28643)) ([56794fc](https://github.com/aws/aws-cdk/commit/56794fc8a0f1f5d3c1ab29e3ee0a46b138895d32)), closes [#26295](https://github.com/aws/aws-cdk/issues/26295)

## [2.124.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.123.0-alpha.0...v2.124.0-alpha.0) (2024-01-26)

## [2.123.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.122.0-alpha.0...v2.123.0-alpha.0) (2024-01-24)


### Features

* **iot-actions-alpha:** open search action in IoT topic rule ([#28748](https://github.com/aws/aws-cdk/issues/28748)) ([84b23cb](https://github.com/aws/aws-cdk/commit/84b23cb07d421ec004e412bf48e837ae0d87f7fe))


### Bug Fixes

* **amplify:** addBranch fails synth with "cannot find entry file..." ([#28772](https://github.com/aws/aws-cdk/issues/28772)) ([cb522bb](https://github.com/aws/aws-cdk/commit/cb522bb65b03e9b0cdcbd01b4f71798e628424f4)), closes [#28658](https://github.com/aws/aws-cdk/issues/28658) [#28764](https://github.com/aws/aws-cdk/issues/28764)
* **redshift:** enableRebootForParameterChanges fails synth with "cannot find entry file…" ([#28760](https://github.com/aws/aws-cdk/issues/28760)) ([4952f36](https://github.com/aws/aws-cdk/commit/4952f36ff70c25c7a56676fedf47ab6571c19cea))

## [2.122.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.121.1-alpha.0...v2.122.0-alpha.0) (2024-01-18)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **appconfig:** `deploymentStrategyId` prop in `fromDeploymentStrategyId` now takes a `DeploymentStrategyId` rather than a `string`. To import a predefined deployment strategy id, use `DeploymentStrategyId.CANARY_10_PERCENT_20_MINUTES`. Otherwise, use `DeploymentStrategyId.fromString('abc123')`.
* **appconfig:** `ApplicationProps.name` renamed to `ApplicationProps.applicationName`
  - **appconfig**: `EnvironmentProps.name` renamed to `EnvironmentProps.environmentName`
  - **appconfig**: `DeploymentStrategyProps.name` renamed to `DeploymentStrategyProps.deploymentStrategyName`
  - **appconfig**: `ExtensionProps.name` renamed to `ExtensionProps.extensionName`

### Bug Fixes

* **amplify:** addBranch fails synth with "cannot find entry file..." ([#28658](https://github.com/aws/aws-cdk/issues/28658)) ([0f2b8f8](https://github.com/aws/aws-cdk/commit/0f2b8f8e329480cd35db1dc9792bff6d2f0a990a)), closes [#27955](https://github.com/aws/aws-cdk/issues/27955) [#28633](https://github.com/aws/aws-cdk/issues/28633) [#28089](https://github.com/aws/aws-cdk/issues/28089)
* **appconfig:** fromDeploymentStrategyId takes an enum-like class rather than a string ([#28743](https://github.com/aws/aws-cdk/issues/28743)) ([2b59ed1](https://github.com/aws/aws-cdk/commit/2b59ed1b54b5b83f22020ed5f2c4b77c6a1292f8)), closes [#28671](https://github.com/aws/aws-cdk/issues/28671)
* **appconfig:** prefix names with resource name ([#28742](https://github.com/aws/aws-cdk/issues/28742)) ([3960720](https://github.com/aws/aws-cdk/commit/396072025ea1282dd28e14158afe339c393bf0d5)), closes [#28671](https://github.com/aws/aws-cdk/issues/28671)

## [2.121.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.121.0-alpha.0...v2.121.1-alpha.0) (2024-01-13)

## [2.121.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.120.0-alpha.0...v2.121.0-alpha.0) (2024-01-12)

## [2.120.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.119.0-alpha.0...v2.120.0-alpha.0) (2024-01-12)

## [2.119.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.118.0-alpha.0...v2.119.0-alpha.0) (2024-01-11)


### Features

* **neptune-alpha:** engineVersion v1.3 ([#28647](https://github.com/aws/aws-cdk/issues/28647)) ([957598b](https://github.com/aws/aws-cdk/commit/957598b5fe08298a9ab52a1d0b2069d6a73d59c0)), closes [#28648](https://github.com/aws/aws-cdk/issues/28648)

## [2.118.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.117.0-alpha.0...v2.118.0-alpha.0) (2024-01-03)


### Features

* **glue:** database description property ([#27744](https://github.com/aws/aws-cdk/issues/27744)) ([cbac240](https://github.com/aws/aws-cdk/commit/cbac24041db7dbc39b4ae1d6da4902b3443528cb)), closes [#27740](https://github.com/aws/aws-cdk/issues/27740)
* **glue-alpha:** add `cfn-glue-table-tableinput-parameters` to Glue table construct ([#27643](https://github.com/aws/aws-cdk/issues/27643)) ([8e15482](https://github.com/aws/aws-cdk/commit/8e15482295c1324eefea020faeb11e4c686357c6))


### Bug Fixes

* **lambda-go:** path with space breaks go build ([#28554](https://github.com/aws/aws-cdk/issues/28554)) ([a8a639e](https://github.com/aws/aws-cdk/commit/a8a639e2a2114162db240361c32c40a596a7a19e)), closes [#28555](https://github.com/aws/aws-cdk/issues/28555)

## [2.117.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.116.1-alpha.0...v2.117.0-alpha.0) (2023-12-26)


### Bug Fixes

* **lambda-python-alpha:** bundling hash logic includes auth tokens in PIP urls, causing an unnecessary rebuild ([#27903](https://github.com/aws/aws-cdk/issues/27903)) ([00331a7](https://github.com/aws/aws-cdk/commit/00331a7292a3e60ea5ba3caf2a4fd4cca2fe7dc5)), closes [#27331](https://github.com/aws/aws-cdk/issues/27331)
* **lambda-python-alpha:** use function architecture ([#18696](https://github.com/aws/aws-cdk/issues/18696)) ([#28449](https://github.com/aws/aws-cdk/issues/28449)) ([c724d27](https://github.com/aws/aws-cdk/commit/c724d277277c45b421dcd89ee1a6d4b54f942d3b))

## [2.116.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.116.0-alpha.0...v2.116.1-alpha.0) (2023-12-22)

## [2.116.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.115.0-alpha.0...v2.116.0-alpha.0) (2023-12-21)


### Features

* **scheduler:** flexible time windows ([#28098](https://github.com/aws/aws-cdk/issues/28098)) ([6554e48](https://github.com/aws/aws-cdk/commit/6554e48908662de31aa5dba4578007c857c2403d))
* **scheduler-targets:** add CodePipeline as target for scheduler ([#27799](https://github.com/aws/aws-cdk/issues/27799)) ([8c44f32](https://github.com/aws/aws-cdk/commit/8c44f3298f6bd6d5b2b081eeef50296d6f716a2d)), closes [#27449](https://github.com/aws/aws-cdk/issues/27449)


### Bug Fixes

* **integ-tests:** apply correct IAM policy to waiterProvider ([#28424](https://github.com/aws/aws-cdk/issues/28424)) ([c488035](https://github.com/aws/aws-cdk/commit/c488035db893532c6aca97c59717a351539fa2ec)), closes [40aws-cdk/integ-tests-alpha/lib/assertions/sdk.ts#L136](https://github.com/40aws-cdk/integ-tests-alpha/lib/assertions/sdk.ts/issues/L136) [40aws-cdk/integ-tests-alpha/lib/assertions/sdk.ts#L247](https://github.com/40aws-cdk/integ-tests-alpha/lib/assertions/sdk.ts/issues/L247) [#27865](https://github.com/aws/aws-cdk/issues/27865)
* **lambda-python-alpha:** pipenv lock -r is no longer supported ([#28317](https://github.com/aws/aws-cdk/issues/28317)) ([f85f486](https://github.com/aws/aws-cdk/commit/f85f486d34e51c4e5d6a8b68b16a35a14f431329)), closes [#28015](https://github.com/aws/aws-cdk/issues/28015) [/github.com/pypa/pipenv/blob/main/CHANGELOG.md#2022813-2022-08-13](https://github.com/aws//github.com/pypa/pipenv/blob/main/CHANGELOG.md/issues/2022813-2022-08-13) [#28015](https://github.com/aws/aws-cdk/issues/28015)

## [2.115.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.114.1-alpha.0...v2.115.0-alpha.0) (2023-12-14)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **scheduler:** The typos in the Schedule and Group construct method names have been fixed, changing `metricSentToDLQTrunacted` to `metricSentToDLQTruncated` and `metricAllSentToDLQTrunacted` to `metricAllSentToDLQTruncated`.
* **redshift:** Further updates of the Redshift table will fail for existing tables, if the table name is changed. Therefore, changing the table name for existing Redshift tables have been disabled.

### Features

* **appconfig-alpha:** add deploy method to configuration constructs ([#28269](https://github.com/aws/aws-cdk/issues/28269)) ([c723ef9](https://github.com/aws/aws-cdk/commit/c723ef913a73fa6a452042db926023d174e86dbf))
* **cloud9-alpha:** support image ids for Amazon Linux 2023 and Ubuntu 22.04 ([#28346](https://github.com/aws/aws-cdk/issues/28346)) ([93681e0](https://github.com/aws/aws-cdk/commit/93681e07ad19c08f60eb2ee5748a2d55c6d2bc45)), closes [/docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/aws-resource-cloud9-environmentec2.html#cfn-cloud9-environmentec2](https://github.com/aws//docs.aws.amazon.com/ja_jp/AWSCloudFormation/latest/UserGuide/aws-resource-cloud9-environmentec2.html/issues/cfn-cloud9-environmentec2)
* **scheduler:** start and end time for schedule construct ([#28306](https://github.com/aws/aws-cdk/issues/28306)) ([0b4ab1d](https://github.com/aws/aws-cdk/commit/0b4ab1d0ba11b3536a2f7b02b537966de6ac0493)), closes [/github.com/aws/aws-cdk/pull/26819#discussion_r1301532299](https://github.com/aws//github.com/aws/aws-cdk/pull/26819/issues/discussion_r1301532299)


### Bug Fixes

* **appconfig-alpha:** extensions always create cdk diff ([#28264](https://github.com/aws/aws-cdk/issues/28264)) ([2075559](https://github.com/aws/aws-cdk/commit/207555919e0462686f6c434d1598e371687679c8)), closes [#27676](https://github.com/aws/aws-cdk/issues/27676)
* **redshift:** tables were dropped on table name change ([#24308](https://github.com/aws/aws-cdk/issues/24308)) ([7ac237b](https://github.com/aws/aws-cdk/commit/7ac237b08c489883962d6b8023799d6c2c40cfba)), closes [#24246](https://github.com/aws/aws-cdk/issues/24246)
* **scheduler:** typo in metricSentToDLQ... methods ([#28307](https://github.com/aws/aws-cdk/issues/28307)) ([8b91e10](https://github.com/aws/aws-cdk/commit/8b91e106e649e6a75b396f350dae9266770ad6cb))

## [2.114.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.114.0-alpha.0...v2.114.1-alpha.0) (2023-12-06)

## [2.114.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.113.0-alpha.0...v2.114.0-alpha.0) (2023-12-05)


### Features

* **appconfig-alpha:** support for composite alarms ([#28156](https://github.com/aws/aws-cdk/issues/28156)) ([d19640b](https://github.com/aws/aws-cdk/commit/d19640b1130f3701945173a81d217763386378a8))
* **appconfig-alpha:** support for relative file paths when importing config ([#28191](https://github.com/aws/aws-cdk/issues/28191)) ([4867294](https://github.com/aws/aws-cdk/commit/4867294467791b14347feef65fc3ad262bc5834a)), closes [#26937](https://github.com/aws/aws-cdk/issues/26937)
* **scheduler-targets-alpha:** `KinesisDataFirehosePutRecord` Target ([#27842](https://github.com/aws/aws-cdk/issues/27842)) ([46f3a00](https://github.com/aws/aws-cdk/commit/46f3a00c5f030fa8c6a92fb76063fba45f9b467c)), closes [#27450](https://github.com/aws/aws-cdk/issues/27450)
* **scheduler-targets-alpha:** `KinesisStreamPutRecord` Target ([#27845](https://github.com/aws/aws-cdk/issues/27845)) ([47a09b5](https://github.com/aws/aws-cdk/commit/47a09b590b5e14385f028e61ed42cd460d122fae)), closes [#27451](https://github.com/aws/aws-cdk/issues/27451)

## [2.113.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.112.0-alpha.0...v2.113.0-alpha.0) (2023-12-01)


### Features

* **msk-alpha:** MSK Kafka versions 2.8.2.tiered and 3.5.1 and StorageMode property ([#27560](https://github.com/aws/aws-cdk/issues/27560)) ([f9f15fa](https://github.com/aws/aws-cdk/commit/f9f15fa448b8a57c2a40c070e105042bdea1f26c))

## [2.112.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.111.0-alpha.0...v2.112.0-alpha.0) (2023-12-01)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **integ-tests:** Fix typo in the `InvocationType` property from `REQUEST_RESPONE` to `REQUEST_RESPONSE`

### Features

* **scheduler-targets:** eventBridge putEvents target ([#27629](https://github.com/aws/aws-cdk/issues/27629)) ([cd12ce4](https://github.com/aws/aws-cdk/commit/cd12ce4b38137f40a78b1462958a50d5b56d926c)), closes [#27454](https://github.com/aws/aws-cdk/issues/27454)
* **scheduler-targets:** SqsSendMessage Target ([#27774](https://github.com/aws/aws-cdk/issues/27774)) ([80c1d26](https://github.com/aws/aws-cdk/commit/80c1d2657cbcb63a88cc2ebd5ca02f4e03c514ac)), closes [#27458](https://github.com/aws/aws-cdk/issues/27458)
* **scheduler-targets-alpha:** `InspectorStartAssessmentRun` Target ([#27850](https://github.com/aws/aws-cdk/issues/27850)) ([073958f](https://github.com/aws/aws-cdk/commit/073958f04d9249d93013db94f21d749bc835904b)), closes [#27453](https://github.com/aws/aws-cdk/issues/27453)
* **scheduler-targets-alpha:** `SnsPublish` scheduler target ([#27838](https://github.com/aws/aws-cdk/issues/27838)) ([ff203a1](https://github.com/aws/aws-cdk/commit/ff203a19893e226d121644ae2589bf8c5b9a8440)), closes [#27459](https://github.com/aws/aws-cdk/issues/27459)


### Bug Fixes

* **cli-lib:** deploy fails with "no such file or directory, open 'node_modules/@aws-cdk/integ-runner/lib/workers/db.json.gz'" ([#28199](https://github.com/aws/aws-cdk/issues/28199)) ([78b34ac](https://github.com/aws/aws-cdk/commit/78b34accfa1cba88cc412b04df42ea5819c2cf4c)), closes [#27813](https://github.com/aws/aws-cdk/issues/27813) [#27983](https://github.com/aws/aws-cdk/issues/27983)
* **integ-tests:** fix typo in InvocationType enum property name  ([#28162](https://github.com/aws/aws-cdk/issues/28162)) ([48c275c](https://github.com/aws/aws-cdk/commit/48c275c57c945c7a3ce318522add83a8630e53b7))
* **msk-alpha:**  cluster deployment fails in `ap-southeast-1` ([#28112](https://github.com/aws/aws-cdk/issues/28112)) ([0ee4199](https://github.com/aws/aws-cdk/commit/0ee41998509c6026a849c337b680dbeb9de82a40)), closes [#28108](https://github.com/aws/aws-cdk/issues/28108)
* **scheduler:** schedule not added to group with unspecified name ([#27927](https://github.com/aws/aws-cdk/issues/27927)) ([cfa2d76](https://github.com/aws/aws-cdk/commit/cfa2d76895d8ff02e0f33df3ecad095de0c4ae3a)), closes [#27885](https://github.com/aws/aws-cdk/issues/27885)

## [2.111.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.110.1-alpha.0...v2.111.0-alpha.0) (2023-11-27)

## [2.110.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.110.0-alpha.0...v2.110.1-alpha.0) (2023-11-21)

## [2.110.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.109.0-alpha.0...v2.110.0-alpha.0) (2023-11-16)

## [2.109.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.108.1-alpha.0...v2.109.0-alpha.0) (2023-11-15)


### Bug Fixes

* **integ-runner:** fails with "no such file or directory, open 'node_modules/@aws-cdk/integ-runner/lib/workers/db.json.gz'" ([#27983](https://github.com/aws/aws-cdk/issues/27983)) ([56daf0b](https://github.com/aws/aws-cdk/commit/56daf0bb59fd4be125d5e2146ca757a183b67114))
* **integ-runner:** update workflow error message is inaccurate ([#27924](https://github.com/aws/aws-cdk/issues/27924)) ([844cd6f](https://github.com/aws/aws-cdk/commit/844cd6f0964e89c9d3b0f798aebddfac477b57af))

## [2.108.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.108.0-alpha.0...v2.108.1-alpha.0) (2023-11-14)

## [2.108.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.107.0-alpha.0...v2.108.0-alpha.0) (2023-11-13)

## [2.107.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.106.1-alpha.0...v2.107.0-alpha.0) (2023-11-13)

## [2.106.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.106.0-alpha.0...v2.106.1-alpha.0) (2023-11-11)

## [2.106.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.105.0-alpha.0...v2.106.0-alpha.0) (2023-11-10)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **appconfig:** `Environment.fromEnvironmentAttributes` function signature changed; property called `attr` is now `attrs`. This should affect only Python users.
  - **appconfig**: `Extension.fromExtensionAttributes` function signature changed; property called `attr` is now `attrs`. This should affect only Python users.

### Features

* **appconfig:** inline YAML support for hosted configuration ([#27696](https://github.com/aws/aws-cdk/issues/27696)) ([de0a9e2](https://github.com/aws/aws-cdk/commit/de0a9e218eeccbe4f685e45f93e25a250733dc51))
* **gamelift:** support Build serverSdkVersion, updated OperatingSystem values ([#27857](https://github.com/aws/aws-cdk/issues/27857)) ([f1bb801](https://github.com/aws/aws-cdk/commit/f1bb801f79e94f3cfadb4950d20b289ca425b829)), closes [#27655](https://github.com/aws/aws-cdk/issues/27655)
* **scheduler-targets:** CodeBuild scheduler target ([#27792](https://github.com/aws/aws-cdk/issues/27792)) ([9d63316](https://github.com/aws/aws-cdk/commit/9d633169a155f45c2803eab42b514d8d6d984f6e)), closes [#27448](https://github.com/aws/aws-cdk/issues/27448)


### Bug Fixes

* **appconfig:** turn on awslint and fix linter errors ([#27893](https://github.com/aws/aws-cdk/issues/27893)) ([5256de7](https://github.com/aws/aws-cdk/commit/5256de755b51c420d4357afa48d01aebf34fef85)), closes [#27894](https://github.com/aws/aws-cdk/issues/27894)

## [2.105.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.104.0-alpha.0...v2.105.0-alpha.0) (2023-11-07)

## [2.104.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.103.1-alpha.0...v2.104.0-alpha.0) (2023-11-02)


### Features

* **appconfig:** support for CfnMonitorsProperty in environments ([#27680](https://github.com/aws/aws-cdk/issues/27680)) ([05f3453](https://github.com/aws/aws-cdk/commit/05f34535561b91b8cc8b37ae296cd7a9323230ca))
* **cloud9-alpha:** add support for `federated-user` and `assumed-role` for Cloud9 environment ownership ([#27001](https://github.com/aws/aws-cdk/issues/27001)) ([00d2ff2](https://github.com/aws/aws-cdk/commit/00d2ff28e7a4ec0d6e97fe4e35d23c1f17ec4969))
* **scheduler-alpha:** target properties override ([#27603](https://github.com/aws/aws-cdk/issues/27603)) ([1433ff2](https://github.com/aws/aws-cdk/commit/1433ff23b07e50e621ee95b2e1aa2323d2bd7378)), closes [#27545](https://github.com/aws/aws-cdk/issues/27545)


### Bug Fixes

* **apigatewayv2:** defaultAuthorizer cannot be applied to HttpRoute ([#27576](https://github.com/aws/aws-cdk/issues/27576)) ([f397071](https://github.com/aws/aws-cdk/commit/f3970718ff8b4571bcfef6ebc0f480cac14e47ee)), closes [#27436](https://github.com/aws/aws-cdk/issues/27436)
* **apigatewayv2:** trigger on websocket connect and disconnect is not working ([#27732](https://github.com/aws/aws-cdk/issues/27732)) ([89f4f86](https://github.com/aws/aws-cdk/commit/89f4f86b27536d5fc891cadb88d679abb9b93b2e)), closes [#19532](https://github.com/aws/aws-cdk/issues/19532) [40aws-cdk/aws-apigatewayv2-integrations-alpha/lib/websocket/lambda.ts#L36](https://github.com/40aws-cdk/aws-apigatewayv2-integrations-alpha/lib/websocket/lambda.ts/issues/L36)

## [2.103.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.103.0-alpha.0...v2.103.1-alpha.0) (2023-10-26)

## [2.103.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.102.1-alpha.0...v2.103.0-alpha.0) (2023-10-25)

### Features

* **schedule-alpha:** support customer managed KMS keys ([#27609](https://github.com/aws/aws-cdk/issues/27609)) ([df24d22](https://github.com/aws/aws-cdk/commit/df24d22a2afae63a6fc7683ddbe2659cc269230d)), closes [#27543](https://github.com/aws/aws-cdk/issues/27543)

## [2.102.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.102.0-alpha.0...v2.102.1-alpha.0) (2023-10-25)

## [2.102.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.101.1-alpha.0...v2.102.0-alpha.0) (2023-10-18)


### Features

* **scheduler:** metrics for all schedules ([#27544](https://github.com/aws/aws-cdk/issues/27544)) ([5448009](https://github.com/aws/aws-cdk/commit/5448009738431aeebdc6fff1c1c19395d2d5a818)), closes [#23394](https://github.com/aws/aws-cdk/issues/23394)

## [2.101.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.101.0-alpha.0...v2.101.1-alpha.0) (2023-10-16)

## [2.101.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.100.0-alpha.0...v2.101.0-alpha.0) (2023-10-13)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **glue-alpha:** `SparkUIProps.prefix` strings in the original `/prefix-name` format will now result in a validation error. To retain the same behavior, prefixes must be changed to the new `prefix-name/` format.

### Features

* **lambda-python-alpha:** add without-urls option for poetry ([#27442](https://github.com/aws/aws-cdk/issues/27442)) ([5893b3f](https://github.com/aws/aws-cdk/commit/5893b3fadb7f54360db6997a98cce8dc74b86cd7)), closes [#27103](https://github.com/aws/aws-cdk/issues/27103)
* **scheduler-targets:** step function start execution target ([#27424](https://github.com/aws/aws-cdk/issues/27424)) ([3a87141](https://github.com/aws/aws-cdk/commit/3a87141cc56609e063787ce855873a059f9880ab)), closes [40aws-cdk/aws-scheduler-targets-alpha/lib/lambda-invoke.ts#L8](https://github.com/40aws-cdk/aws-scheduler-targets-alpha/lib/lambda-invoke.ts/issues/L8) [#27377](https://github.com/aws/aws-cdk/issues/27377)


### Bug Fixes

* **glue-alpha:** prefix validation logic is incorrect ([#27472](https://github.com/aws/aws-cdk/issues/27472)) ([b898d3b](https://github.com/aws/aws-cdk/commit/b898d3b9fe0d5f9ddc46c2deb71d0a95f88677fb)), closes [#27396](https://github.com/aws/aws-cdk/issues/27396)
* **integ-tests:** cannot make two or more identical assertions ([#27380](https://github.com/aws/aws-cdk/issues/27380)) ([ea06f7d](https://github.com/aws/aws-cdk/commit/ea06f7db4857e12e9b13508c64b5321a841e6dc4)), closes [#22043](https://github.com/aws/aws-cdk/issues/22043) [#23049](https://github.com/aws/aws-cdk/issues/23049)

## [2.100.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.99.1-alpha.0...v2.100.0-alpha.0) (2023-10-06)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **redshift:** the behavior of redshift tables has changed. UPDATE action will not be triggered on new table names and instead be triggered on table id changes.


### Bug Fixes

* **redshift:** UserTablePriviliges to track changes using table IDs ([#26955](https://github.com/aws/aws-cdk/issues/26955)) ([7e4fdc7](https://github.com/aws/aws-cdk/commit/7e4fdc7ec12eb17224c4156ce9340da8c2bddc72)), closes [#26558](https://github.com/aws/aws-cdk/issues/26558)

## [2.99.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.99.0-alpha.0...v2.99.1-alpha.0) (2023-09-29)

## [2.99.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.98.0-alpha.0...v2.99.0-alpha.0) (2023-09-27)


### Features

* **apprunner:** add HealthCheckConfiguration property in Service ([#27029](https://github.com/aws/aws-cdk/issues/27029)) ([4e8c9c4](https://github.com/aws/aws-cdk/commit/4e8c9c4dfdae690d9f6650bbc57bacdb83dec68c)), closes [#26972](https://github.com/aws/aws-cdk/issues/26972)


### Bug Fixes

* **appconfig:** allow multiple environment monitor roles to be created ([#27243](https://github.com/aws/aws-cdk/issues/27243)) ([9312c97](https://github.com/aws/aws-cdk/commit/9312c9763813af4ac6d2be96e78f6aeaefeeb32c))

## [2.98.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.97.1-alpha.0...v2.98.0-alpha.0) (2023-09-26)


### Features

* **scheduler:** disable Schedule on creation ([#27236](https://github.com/aws/aws-cdk/issues/27236)) ([193cd3f](https://github.com/aws/aws-cdk/commit/193cd3f575974e4058fcec957640a3d28d114fd1))

## [2.97.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.97.0-alpha.0...v2.97.1-alpha.0) (2023-09-25)

## [2.97.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.96.2-alpha.0...v2.97.0-alpha.0) (2023-09-22)


### Features

* **cli-lib-alpha:** support hotswap deployments ([#26786](https://github.com/aws/aws-cdk/issues/26786)) ([e01faff](https://github.com/aws/aws-cdk/commit/e01faffcf7228fd1a7632ff32617c77547bd8c7b)), closes [#26785](https://github.com/aws/aws-cdk/issues/26785)


### Bug Fixes

* **synthetics:** synth-time failure for canary assets in nested stages ([#27167](https://github.com/aws/aws-cdk/issues/27167)) ([7a04a5a](https://github.com/aws/aws-cdk/commit/7a04a5a280a3946692e3c4120061bd4e57ab1d6c)), closes [#27089](https://github.com/aws/aws-cdk/issues/27089) [#26291](https://github.com/aws/aws-cdk/issues/26291)

## [2.96.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.96.1-alpha.0...v2.96.2-alpha.0) (2023-09-14)

## [2.96.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.96.0-alpha.0...v2.96.1-alpha.0) (2023-09-14)

## [2.96.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.95.1-alpha.0...v2.96.0-alpha.0) (2023-09-13)


### Features

* **glue:** add ExternalTable for use with connections ([#24753](https://github.com/aws/aws-cdk/issues/24753)) ([1c03cb3](https://github.com/aws/aws-cdk/commit/1c03cb383491c164bc0914283fc3de171f6abae1)), closes [#24741](https://github.com/aws/aws-cdk/issues/24741)


### Bug Fixes

* **integ-tests:** use transformToString on API call response body ([#27122](https://github.com/aws/aws-cdk/issues/27122)) ([b0bbd5e](https://github.com/aws/aws-cdk/commit/b0bbd5e5bf8ec5d46a0afb067c8784b8fac18604)), closes [1#L573-L576](https://github.com/aws/1/issues/L573-L576) [#27114](https://github.com/aws/aws-cdk/issues/27114)
* **synthetics:** include auto-delete-underlying-resources in package ([#27096](https://github.com/aws/aws-cdk/issues/27096)) ([5046a9b](https://github.com/aws/aws-cdk/commit/5046a9b67a50bad6748077ca16a977d0844f1775))

## [2.95.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.95.0-alpha.0...v2.95.1-alpha.0) (2023-09-08)

## [2.95.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.94.0-alpha.0...v2.95.0-alpha.0) (2023-09-07)


### Bug Fixes

* **integ-tests:** Uint8Arrays are not decoded properly  ([#27009](https://github.com/aws/aws-cdk/issues/27009)) ([47ab5c8](https://github.com/aws/aws-cdk/commit/47ab5c837c598e8d854f21e82602c21098676019))

## [2.94.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.93.0-alpha.0...v2.94.0-alpha.0) (2023-09-01)


### Features

* **amplify:** enables apps hosted with server side rendering ([#26861](https://github.com/aws/aws-cdk/issues/26861)) ([c67da83](https://github.com/aws/aws-cdk/commit/c67da83d9c77eaca41ce0691dddad6da11ed397c)), closes [#24076](https://github.com/aws/aws-cdk/issues/24076) [#23325](https://github.com/aws/aws-cdk/issues/23325)
* **scheduler:** base target methods and lambda invoke target ([#26575](https://github.com/aws/aws-cdk/issues/26575)) ([39cbd46](https://github.com/aws/aws-cdk/commit/39cbd46f5d25d2304415aa2f0b5034dca0f260d8))
* **synthetics-alpha:** add latest two NodeJS runtimes ([#26967](https://github.com/aws/aws-cdk/issues/26967)) ([0a0b37c](https://github.com/aws/aws-cdk/commit/0a0b37c5ac5c38abe698f82f5e4f0e0f2cd051b7))

## [2.93.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.92.0-alpha.0...v2.93.0-alpha.0) (2023-08-23)


### Features

* **app-staging-synthesizer:** enable tag immutability ([#26656](https://github.com/aws/aws-cdk/issues/26656)) ([0bcc4b4](https://github.com/aws/aws-cdk/commit/0bcc4b4b9d0a3dab29be343c4c3db4da7bbde00a))
* **synthetics:** enable auto delete lambdas via custom resource ([#26580](https://github.com/aws/aws-cdk/issues/26580)) ([6d1dc5b](https://github.com/aws/aws-cdk/commit/6d1dc5befd4b76d8799417185d862e81da0a6796)), closes [#18448](https://github.com/aws/aws-cdk/issues/18448)


### Bug Fixes

* **lambda-python:** poetry bundling is broken after Aug 20 ([#26823](https://github.com/aws/aws-cdk/issues/26823)) ([95f8cef](https://github.com/aws/aws-cdk/commit/95f8cef0505dd2deb8ee5e45ab98c6ab1b764b02))
* **redshift:** adding distKey to an existing table fails deployment ([#26789](https://github.com/aws/aws-cdk/issues/26789)) ([8c9f0e2](https://github.com/aws/aws-cdk/commit/8c9f0e2391ad3f67b033758706c5611525081c10)), closes [#26733](https://github.com/aws/aws-cdk/issues/26733)

## [2.92.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.91.0-alpha.0...v2.92.0-alpha.0) (2023-08-15)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **batch:** if using spot instances on your Compute Environments, they will default to `SPOT_PRICE_CAPACITY_OPTIMIZED` instead of `SPOT_CAPACITY_OPTIMIZED` now.

### Features

* **batch:** grantSubmitJob method ([#26729](https://github.com/aws/aws-cdk/issues/26729)) ([716871f](https://github.com/aws/aws-cdk/commit/716871f792bf5563fc952846c1ae746eafcc2dfa)), closes [#25574](https://github.com/aws/aws-cdk/issues/25574)
* **batch:** set default spot allocation strategy to `SPOT_PRICE_CAPACITY_OPTIMIZED` ([#26731](https://github.com/aws/aws-cdk/issues/26731)) ([e0ca252](https://github.com/aws/aws-cdk/commit/e0ca252acee8290558edddde137458a055ad0b9e))

## [2.91.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.90.0-alpha.0...v2.91.0-alpha.0) (2023-08-10)


### Features

* **appconfig:** L2 constructs ([#26639](https://github.com/aws/aws-cdk/issues/26639)) ([e479bd4](https://github.com/aws/aws-cdk/commit/e479bd4353aefa5e48189d2c71f6067489afe141))
* **glue:** Job construct does not honor SparkUIProps S3 prefix when granting S3 access ([#26696](https://github.com/aws/aws-cdk/issues/26696)) ([42250f1](https://github.com/aws/aws-cdk/commit/42250f1df04b7c2ffb637c8943444ed8c0dab2df)), closes [#19862](https://github.com/aws/aws-cdk/issues/19862)


### Bug Fixes

* **glue:** synth time validation does not work in Python/Java/C#/Go ([#26650](https://github.com/aws/aws-cdk/issues/26650)) ([dba8cf3](https://github.com/aws/aws-cdk/commit/dba8cf3877663b3911c6da724f2cc5906ea60159)), closes [#26620](https://github.com/aws/aws-cdk/issues/26620)

## [2.90.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.89.0-alpha.0...v2.90.0-alpha.0) (2023-08-04)


### Features

* **glue:** glue tables can include storage parameters ([#24498](https://github.com/aws/aws-cdk/issues/24498)) ([f1df9ab](https://github.com/aws/aws-cdk/commit/f1df9ab2ba29051016f052ffe9a629ca698289b8)), closes [#23132](https://github.com/aws/aws-cdk/issues/23132)


### Bug Fixes

* **app-staging-synthesizer:** misleading error message about environment-agnostic/aware stacks ([#26607](https://github.com/aws/aws-cdk/issues/26607)) ([7e2f335](https://github.com/aws/aws-cdk/commit/7e2f335b60bda549c6abd628863b3535f9e9f153))
* **synthetics:** updated handler validation ([#26569](https://github.com/aws/aws-cdk/issues/26569)) ([1eaec92](https://github.com/aws/aws-cdk/commit/1eaec92cd7cc201c92990ab1f57a8299107327db)), closes [#26540](https://github.com/aws/aws-cdk/issues/26540)

## [2.89.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.88.0-alpha.0...v2.89.0-alpha.0) (2023-07-28)


### Features

* **app-staging-synthesizer:** option to specify staging stack name prefix ([#26324](https://github.com/aws/aws-cdk/issues/26324)) ([1b36124](https://github.com/aws/aws-cdk/commit/1b3612457078f8195fb5a73b9f0e44caf99fae96))
* **apprunner:** make `Service` implement `IGrantable` ([#26130](https://github.com/aws/aws-cdk/issues/26130)) ([6033c9a](https://github.com/aws/aws-cdk/commit/6033c9a01322be74f8ae7ddd0a3856cc22e28975)), closes [#26089](https://github.com/aws/aws-cdk/issues/26089)
* **neptune-alpha:** support for Neptune serverless ([#26445](https://github.com/aws/aws-cdk/issues/26445)) ([b42dbc8](https://github.com/aws/aws-cdk/commit/b42dbc800eabff64bc86cb8fb5629c2ce7496767)), closes [#26428](https://github.com/aws/aws-cdk/issues/26428)
* **scheduler:** ScheduleGroup ([#26196](https://github.com/aws/aws-cdk/issues/26196)) ([27dc8ff](https://github.com/aws/aws-cdk/commit/27dc8ffd62d450154ab2574cc453bb5fcdd7c0b8))


### Bug Fixes

* **cli-lib:** set skipLibCheck on generateSchema to prevent intermittent test failures ([#26551](https://github.com/aws/aws-cdk/issues/26551)) ([1807f57](https://github.com/aws/aws-cdk/commit/1807f5754885e4b1b1c8d12ca7a1cc7efab9ef2c))

## [2.88.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.87.0-alpha.0...v2.88.0-alpha.0) (2023-07-20)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **apprunner-alpha:** This change will be destructive if the `serviceName` is set on an existing resources.

### Features

* **glue:** support Data Quality ruleset ([#26272](https://github.com/aws/aws-cdk/issues/26272)) ([af3a188](https://github.com/aws/aws-cdk/commit/af3a18810847e68aa55865e380b5e4d7f9ba5edf))
* **glue:** validate maxCapacity, workerCount, and workerType ([#26241](https://github.com/aws/aws-cdk/issues/26241)) ([349e4d4](https://github.com/aws/aws-cdk/commit/349e4d4401eb6b785ebc325905a212a41d664e1f))
* **iot-actions:** iot rule https action l2 construct ([#25535](https://github.com/aws/aws-cdk/issues/25535)) ([3aee826](https://github.com/aws/aws-cdk/commit/3aee82692533a12ee44953ec4039d7cc8d6129e3)), closes [#25491](https://github.com/aws/aws-cdk/issues/25491)
* **synthetics:** lifecycle rules for auto-generated artifact buckets ([#26290](https://github.com/aws/aws-cdk/issues/26290)) ([ad0d40c](https://github.com/aws/aws-cdk/commit/ad0d40cc3a75f1cadc044393b5cb3ec7e9ab71a4)), closes [#22863](https://github.com/aws/aws-cdk/issues/22863) [#22634](https://github.com/aws/aws-cdk/issues/22634)


### Bug Fixes

* **apprunner-alpha:** respect serviceName property ([#26238](https://github.com/aws/aws-cdk/issues/26238)) ([6da9a4c](https://github.com/aws/aws-cdk/commit/6da9a4c13444d82061ffd7d1f9326ca03c2bf367)), closes [#26237](https://github.com/aws/aws-cdk/issues/26237)
* **batch:** grant execution role logs:CreateLogStream by default ([#26288](https://github.com/aws/aws-cdk/issues/26288)) ([c755f50](https://github.com/aws/aws-cdk/commit/c755f50f7d2240345c3e9ee1c262a3b194db1618)), closes [#25675](https://github.com/aws/aws-cdk/issues/25675)
* **batch:** SSM parameters can't be used as ECS Container secrets ([#26373](https://github.com/aws/aws-cdk/issues/26373)) ([bc3d6a7](https://github.com/aws/aws-cdk/commit/bc3d6a7a82fb76c3ab3915abf4ba9660a65b3414)), closes [#26339](https://github.com/aws/aws-cdk/issues/26339)
* **integ-tests-alpha:** assertions handler is broken ([#26400](https://github.com/aws/aws-cdk/issues/26400)) ([111a1cf](https://github.com/aws/aws-cdk/commit/111a1cfce0aaa7497ebd1e966f76bb3ed485d857)), closes [#26271](https://github.com/aws/aws-cdk/issues/26271) [#26359](https://github.com/aws/aws-cdk/issues/26359) [#26360](https://github.com/aws/aws-cdk/issues/26360)
* **integ-tests-alpha:** incorrect sdk client resolution ([#26271](https://github.com/aws/aws-cdk/issues/26271)) ([17e343a](https://github.com/aws/aws-cdk/commit/17e343adbd815adb276ec4ccdf5eba2d8b39607a))
* **redshift-alpha:** incorrect CR runtime version ([#26406](https://github.com/aws/aws-cdk/issues/26406)) ([c8d8421](https://github.com/aws/aws-cdk/commit/c8d8421370800384322b37b43c9644d3afec922d)), closes [#26397](https://github.com/aws/aws-cdk/issues/26397)
* **synthetics:** asset code validation failed on bundled assets ([#26291](https://github.com/aws/aws-cdk/issues/26291)) ([02a5482](https://github.com/aws/aws-cdk/commit/02a5482263b993e02c57923bda5e186d72255ade)), closes [#19342](https://github.com/aws/aws-cdk/issues/19342) [#11630](https://github.com/aws/aws-cdk/issues/11630)

## [2.87.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.86.0-alpha.0...v2.87.0-alpha.0) (2023-07-06)


### Features

* **cli-lib:** support bootstrap command ([#26205](https://github.com/aws/aws-cdk/issues/26205)) ([9364e94](https://github.com/aws/aws-cdk/commit/9364e94d1b343d18d1ceceee2881f2cc59d67980))
* **glue:** add ExecutionClass for FLEX ([#26203](https://github.com/aws/aws-cdk/issues/26203)) ([db923dd](https://github.com/aws/aws-cdk/commit/db923dd2df39d4085ed088d18dc93044e5a0d690)), closes [#22224](https://github.com/aws/aws-cdk/issues/22224)
* **iot:** add action to start Step Functions State Machine ([#26059](https://github.com/aws/aws-cdk/issues/26059)) ([bd86993](https://github.com/aws/aws-cdk/commit/bd86993cb2e528ae036347da82c86276165111e7)), closes [#17698](https://github.com/aws/aws-cdk/issues/17698)
* **scheduler:** ScheduleTargetInput ([#25663](https://github.com/aws/aws-cdk/issues/25663)) ([bc9f3de](https://github.com/aws/aws-cdk/commit/bc9f3de653248de5808f83b7fb8f3ed5f6fc554e))


### Bug Fixes

* **batch:** Allow ECS JobDefinition Containers to pass Secrets as Environment Variables & Enable Kubernetes Secret Volumes ([#26126](https://github.com/aws/aws-cdk/issues/26126)) ([dc6f120](https://github.com/aws/aws-cdk/commit/dc6f120a0bf6c9335a82677e7b3c112245bf06ae)), closes [#25559](https://github.com/aws/aws-cdk/issues/25559)
* **cli-lib:** bundle bootstrap template ([#26229](https://github.com/aws/aws-cdk/issues/26229)) ([41cb288](https://github.com/aws/aws-cdk/commit/41cb2883e637a429c9eeb30c48544b69dbc7b065)), closes [#26224](https://github.com/aws/aws-cdk/issues/26224)
* **glue:** support Ray jobs with Runtime parameter ([#25867](https://github.com/aws/aws-cdk/issues/25867)) ([8153237](https://github.com/aws/aws-cdk/commit/81532375a8745bc7ffb439e53d042b251a43e43e)), closes [#25787](https://github.com/aws/aws-cdk/issues/25787)

## [2.86.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.85.0-alpha.0...v2.86.0-alpha.0) (2023-06-29)


### Features

* **app-staging-synthesizer:** select different bootstrap region ([#26129](https://github.com/aws/aws-cdk/issues/26129)) ([2fec6a4](https://github.com/aws/aws-cdk/commit/2fec6a4cd09bd08b7183f1e67d5d7eb487e4ac29))
* **integ-runner:** integ-runner --watch ([#26087](https://github.com/aws/aws-cdk/issues/26087)) ([1fe2f09](https://github.com/aws/aws-cdk/commit/1fe2f095a0bc0aafb6b2dbd0cdaae79cc2e59ddd))
* **integ-tests:** new HttpApiCall method to easily make http calls ([#26102](https://github.com/aws/aws-cdk/issues/26102)) ([00b9c84](https://github.com/aws/aws-cdk/commit/00b9c84ecf17c05a4c794ba7b5bdc9d83b2fba16))


### Bug Fixes

* **batch-alpha:** cannot import FargateComputeEnvironment with fromFargateComputeEnvironmentArn ([#25985](https://github.com/aws/aws-cdk/issues/25985)) ([05810f4](https://github.com/aws/aws-cdk/commit/05810f44f3fa008c07c6fe39bacd2a00c52b32a0)), closes [40aws-cdk/aws-batch-alpha/lib/managed-compute-environment.ts#L1071](https://github.com/40aws-cdk/aws-batch-alpha/lib/managed-compute-environment.ts/issues/L1071) [40aws-cdk/aws-batch-alpha/lib/managed-compute-environment.ts#L1077-L1079](https://github.com/40aws-cdk/aws-batch-alpha/lib/managed-compute-environment.ts/issues/L1077-L1079) [#25979](https://github.com/aws/aws-cdk/issues/25979)

## [2.85.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.84.0-alpha.0...v2.85.0-alpha.0) (2023-06-21)


### Features

* **app-staging-synthesizer:** clean up staging resources on deletion ([#25906](https://github.com/aws/aws-cdk/issues/25906)) ([3b14213](https://github.com/aws/aws-cdk/commit/3b142136524db7c1e9bff1a082b87219ea9ee1ff)), closes [#25722](https://github.com/aws/aws-cdk/issues/25722)
* **batch:** `ephemeralStorage` property on job definitions ([#25399](https://github.com/aws/aws-cdk/issues/25399)) ([a8768f4](https://github.com/aws/aws-cdk/commit/a8768f4da1bebbc4fd45b40e92ed82e868bb2a1b)), closes [#25393](https://github.com/aws/aws-cdk/issues/25393)


### Bug Fixes

* **apprunner:** incorrect serviceName  ([#26015](https://github.com/aws/aws-cdk/issues/26015)) ([ad89f01](https://github.com/aws/aws-cdk/commit/ad89f0182e218eee01b0aef84b055a96556dda59)), closes [#26002](https://github.com/aws/aws-cdk/issues/26002)

## [2.84.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.83.1-alpha.0...v2.84.0-alpha.0) (2023-06-13)


### Bug Fixes

* **batch:** computeEnvironmentName is not set in FargateComputeEnvironment ([#25944](https://github.com/aws/aws-cdk/issues/25944)) ([fb9f559](https://github.com/aws/aws-cdk/commit/fb9f559ba0c40f5df5dc6d2a856d88826913eed4))

## [2.83.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.83.0-alpha.0...v2.83.1-alpha.0) (2023-06-09)

## [2.83.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.82.0-alpha.0...v2.83.0-alpha.0) (2023-06-07)


### Features

* **cloud9:** support setting automaticStopTimeMinutes ([#25593](https://github.com/aws/aws-cdk/issues/25593)) ([437345e](https://github.com/aws/aws-cdk/commit/437345e2ca72e67714334f4b9cb7da8f23c4a970)), closes [#25592](https://github.com/aws/aws-cdk/issues/25592)

## [2.82.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.81.0-alpha.0...v2.82.0-alpha.0) (2023-06-01)


### Features

* **synthetics:** support runtime nodejs puppeteer 4.0 ([#25553](https://github.com/aws/aws-cdk/issues/25553)) ([1d7a9a8](https://github.com/aws/aws-cdk/commit/1d7a9a80b08d41ce8759bed9286adaa8259c2bc8)), closes [#25493](https://github.com/aws/aws-cdk/issues/25493)
* **app-staging-synthesizer:** new synthesizer separates assets out per CDK application  ([#24430](https://github.com/aws/aws-cdk/issues/24430)) ([ae21ecc](https://github.com/aws/aws-cdk/commit/ae21ecc2a72be14ececdf0c5b8649e49dc456b0c))

## [2.81.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.80.0-alpha.0...v2.81.0-alpha.0) (2023-05-25)


### Features

* **batch-alpha:** tag instances launched from your managed CEs ([#25643](https://github.com/aws/aws-cdk/issues/25643)) ([8498740](https://github.com/aws/aws-cdk/commit/849874045cd1e877619c3b636e6f16a58c85b4a1))

## [2.80.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.79.1-alpha.0...v2.80.0-alpha.0) (2023-05-19)

## [2.79.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.79.0-alpha.0...v2.79.1-alpha.0) (2023-05-11)

## [2.79.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.78.0-alpha.0...v2.79.0-alpha.0) (2023-05-10)


### Bug Fixes

* **servicecatalogappregistry:** Revert deprecated method to keep deprecated method in alpha version ([b20b123](https://github.com/aws/aws-cdk/commit/b20b1f231e12007e7d064cdc4f0c9dc7354827a3))
* **batch:** JobDefinition's ContainerDefinition's Image is synthesized with `[Object object]` ([#25250](https://github.com/aws/aws-cdk/issues/25250)) ([b3d0d57](https://github.com/aws/aws-cdk/commit/b3d0d570fe02e124f4497e35eb87c96c0eb8a1d5))

## [2.78.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.77.0-alpha.0...v2.78.0-alpha.0) (2023-05-03)

## [2.77.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.76.0-alpha.0...v2.77.0-alpha.0) (2023-04-26)

## [2.76.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.76.0-alpha.0...v2.76.1-alpha.0) (2023-04-21)

## [2.76.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.75.1-alpha.0...v2.76.0-alpha.0) (2023-04-19)

## [2.75.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.75.0-alpha.0...v2.75.1-alpha.0) (2023-04-18)

## [2.75.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.74.0-alpha.0...v2.75.0-alpha.0) (2023-04-17)

## [2.74.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.73.0-alpha.0...v2.74.0-alpha.0) (2023-04-13)

## [2.73.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.72.1-alpha.0...v2.73.0-alpha.0) (2023-04-05)

## [2.72.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.72.0-alpha.0...v2.72.1-alpha.0) (2023-03-30)

## [2.72.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.71.0-alpha.0...v2.72.0-alpha.0) (2023-03-29)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **servicecatalogappregistry:** This commit involves share replacement during the deployment of `ApplicationAssociator` due to share construct id update. After this change, frequent share replacements due to structural change in `Application` construct should be avoided. `Application.shareApplication` starts to require construct id (first argument) and share name (added in `ShareOption`) as input.
* **ivs:** Renamed ChannelProps.name to ChannelProps.channelName
* Renamed PlaybackKeyPairProps.name to PlaybackKeyPairProps.playbackKeyPairName
* Channel now generates a physical name if one is not provided
* PlaybackKeyPair now generates a physical name if one is not provided

### Bug Fixes

* **integ-runner:** update workflow doesn't support resource replacement ([#24720](https://github.com/aws/aws-cdk/issues/24720)) ([07d3aa7](https://github.com/aws/aws-cdk/commit/07d3aa74e6c1a7b3b7ddf298cf3cc4b7ff180b48))
* **ivs:** Not a standard physical name pattern ([#24706](https://github.com/aws/aws-cdk/issues/24706)) ([7d17fe3](https://github.com/aws/aws-cdk/commit/7d17fe32d20cd847733bffdd899c4659a7b0003c))
* **servicecatalogappregistry:** RAM Share is replaced on every change to Application ([#24760](https://github.com/aws/aws-cdk/issues/24760)) ([8977d0d](https://github.com/aws/aws-cdk/commit/8977d0d2b567c9fcf32076b66f2dcb7f993bb22a))

## [2.71.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.70.0-alpha.0...v2.71.0-alpha.0) (2023-03-28)

## [2.70.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.69.0-alpha.0...v2.70.0-alpha.0) (2023-03-22)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **servicecatalogappregistry:** This commit contains destructive changes to the RAM Share.
Since the application RAM share name is calculated by the application construct, where one method is added. Integration test detects a breaking change where RAM share will be created. Integration test snapshot is updated to cater this destructive change.

### Features

* **servicecatalogappregistry:** add attribute groups to an application ([#24672](https://github.com/aws/aws-cdk/issues/24672)) ([7baffa2](https://github.com/aws/aws-cdk/commit/7baffa239a7904cd73ac73537101ed5bd40aa9a0))

## [2.69.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.68.0-alpha.0...v2.69.0-alpha.0) (2023-03-14)


### Features

* **kinesisanalytics-flink:** VPC support for Flink applications ([#24442](https://github.com/aws/aws-cdk/issues/24442)) ([7c7ad6d](https://github.com/aws/aws-cdk/commit/7c7ad6d18bd0d48a30858c1964d27d8a02b274ae)), closes [40aws-cdk/aws-lambda/lib/function.ts#L170](https://github.com/40aws-cdk/aws-lambda/lib/function.ts/issues/L170) [#21104](https://github.com/aws/aws-cdk/issues/21104)

## [2.68.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.67.0-alpha.0...v2.68.0-alpha.0) (2023-03-08)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **servicecatalogappregistry:** This commit contains destructive changes to the RAM Share.
Since the application RAM share name is calculated by the application construct, where one property is removed. Integration test detects a breaking change where RAM share will be created. Integration test snapshot is updated to cater this destructive change.

### Features

* **msk:** add Kafka version 3.3.2 ([#24440](https://github.com/aws/aws-cdk/issues/24440)) ([1b2014e](https://github.com/aws/aws-cdk/commit/1b2014eef9e3f2190b2cce79c55f635cc1f167e3)), closes [#24432](https://github.com/aws/aws-cdk/issues/24432)
* **redshift:** column compression encodings and comments can now be customised ([#24177](https://github.com/aws/aws-cdk/issues/24177)) ([1ca3e00](https://github.com/aws/aws-cdk/commit/1ca3e0027323e84aacade4d9bd058bbc5687a7ab)), closes [#24165](https://github.com/aws/aws-cdk/issues/24165) [#23597](https://github.com/aws/aws-cdk/issues/23597) [#22506](https://github.com/aws/aws-cdk/issues/22506)
* **redshift:** columns require an id attribute (under feature flag) ([#24272](https://github.com/aws/aws-cdk/issues/24272)) ([9a07ab0](https://github.com/aws/aws-cdk/commit/9a07ab008d1b6d23e9a302921f1a5165a21fb128)), closes [#24234](https://github.com/aws/aws-cdk/issues/24234)


### Bug Fixes

* **servicecatalogappregistry:** allow disabling automatic CfnOutput ([#24483](https://github.com/aws/aws-cdk/issues/24483)) ([3db1a0d](https://github.com/aws/aws-cdk/commit/3db1a0d0bcf615871a225919eed235b78904e144)), closes [#23779](https://github.com/aws/aws-cdk/issues/23779)
* **servicecatalogappregistry:** Associate an application with attribute group ([#24378](https://github.com/aws/aws-cdk/issues/24378)) ([d1264c1](https://github.com/aws/aws-cdk/commit/d1264c1c414257fb8dd5288fdc24cfe9605cdf90))

## [2.67.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.66.1-alpha.0...v2.67.0-alpha.0) (2023-03-02)


### Features

* **msk:** add Kafka versions 3.1.1, 3.2.0, and  and 3.3.1 ([#23918](https://github.com/aws/aws-cdk/issues/23918)) ([53a1d5f](https://github.com/aws/aws-cdk/commit/53a1d5fd81eabf5e9d846411754a554549f9f62c)), closes [#23899](https://github.com/aws/aws-cdk/issues/23899)


### Bug Fixes

* **servicecatalogappregistry:** applicationName can not be changed after deployment ([#24409](https://github.com/aws/aws-cdk/issues/24409)) ([6aa763f](https://github.com/aws/aws-cdk/commit/6aa763f100e5561f4554627116a458abba930480))

## [2.66.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.66.0-alpha.0...v2.66.1-alpha.0) (2023-02-23)

## [2.66.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.65.0-alpha.0...v2.66.0-alpha.0) (2023-02-21)


### Features

* **apigatewayv2:** allow websockets routes to return response to client ([#22984](https://github.com/aws/aws-cdk/issues/22984)) ([f8fe1d2](https://github.com/aws/aws-cdk/commit/f8fe1d292feb3fc39a99687bf454a829302c4ff5))
* **lambda-python:** add optional poetry bundling exclusion list parameter ([#23670](https://github.com/aws/aws-cdk/issues/23670)) ([53beeae](https://github.com/aws/aws-cdk/commit/53beeaed04bfe295e9f840e65f9c89db00cac692)), closes [#22585](https://github.com/aws/aws-cdk/issues/22585) [#22585](https://github.com/aws/aws-cdk/issues/22585)
* **redshift:** optionally reboot Clusters to apply parameter changes  ([#22063](https://github.com/aws/aws-cdk/issues/22063)) ([f61d950](https://github.com/aws/aws-cdk/commit/f61d950aaeba13bd6501b7c8971a9115f4a53f08)), closes [#22009](https://github.com/aws/aws-cdk/issues/22009) [#22055](https://github.com/aws/aws-cdk/issues/22055) [#22059](https://github.com/aws/aws-cdk/issues/22059)


### Bug Fixes

* **servicecatalogappregistry:** Allow user to control stack id via stack name for Application stack ([#24171](https://github.com/aws/aws-cdk/issues/24171)) ([0c7c7e4](https://github.com/aws/aws-cdk/commit/0c7c7e4a7c34957ff7877eda5171f82c5feaba1d)), closes [#24160](https://github.com/aws/aws-cdk/issues/24160)

## [2.65.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.64.0-alpha.0...v2.65.0-alpha.0) (2023-02-15)


### Features

* **glue:** support Ray jobs ([#23822](https://github.com/aws/aws-cdk/issues/23822)) ([8de50d6](https://github.com/aws/aws-cdk/commit/8de50d624c8703a12713dcffbc764688868f22b0))
* **redshift:** IAM roles can be attached to a cluster, post creation ([#23791](https://github.com/aws/aws-cdk/issues/23791)) ([1a46808](https://github.com/aws/aws-cdk/commit/1a46808b03e8f6d09846f999ae3dc65b190f5f26)), closes [#22632](https://github.com/aws/aws-cdk/issues/22632)
* **synthetics:** support runtime 3.9 ([#24101](https://github.com/aws/aws-cdk/issues/24101)) ([9d23cad](https://github.com/aws/aws-cdk/commit/9d23caded8aca42d3b78de1bc7e89c38a4d6805e))

## [2.64.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.63.2-alpha.0...v2.64.0-alpha.0) (2023-02-09)


### Features

* **cloud9:** support setting environment owner ([#23878](https://github.com/aws/aws-cdk/issues/23878)) ([08a2f36](https://github.com/aws/aws-cdk/commit/08a2f363093f39d04026778bb8d5d7f673698b57)), closes [#22474](https://github.com/aws/aws-cdk/issues/22474)
* **redshift:** Tables can include comments ([#23847](https://github.com/aws/aws-cdk/issues/23847)) ([46cadd4](https://github.com/aws/aws-cdk/commit/46cadd4b2dd417e1484ba63389b33e1504cfd842)), closes [#22682](https://github.com/aws/aws-cdk/issues/22682)


### Bug Fixes

* **servicecatalogappregistry:** default stack name is not meaningful and causes conflict when multiple stacks deployed to the same account-region ([#23823](https://github.com/aws/aws-cdk/issues/23823)) ([420b5ff](https://github.com/aws/aws-cdk/commit/420b5ff2bd08311f2c8cabbe0787c0e0bf4f8ae3))

## [2.63.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.63.1-alpha.0...v2.63.2-alpha.0) (2023-02-04)

## [2.63.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.63.0-alpha.0...v2.63.1-alpha.0) (2023-02-03)

## [2.63.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.62.2-alpha.0...v2.63.0-alpha.0) (2023-01-31)


### Features

* **synthetics:** Adding DeleteLambdaResourcesOnCanaryDeletion prop to the canary L2 ([#23820](https://github.com/aws/aws-cdk/issues/23820)) ([45c191e](https://github.com/aws/aws-cdk/commit/45c191efa865e0aef6fc9d7fa4cd9d56d98a7cc9))
* **redshift:** support default role for redshift clusters ([#22551](https://github.com/aws/aws-cdk/issues/22551))

## [2.62.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.62.1-alpha.0...v2.62.2-alpha.0) (2023-01-27)

## [2.62.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.62.0-alpha.0...v2.62.1-alpha.0) (2023-01-26)

## [2.62.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.61.1-alpha.0...v2.62.0-alpha.0) (2023-01-25)


### Features

* **apprunner:** apprunner secrets manager ([#23692](https://github.com/aws/aws-cdk/issues/23692)) ([a914fc0](https://github.com/aws/aws-cdk/commit/a914fc0614cd9aa634c5724c3474c99fd3888d98))


### Bug Fixes

* **integ-runner:** cleanup tmp snapshot before running test ([#23773](https://github.com/aws/aws-cdk/issues/23773)) ([366f2ab](https://github.com/aws/aws-cdk/commit/366f2ab6fbedaf33630a40d5306746c6d363f05c))

## [2.61.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.61.0-alpha.0...v2.61.1-alpha.0) (2023-01-20)

## [2.61.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.60.0-alpha.0...v2.61.0-alpha.0) (2023-01-18)


### Features

* **cli-lib:** [JS/TS only] experimental support for programmatic CLI api ([#22836](https://github.com/aws/aws-cdk/issues/22836)) ([0b6b716](https://github.com/aws/aws-cdk/commit/0b6b7166c3f0348cc33fd3a0d19637351ea3b05b))


### Bug Fixes

* **glue:** --conf parameter is no longer a reserved keyword for glue jobs ([#23673](https://github.com/aws/aws-cdk/issues/23673)) ([3d0f4ba](https://github.com/aws/aws-cdk/commit/3d0f4ba6dd92ad7b91b00fad6cbab873964683fc))
* **servicecatalogappregistry:** outputs are not deployable ([#23652](https://github.com/aws/aws-cdk/issues/23652)) ([fa9eef0](https://github.com/aws/aws-cdk/commit/fa9eef081ead451a4d38bf083eda02af09fff482)), closes [#23641](https://github.com/aws/aws-cdk/issues/23641)

## [2.60.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.59.0-alpha.0...v2.60.0-alpha.0) (2023-01-11)


### Features

* **gamelift:** add MatchmakingConfiguration L2 Construct for GameLift ([#23326](https://github.com/aws/aws-cdk/issues/23326)) ([9b2573b](https://github.com/aws/aws-cdk/commit/9b2573b32e8535d3db21f07647f099c9e01eb292))
* **integ-runner:** support `--language` presets for JavaScript, TypeScript, Python and Go ([#22058](https://github.com/aws/aws-cdk/issues/22058)) ([22673b2](https://github.com/aws/aws-cdk/commit/22673b2ea40c13b6c10a2c7c628ce5cc534f5840)), closes [#21169](https://github.com/aws/aws-cdk/issues/21169)

## [2.59.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.58.1-alpha.0...v2.59.0-alpha.0) (2023-01-03)

## [2.58.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.58.0-alpha.0...v2.58.1-alpha.0) (2022-12-30)

## [2.58.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.57.0-alpha.0...v2.58.0-alpha.0) (2022-12-28)

## [2.57.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.56.1-alpha.0...v2.57.0-alpha.0) (2022-12-27)


### Bug Fixes

* **aws-redshift:** Columns are not dropped on removal from array ([#23011](https://github.com/aws/aws-cdk/issues/23011)) ([2981313](https://github.com/aws/aws-cdk/commit/298131312b513c0e73865e6fff74c189ee99e328)), closes [#22208](https://github.com/aws/aws-cdk/issues/22208)

## [2.56.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.56.0-alpha.0...v2.56.1-alpha.0) (2022-12-23)

## [2.56.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.55.1-alpha.0...v2.56.0-alpha.0) (2022-12-21)


### Features

* **integ-tests:** add serializedJson on match utility ([#23218](https://github.com/aws/aws-cdk/issues/23218)) ([1a62dc4](https://github.com/aws/aws-cdk/commit/1a62dc4590d725d3c03861af434a24789eaa0a2e))
* **servicecatalogappregistry:** Cross region warning and default application tag ([#23412](https://github.com/aws/aws-cdk/issues/23412)) ([8d359ae](https://github.com/aws/aws-cdk/commit/8d359ae35877ce066e419f7e2e7da2b0deb587e6))

## [2.55.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.55.0-alpha.0...v2.55.1-alpha.0) (2022-12-16)

## [2.55.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.54.0-alpha.0...v2.55.0-alpha.0) (2022-12-14)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **appsync:** `DataSource.createResolver`,
`DataSource.createFunction`, and `GraphQlApi.createResolver` now require
2 arguments instead of 1.
* **appsync:** Renames `Schema` to `SchemaFile` that implements `ISchema`. Removes all `addXxx` type methods from `GraphQlApi`.

### Features

* **aws-lambda-python:** add command hooks for bundling to allow for execution of custom commands in the build container ([#23330](https://github.com/aws/aws-cdk/issues/23330)) ([d3d071f](https://github.com/aws/aws-cdk/commit/d3d071f51fab61ae0e484f143e68e698bba48537))
* **gamelift:** add GameSessionQueue L2 Construct for GameLift ([#23266](https://github.com/aws/aws-cdk/issues/23266)) ([1ded644](https://github.com/aws/aws-cdk/commit/1ded64430d8258f6666743e245ef5ac31ed4bf0b))


### Bug Fixes

* **appsync:** unexpected resolver replacement ([#23322](https://github.com/aws/aws-cdk/issues/23322)) ([6dc15d4](https://github.com/aws/aws-cdk/commit/6dc15d40764dc71fe6a3b70691f586e96cdcf730)), closes [#13269](https://github.com/aws/aws-cdk/issues/13269)
* **servicecatalogappregistry:** synth error when associating a nested stack ([#23248](https://github.com/aws/aws-cdk/issues/23248)) ([30301d9](https://github.com/aws/aws-cdk/commit/30301d9e5ab4af86c5e48a5ad47013924acdfed7))


### Miscellaneous Chores

* **appsync:** removes codefirst schema generation ([#23250](https://github.com/aws/aws-cdk/issues/23250)) ([2bd1e41](https://github.com/aws/aws-cdk/commit/2bd1e4184aea054766f7872b300b960b2b83ef06))

## [2.54.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.53.0-alpha.0...v2.54.0-alpha.0) (2022-12-07)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **servicecatalogappregistry:** Stack inside ApplicationAssociator is no longer is created inside ApplicationAssociator Construct scope. The stack will now get created inside cdk.App scope.
* ** servicecatalogappregistry:** stackId  will no longer have ApplicationAssociator Construct scope.

### All Submissions:

* [ X] Have you followed the guidelines in our [Contributing guide?](https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md)

### Adding new Unconventional Dependencies:

* [ ] This PR adds new unconventional dependencies following the process described [here](https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md/#adding-new-unconventional-dependencies)

### New Features

* [ ] Have you added the new feature to an [integration test](https://github.com/aws/aws-cdk/blob/main/INTEGRATION_TESTS.md)?
	* [ ] Did you use `yarn integ` to deploy the infrastructure and generate the snapshot (i.e. `yarn integ` without `--dry-run`)?

*By submitting this pull request, I confirm that my contribution is made under the terms of the Apache-2.0 license*

### Features

* **gamelift:** add Alias L2 Construct for GameLift ([#23042](https://github.com/aws/aws-cdk/issues/23042)) ([49d5c3a](https://github.com/aws/aws-cdk/commit/49d5c3a21ae1fa15bf1be4c6b81194800b016372))
* **gamelift:** add MatchmakingRuleSet L2 Construct for GameLift ([#23091](https://github.com/aws/aws-cdk/issues/23091)) ([ad8a704](https://github.com/aws/aws-cdk/commit/ad8a704cce7c09bf51f6ee4d688d00fcb2c86472))
* **gamelift:** add support for buildArn output attribute ([#23070](https://github.com/aws/aws-cdk/issues/23070)) ([08f2995](https://github.com/aws/aws-cdk/commit/08f2995784cdc0fd43ec10af534c49a8466b5351))
* **glue:** support glue version 4.0 ([#23223](https://github.com/aws/aws-cdk/issues/23223)) ([fe08aa9](https://github.com/aws/aws-cdk/commit/fe08aa900103f93ca5ea4c3fc3cdc6b31d4b52d9)), closes [#23220](https://github.com/aws/aws-cdk/issues/23220)
* **lambda-go:** allow configuration of GOPROXY ([#23171](https://github.com/aws/aws-cdk/issues/23171)) ([d189161](https://github.com/aws/aws-cdk/commit/d189161964f7169f1c0918cdec0fca9cacec4d61))
* **location:** `PlaceIndex` ([#22853](https://github.com/aws/aws-cdk/issues/22853)) ([50422df](https://github.com/aws/aws-cdk/commit/50422df24f00b10a5487aa56bdf7220846ebbeaa))
* **sagemaker:** add Endpoint L2 construct ([#22886](https://github.com/aws/aws-cdk/issues/22886)) ([bf7586b](https://github.com/aws/aws-cdk/commit/bf7586b16a6f7706d8d7da3a6e0aed955f159e15)), closes [#2809](https://github.com/aws/aws-cdk/issues/2809)


### Bug Fixes

* **appsync:** fully qualify service principal ([#23054](https://github.com/aws/aws-cdk/issues/23054)) ([0bfce89](https://github.com/aws/aws-cdk/commit/0bfce8965ee50ab79054e6f5a4c6bbecb0955e19))
* **servicecatalogappregistry:** creating ApplicationStack in AppScope to give user more control over the passed stackId ([#22977](https://github.com/aws/aws-cdk/issues/22977)) ([85fe047](https://github.com/aws/aws-cdk/commit/85fe047a94494794afc1ef6c1c788219af3eb0cb)), closes [#22973](https://github.com/aws/aws-cdk/issues/22973)

## [2.53.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.52.1-alpha.0...v2.53.0-alpha.0) (2022-11-29)

## [2.52.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.52.0-alpha.0...v2.52.1-alpha.0) (2022-11-28)


### Bug Fixes

* **appsync:** fully qualify service principal ([#23054](https://github.com/aws/aws-cdk/issues/23054)) ([d7141dd](https://github.com/aws/aws-cdk/commit/d7141dd7318fd6ee45206dfb35553e4a528decf9))

## [2.52.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.51.1-alpha.0...v2.52.0-alpha.0) (2022-11-27)

## [2.51.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.51.0-alpha.0...v2.51.1-alpha.0) (2022-11-18)

## [2.51.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.50.0-alpha.0...v2.51.0-alpha.0) (2022-11-18)


### Features

* **gamelift:** add BuildFleet L2 Construct for GameLift ([#22835](https://github.com/aws/aws-cdk/issues/22835)) ([834fab4](https://github.com/aws/aws-cdk/commit/834fab4983526eced3ddbdd58c5bfefbe757715d))
* **gamelift:** add GameServerGroup L2 Construct for GameLift ([#22762](https://github.com/aws/aws-cdk/issues/22762)) ([ef74116](https://github.com/aws/aws-cdk/commit/ef74116aad56abbcea788ef3662a6ae6e74c4145))
* **integ-runner:** support config file ([#22937](https://github.com/aws/aws-cdk/issues/22937)) ([4f49efe](https://github.com/aws/aws-cdk/commit/4f49efe35eaffd662c22e0d80e9b7fadeb25ab37))
* **integ-runner:** support custom `--app` commands ([#22761](https://github.com/aws/aws-cdk/issues/22761)) ([a7bb6e1](https://github.com/aws/aws-cdk/commit/a7bb6e1a8a9a9f3fe534069ec77b4f6b10307c9f)), closes [#22521](https://github.com/aws/aws-cdk/issues/22521)
* **integ-runner:** support custom `--test-regex` to match integ test files ([#22786](https://github.com/aws/aws-cdk/issues/22786)) ([fa1a439](https://github.com/aws/aws-cdk/commit/fa1a4395230790c89d5c468306759f4a6c5f7e0c)), closes [#22761](https://github.com/aws/aws-cdk/issues/22761) [#22521](https://github.com/aws/aws-cdk/issues/22521)
* **integ-runner:** support snapshot diff on nested stacks ([#22881](https://github.com/aws/aws-cdk/issues/22881)) ([5b3d06d](https://github.com/aws/aws-cdk/commit/5b3d06d808d1eb110943b2c68de75ae6d5b5e624))
* **sagemaker:** add EndpointConfig L2 construct ([#22865](https://github.com/aws/aws-cdk/issues/22865)) ([0e97c15](https://github.com/aws/aws-cdk/commit/0e97c15b49d02b44ea4916a3f29a156ca69a5a23)), closes [#2809](https://github.com/aws/aws-cdk/issues/2809)
* **sagemaker:** add Model L2 construct ([#22549](https://github.com/aws/aws-cdk/issues/22549)) ([93915f1](https://github.com/aws/aws-cdk/commit/93915f113e656de8374c99e42135f698f0877685)), closes [#2809](https://github.com/aws/aws-cdk/issues/2809)


### Bug Fixes

* **gamelift:** restrict policy to access Script / Build content in S3 ([#22767](https://github.com/aws/aws-cdk/issues/22767)) ([c936002](https://github.com/aws/aws-cdk/commit/c93600260c86dfbc6a8f8f2399a6ac9d0f4b4d35))

## [2.50.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.49.1-alpha.0...v2.50.0-alpha.0) (2022-11-01)


### Features

* **synthetics:** aws synthetics runtime version syn-nodejs-puppeteer-3.8 ([#22707](https://github.com/aws/aws-cdk/issues/22707)) ([228c865](https://github.com/aws/aws-cdk/commit/228c86532118b143e365b2268d06ee3a36fcf3a7))

## [2.49.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.49.0-alpha.0...v2.49.1-alpha.0) (2022-10-31)

## [2.49.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.48.0-alpha.0...v2.49.0-alpha.0) (2022-10-27)

## [2.48.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.47.0-alpha.0...v2.48.0-alpha.0) (2022-10-27)


### Features

* **synthetics:** runtime version syn-nodejs-puppeteer-3.7 ([#22610](https://github.com/aws/aws-cdk/issues/22610)) ([326637c](https://github.com/aws/aws-cdk/commit/326637c2879657bfac33b5cc60dced7471abf7c8))

## [2.47.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.46.0-alpha.0...v2.47.0-alpha.0) (2022-10-20)


### Features

* **redshift:** support enhanced vpc routing when creating redshift cluster ([#22499](https://github.com/aws/aws-cdk/issues/22499)) ([e2b18e7](https://github.com/aws/aws-cdk/commit/e2b18e7b47eb7a87ae37356a9719c055e58e6e6c))


### Bug Fixes

* **integ-runner:** Fix call to spawnSync for hooks commands ([#22429](https://github.com/aws/aws-cdk/issues/22429)) ([9139ca9](https://github.com/aws/aws-cdk/commit/9139ca96ffc010e13393aff927d7b7eacfbae4f9)), closes [#22344](https://github.com/aws/aws-cdk/issues/22344)
* **lambda-python:** root-owned cache items not cleaned up after install ([#22512](https://github.com/aws/aws-cdk/issues/22512)) ([5ef65e0](https://github.com/aws/aws-cdk/commit/5ef65e042c747bedf9d770b47e540393454762f2)), closes [#22398](https://github.com/aws/aws-cdk/issues/22398)

## [2.46.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.45.0-alpha.0...v2.46.0-alpha.0) (2022-10-13)


### Features

* **integ-tests:** add ability to `wait` for assertions to pass ([#22335](https://github.com/aws/aws-cdk/issues/22335)) ([700f9c4](https://github.com/aws/aws-cdk/commit/700f9c4d465684b784f50ec74e897c9031a639c5))
* **integ-tests:** allow for user provided assertions stack ([#22404](https://github.com/aws/aws-cdk/issues/22404)) ([39089f5](https://github.com/aws/aws-cdk/commit/39089f5eabc61c2a546391742ff2cf906f8e0f8b))
* **synthetics:** new runtime synthetics nodejs puppeteer 3 6 ([#22374](https://github.com/aws/aws-cdk/issues/22374)) ([e0c0b56](https://github.com/aws/aws-cdk/commit/e0c0b56dded26a897dc6243298947bd4e69321b2))


### Bug Fixes

* **appsync:** can not use Tokens in the name of a DataSource ([#22378](https://github.com/aws/aws-cdk/issues/22378)) ([511eb79](https://github.com/aws/aws-cdk/commit/511eb79cba734bcd9e013d5dfbf262c75a522f09)), closes [#18900](https://github.com/aws/aws-cdk/issues/18900)
* **aws-lambda-python:** export poetry dependencies without hashes ([#22351](https://github.com/aws/aws-cdk/issues/22351)) ([76482f6](https://github.com/aws/aws-cdk/commit/76482f6847a46806c1a309d2f9335a3d6e838fc6)), closes [#14201](https://github.com/aws/aws-cdk/issues/14201) [#19232](https://github.com/aws/aws-cdk/issues/19232)
* **lambda-python:** commands run non-sequentially on Graviton when building container image ([#22398](https://github.com/aws/aws-cdk/issues/22398)) ([e427fd6](https://github.com/aws/aws-cdk/commit/e427fd6f4a186784e345b8f88424d74c004f1e5a)), closes [#22012](https://github.com/aws/aws-cdk/issues/22012)

## [2.45.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.44.0-alpha.0...v2.45.0-alpha.0) (2022-10-06)


### Features

* **gamelift:** add Build L2 constructs for GameLift ([#22313](https://github.com/aws/aws-cdk/issues/22313)) ([983d26e](https://github.com/aws/aws-cdk/commit/983d26e4e7cbb40fe1148ec635efe8093d850835))
* **gamelift:** add Script L2 Construct for GameLift ([#22343](https://github.com/aws/aws-cdk/issues/22343)) ([da181ba](https://github.com/aws/aws-cdk/commit/da181bac2a7fee2cad8915006d4501074fcb04d4))
* **neptune:** enable cloudwatch logs exports ([#22004](https://github.com/aws/aws-cdk/issues/22004)) ([2b2bb01](https://github.com/aws/aws-cdk/commit/2b2bb01dbe00c79e7f5a0513a2e1f76f6cdcbc11)), closes [#20248](https://github.com/aws/aws-cdk/issues/20248) [#15888](https://github.com/aws/aws-cdk/issues/15888)
* **servicecatalogappregistry:** application-associator L2 Construct ([#22024](https://github.com/aws/aws-cdk/issues/22024)) ([a2b7a46](https://github.com/aws/aws-cdk/commit/a2b7a4624638a458bfb6e8e09c67a77e48e1d167))

## [2.44.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.43.1-alpha.0...v2.44.0-alpha.0) (2022-09-28)


### Features

* **integ-tests:** chain assertion api calls ([#22196](https://github.com/aws/aws-cdk/issues/22196)) ([530e07b](https://github.com/aws/aws-cdk/commit/530e07bdc87ab94bbd5ed28debac98400a8152cc))
* **neptune:** introduce metric method to cluster and instance ([#21995](https://github.com/aws/aws-cdk/issues/21995)) ([02ed837](https://github.com/aws/aws-cdk/commit/02ed8371276d504ba9fe09687d45409ad7cca288)), closes [#20248](https://github.com/aws/aws-cdk/issues/20248)

## [2.43.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.43.0-alpha.0...v2.43.1-alpha.0) (2022-09-23)

## [2.43.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.42.1-alpha.0...v2.43.0-alpha.0) (2022-09-21)


### Bug Fixes

* **integ-tests:** AwsApiCall Custom Resource length could be greater than 60 characters ([#22119](https://github.com/aws/aws-cdk/issues/22119)) ([35b2806](https://github.com/aws/aws-cdk/commit/35b280616a420987b6553f73bc91a736b06d4e1a))
* **integ-tests:** can't enable lookups when creating an IntegTest ([#22075](https://github.com/aws/aws-cdk/issues/22075)) ([d0e0ab9](https://github.com/aws/aws-cdk/commit/d0e0ab9d3744372edd56aa984daac4de26272673))

## [2.42.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.42.0-alpha.0...v2.42.1-alpha.0) (2022-09-19)

## [2.42.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.41.0-alpha.0...v2.42.0-alpha.0) (2022-09-15)


### Features

* **neptune:** add engine version 1.2.0.0 ([#21908](https://github.com/aws/aws-cdk/issues/21908)) ([be65da6](https://github.com/aws/aws-cdk/commit/be65da6ec1ab9c82d04f662a69bd1ae1147dff25)), closes [#21877](https://github.com/aws/aws-cdk/issues/21877)
* **neptune:** introduce cluster grant method for granular actions ([#21926](https://github.com/aws/aws-cdk/issues/21926)) ([42e559d](https://github.com/aws/aws-cdk/commit/42e559d49e9fdb43f37a0b53ef5a85cb6bc5f36d)), closes [#21877](https://github.com/aws/aws-cdk/issues/21877)


### Bug Fixes

* **lambda-python:** bundling artifacts are written to the entry path ([#21967](https://github.com/aws/aws-cdk/issues/21967)) ([bc4427c](https://github.com/aws/aws-cdk/commit/bc4427cc874e7eea7cfba5f88d536a805d771bc6)), closes [#19231](https://github.com/aws/aws-cdk/issues/19231)

## [2.41.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.40.0-alpha.0...v2.41.0-alpha.0) (2022-09-07)


### Features

* **batch:** add propagate tags prop in job definition ([#21904](https://github.com/aws/aws-cdk/issues/21904)) ([1bc4526](https://github.com/aws/aws-cdk/commit/1bc4526261c2fbdd6ce6c371ba1d9da2f79e07bd)), closes [#21740](https://github.com/aws/aws-cdk/issues/21740)


### Bug Fixes

* **lambda-python:** bundling with poetry is broken ([#21945](https://github.com/aws/aws-cdk/issues/21945)) ([4b37157](https://github.com/aws/aws-cdk/commit/4b37157b47ab38124b62649649d0df9b701cb7fe)), closes [#21867](https://github.com/aws/aws-cdk/issues/21867)
* **lambda-python:** poetry bundling fails on python3.7 ([#21950](https://github.com/aws/aws-cdk/issues/21950)) ([809e1b0](https://github.com/aws/aws-cdk/commit/809e1b0d5dc29be02f95ea4361b6f87f94325f3d))

## [2.40.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.39.1-alpha.0...v2.40.0-alpha.0) (2022-08-31)


### Features

* **glue:** Added value to PythonVersion enum ([#21670](https://github.com/aws/aws-cdk/issues/21670)) ([9774d4c](https://github.com/aws/aws-cdk/commit/9774d4ce11287d91278290369dc783a83d784fdf)), closes [#21568](https://github.com/aws/aws-cdk/issues/21568) [/github.com/aws/aws-cdk/issues/21568#issuecomment-1219668861](https://github.com/aws//github.com/aws/aws-cdk/issues/21568/issues/issuecomment-1219668861)
* **msk:** added msk cluster sasl iam property ([#21798](https://github.com/aws/aws-cdk/issues/21798)) ([d30a530](https://github.com/aws/aws-cdk/commit/d30a530a68d97ac455125bf4a2154a31adcb9582))


### Bug Fixes

* **integ-runner:** array arguments aren't recognizing multiple options ([#21763](https://github.com/aws/aws-cdk/issues/21763)) ([d942324](https://github.com/aws/aws-cdk/commit/d942324cef7646397f9359dfb91819ded72874b0)), closes [#20384](https://github.com/aws/aws-cdk/issues/20384)

## [2.39.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.39.0-alpha.0...v2.39.1-alpha.0) (2022-08-29)


### Bug Fixes

* **python:** NameError name 'SubnetSelection' is not defined ([#21790](https://github.com/aws/aws-cdk/issues/21790)) ([eaaba39](https://github.com/aws/aws-cdk/commit/eaaba39e21f8b76dfa01cb5515a25d8600e73eee)), closes [#21790](https://github.com/aws/aws-cdk/issues/21790)

## [2.39.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.38.1-alpha.0...v2.39.0-alpha.0) (2022-08-25)


### Features

* **servicecatalogappregistry:** add sharing of applications and attribute groups ([#20850](https://github.com/aws/aws-cdk/issues/20850)) ([cf3bb6e](https://github.com/aws/aws-cdk/commit/cf3bb6e9ced5e3d18e782e7144858078c70cdcf9))


### Bug Fixes

* **aws-batch:** Support omitting ComputeEnvironment security groups so that they can be specified in Launch Template ([#21579](https://github.com/aws/aws-cdk/issues/21579)) ([33b00dd](https://github.com/aws/aws-cdk/commit/33b00dd063bf690bef4a91a91b468ba4a8a8531e)), closes [#21577](https://github.com/aws/aws-cdk/issues/21577)
* **integ-runner:** ignoring asset changes doesn't work with new style assets ([#21638](https://github.com/aws/aws-cdk/issues/21638)) ([7857f55](https://github.com/aws/aws-cdk/commit/7857f55e8e7748920f8c97b08c13a04b9c8598ab))
* **integ-tests:** assertions stack not deployed on v2 ([#21646](https://github.com/aws/aws-cdk/issues/21646)) ([ee1b66d](https://github.com/aws/aws-cdk/commit/ee1b66d1c9de6fcd284ee359db3ab232084fe6c7)), closes [#21639](https://github.com/aws/aws-cdk/issues/21639)

## [2.38.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.38.0-alpha.0...v2.38.1-alpha.0) (2022-08-18)

## [2.38.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.37.1-alpha.0...v2.38.0-alpha.0) (2022-08-17)


### Features

* **appsync:** expose the AppSyncDomain of the custom domain of an AppSync api ([#21554](https://github.com/aws/aws-cdk/issues/21554)) ([d1097b5](https://github.com/aws/aws-cdk/commit/d1097b5199727b3de6c98850f8efe0a9fae53706))

## [2.37.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.37.0-alpha.0...v2.37.1-alpha.0) (2022-08-10)

## [2.37.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.36.0-alpha.0...v2.37.0-alpha.0) (2022-08-09)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **redshift:** The way to specify a logging bucket and prefix will change to use an interface.

### Features

* **apigatewayv2:** WebSocket API - IAM authorizer support ([#21393](https://github.com/aws/aws-cdk/issues/21393)) ([a1a6e6c](https://github.com/aws/aws-cdk/commit/a1a6e6cf2e03110322ea39e806d3d8206b609843))
* **appsync:** allow user to configure log retention time ([#21418](https://github.com/aws/aws-cdk/issues/21418)) ([a2bb263](https://github.com/aws/aws-cdk/commit/a2bb263ec40c842dc332f2a55d494849665d38ba)), closes [#20536](https://github.com/aws/aws-cdk/issues/20536)
* **batch:** ComputeEnvironment implements IConnectable ([#21458](https://github.com/aws/aws-cdk/issues/21458)) ([4bc9651](https://github.com/aws/aws-cdk/commit/4bc965102f632eae7314cfadf9c7310dadaf2782)), closes [#20983](https://github.com/aws/aws-cdk/issues/20983)
* **integ-runner:** add option to show deployment output ([#21466](https://github.com/aws/aws-cdk/issues/21466)) ([289fb96](https://github.com/aws/aws-cdk/commit/289fb96810ba8c2dd4d58dad06401c10eeddd45c))
* **iot-actions:** add support for DynamoDBv2 rule ([#20171](https://github.com/aws/aws-cdk/issues/20171)) ([a57dec3](https://github.com/aws/aws-cdk/commit/a57dec3db30ef71511862f7afff21b28e59fe5ad)), closes [#20162](https://github.com/aws/aws-cdk/issues/20162)
* **iot-actions:** support for sending messages to iot-events ([#19953](https://github.com/aws/aws-cdk/issues/19953)) ([35fc169](https://github.com/aws/aws-cdk/commit/35fc169ea1743ef2227e210075c95ad7c469f6d7))
* **iotevents:** support timer actions ([#19949](https://github.com/aws/aws-cdk/issues/19949)) ([af301dd](https://github.com/aws/aws-cdk/commit/af301ddcaea71be7b90d9d6ac1c903dfaeb10f60))


### Bug Fixes

* **redshift:** deploy fails when creating logging bucket without s3 key ([#21243](https://github.com/aws/aws-cdk/issues/21243)) ([220177f](https://github.com/aws/aws-cdk/commit/220177fdb7aecc2764ffca31c48004fd54275e3a))

## [2.36.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.35.0-alpha.0...v2.36.0-alpha.0) (2022-08-08)

## [2.35.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.34.2-alpha.0...v2.35.0-alpha.0) (2022-08-02)


### Bug Fixes

* **cognito-identitypool:** providerUrl causes error when mappingKey is not provided and it is a token  ([#21191](https://github.com/aws/aws-cdk/issues/21191)) ([d91c904](https://github.com/aws/aws-cdk/commit/d91c9045b2ca027947c94ff8b93adb80f8ca8434)), closes [#19222](https://github.com/aws/aws-cdk/issues/19222) [/github.com/aws/aws-cdk/pull/21056#issuecomment-1178879318](https://github.com/aws//github.com/aws/aws-cdk/pull/21056/issues/issuecomment-1178879318)

## [2.34.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.34.1-alpha.0...v2.34.2-alpha.0) (2022-07-29)

## [2.34.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.34.0-alpha.0...v2.34.1-alpha.0) (2022-07-29)

### Bug Fixes

* Revert to `jsii-pacmak@1.62.0` as dynamic runtime type-checking it introduced for Python results in incorrect code being produced.


## [2.34.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.33.0-alpha.0...v2.34.0-alpha.0) (2022-07-28)


### Features

* **appsync:** support for read consistency during DynamoDB reads ([#20793](https://github.com/aws/aws-cdk/issues/20793)) ([0b911ef](https://github.com/aws/aws-cdk/commit/0b911efd75c02bb6117d6e32c0112f58da5192b7))
* **batch:** add default AWS_ACCOUNT and AWS_REGION to Batch container, if they are not explicitly set ([#21041](https://github.com/aws/aws-cdk/issues/21041)) ([eed854e](https://github.com/aws/aws-cdk/commit/eed854ed4be6b76abc909721d6baa14140681dcc))

## [2.33.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.32.1-alpha.0...v2.33.0-alpha.0) (2022-07-19)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **cloud9:** The imageId parameter is now required and deployments will fail without it

### Features

* **cloud9:** support imageid when creating cloud9 environment ([#21194](https://github.com/aws/aws-cdk/issues/21194)) ([dcf3eb3](https://github.com/aws/aws-cdk/commit/dcf3eb3ab65eb84c59b61fb08b6436d94c55d7e5))
* **redshift:** adds elasticIp parameter to redshift cluster ([#21085](https://github.com/aws/aws-cdk/issues/21085)) ([c88030f](https://github.com/aws/aws-cdk/commit/c88030f39b38965f33d221f1bb28331a3277ae96)), closes [#19191](https://github.com/aws/aws-cdk/issues/19191)

## [2.32.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.32.0-alpha.0...v2.32.1-alpha.0) (2022-07-15)

## [2.32.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.31.2-alpha.0...v2.32.0-alpha.0) (2022-07-14)


### Features

* **appsync:** set max batch size when using batch invoke  ([#20995](https://github.com/aws/aws-cdk/issues/20995)) ([69d25a6](https://github.com/aws/aws-cdk/commit/69d25a6f26f03c6589b350803431de23fe598ae0)), closes [#20467](https://github.com/aws/aws-cdk/issues/20467)
* **batch:** add launchTemplateId in LaunchTemplateSpecification ([#20184](https://github.com/aws/aws-cdk/issues/20184)) ([269b8d0](https://github.com/aws/aws-cdk/commit/269b8d0ca737a1bad6736a2d5ed234602cd8f469)), closes [#20163](https://github.com/aws/aws-cdk/issues/20163)
* **glue:** enable partition filtering on tables ([#21081](https://github.com/aws/aws-cdk/issues/21081)) ([bf35048](https://github.com/aws/aws-cdk/commit/bf35048cc5f907c7226f60aa8b3b4b8b500d2bc0)), closes [#20825](https://github.com/aws/aws-cdk/issues/20825)
* **integ-tests:** expose adding IAM policies to the assertion provider ([#20769](https://github.com/aws/aws-cdk/issues/20769)) ([c2f40b7](https://github.com/aws/aws-cdk/commit/c2f40b7c97b822f258f953b572ba2e7a99403f89))
* **neptune:** add engine version 1.1.1.0 ([#21079](https://github.com/aws/aws-cdk/issues/21079)) ([a113816](https://github.com/aws/aws-cdk/commit/a1138161ca295ad4a81fe32b51beb82438653144)), closes [#20869](https://github.com/aws/aws-cdk/issues/20869)
* **redshift:** adds classic or elastic resize type option ([#21084](https://github.com/aws/aws-cdk/issues/21084)) ([b5e9c1a](https://github.com/aws/aws-cdk/commit/b5e9c1a99be6898c544f91781ceb4ee1d371a03e)), closes [#19430](https://github.com/aws/aws-cdk/issues/19430)


### Bug Fixes

* **appsync:** domain name api association fails when domain name creation is in the same stack ([#20173](https://github.com/aws/aws-cdk/issues/20173)) ([c1495f0](https://github.com/aws/aws-cdk/commit/c1495f0b700cedc04b556844397048ee41a7d891)), closes [#18395](https://github.com/aws/aws-cdk/issues/18395)
* **integ-runner:** test names change depending on the discovery directory ([#21093](https://github.com/aws/aws-cdk/issues/21093)) ([d38f78c](https://github.com/aws/aws-cdk/commit/d38f78c3fc9ba37b3a1033dabe89cd60dfd81b8f))

## [2.31.2-alpha.0](https://github.com/aws/aws-cdk/compare/v2.31.1-alpha.0...v2.31.2-alpha.0) (2022-07-13)

## [2.31.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.31.0-alpha.0...v2.31.1-alpha.0) (2022-07-08)

## [2.31.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.30.0-alpha.0...v2.31.0-alpha.0) (2022-07-06)


### Features

* **batch:** add secrets props to job definition ([#20871](https://github.com/aws/aws-cdk/issues/20871)) ([9b1051f](https://github.com/aws/aws-cdk/commit/9b1051f86abdfa6448b14cdae8e1ef9acb1e6688)), closes [#19506](https://github.com/aws/aws-cdk/issues/19506) [#10976](https://github.com/aws/aws-cdk/issues/10976)

## [2.30.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.29.1-alpha.0...v2.30.0-alpha.0) (2022-07-01)

## [2.29.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.29.0-alpha.0...v2.29.1-alpha.0) (2022-06-24)

## [2.29.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.28.1-alpha.0...v2.29.0-alpha.0) (2022-06-22)

## [2.28.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.28.0-alpha.0...v2.28.1-alpha.0) (2022-06-15)

## [2.28.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.27.0-alpha.0...v2.28.0-alpha.0) (2022-06-14)


### Bug Fixes

* **appsync:** Create Lambda permission when using Lambda Authorizer(#… ([#20641](https://github.com/aws/aws-cdk/issues/20641)) ([6176400](https://github.com/aws/aws-cdk/commit/61764009648a4602ffa403adda903442c48c45df)), closes [#20234](https://github.com/aws/aws-cdk/issues/20234)
* **integ-runner:** don't allow new legacy tests ([#20614](https://github.com/aws/aws-cdk/issues/20614)) ([c946615](https://github.com/aws/aws-cdk/commit/c94661508e2a97b52e9284ba4093d9864d2d5f0b))

## [2.27.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.26.0-alpha.0...v2.27.0-alpha.0) (2022-06-02)


### Features

* **integ-runner:** publish integ-runner cli ([#20477](https://github.com/aws/aws-cdk/issues/20477)) ([7779531](https://github.com/aws/aws-cdk/commit/777953106ac550b058fdaa3ccde25b62be07defa))


### Bug Fixes

* **integ-runner:** catch snapshot errors, treat `--from-file` as command-line ([#20523](https://github.com/aws/aws-cdk/issues/20523)) ([cedfde8](https://github.com/aws/aws-cdk/commit/cedfde8cb07eb879ee384bda93bba813ede91699))
* **integ-runner:** don't throw error if tests pass ([#20511](https://github.com/aws/aws-cdk/issues/20511)) ([c274c2f](https://github.com/aws/aws-cdk/commit/c274c2f983de2dfd20ed2886a3c50f7fd3f6b3f4)), closes [#20384](https://github.com/aws/aws-cdk/issues/20384)

## [2.26.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.25.0-alpha.0...v2.26.0-alpha.0) (2022-05-27)


### Features

* **apprunner:** VpcConnector construct ([#20471](https://github.com/aws/aws-cdk/issues/20471)) ([5052191](https://github.com/aws/aws-cdk/commit/50521911f22f433323d700db77530e883762138a))


### Bug Fixes

* **integ-runner:** always resynth on deploy ([#20508](https://github.com/aws/aws-cdk/issues/20508)) ([7138057](https://github.com/aws/aws-cdk/commit/71380571b878a50fe4b754c7dac78da075a98242))
* **integ-tests:** DeployAssert should be private ([#20466](https://github.com/aws/aws-cdk/issues/20466)) ([0f52813](https://github.com/aws/aws-cdk/commit/0f52813bcf6a48c352f697004a899461dd06935d))

## [2.25.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.24.1-alpha.0...v2.25.0-alpha.0) (2022-05-20)


### Features

* **cloud9:** configure Connection Type of Ec2Environment ([#20250](https://github.com/aws/aws-cdk/issues/20250)) ([01708bc](https://github.com/aws/aws-cdk/commit/01708bc7cf842eab7e1d1fc58bf42e4724624c0a)), closes [#17027](https://github.com/aws/aws-cdk/issues/17027)
* **integ-tests:** enhancements to integ-tests ([#20180](https://github.com/aws/aws-cdk/issues/20180)) ([3ff3fb7](https://github.com/aws/aws-cdk/commit/3ff3fb7c5ec9636022b3046036376c09a3166fb0))


### Bug Fixes

* **amplify:** custom headers break with tokens ([#20395](https://github.com/aws/aws-cdk/issues/20395)) ([765f441](https://github.com/aws/aws-cdk/commit/765f44177298b645c88a29587b52619e91a8757c))

## [2.24.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.24.0-alpha.0...v2.24.1-alpha.0) (2022-05-12)

## [2.24.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.23.0-alpha.0...v2.24.0-alpha.0) (2022-05-11)


### Bug Fixes

* **appsync:** incorrect region used for imported Cognito user pool ([#20193](https://github.com/aws/aws-cdk/issues/20193)) ([3e0393e](https://github.com/aws/aws-cdk/commit/3e0393e63e84d631545734425482deae687520f1)), closes [#20195](https://github.com/aws/aws-cdk/issues/20195)

## [2.23.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.22.0-alpha.0...v2.23.0-alpha.0) (2022-05-04)


### Features

* **redshift:** expose user.secret as property ([#17520](https://github.com/aws/aws-cdk/issues/17520)) ([#20078](https://github.com/aws/aws-cdk/issues/20078)) ([8da006a](https://github.com/aws/aws-cdk/commit/8da006ab551213ecbdb6dc26860fe90c1d2e95e2))


### Bug Fixes

* **integ-runner:** disable-update-workflow default is 'false' instead of false ([#20073](https://github.com/aws/aws-cdk/issues/20073)) ([9f7aa65](https://github.com/aws/aws-cdk/commit/9f7aa654ab92c16743b015f7121a3dc542a7e01a))
* **integ-runner:** only diff registered stacks ([#20100](https://github.com/aws/aws-cdk/issues/20100)) ([721bd4b](https://github.com/aws/aws-cdk/commit/721bd4b24de8e410fd9181eff7e5431c13bad208))

## [2.22.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.21.1-alpha.0...v2.22.0-alpha.0) (2022-04-27)


### Features

* **integ-tests:** add `IntegTest` to group test cases ([#20015](https://github.com/aws/aws-cdk/issues/20015)) ([b4f8d91](https://github.com/aws/aws-cdk/commit/b4f8d91318087135c5549c22b43a1e679d70b3ca))
* **integ-tests:** make assertions on deployed infrastructure ([#20071](https://github.com/aws/aws-cdk/issues/20071)) ([8362efe](https://github.com/aws/aws-cdk/commit/8362efe8f1951289236034161d7560f20975b0ec))


### Bug Fixes

* **lambda-python:** handler path is incorrectly generated when using PythonFunction ([#20083](https://github.com/aws/aws-cdk/issues/20083)) ([6787376](https://github.com/aws/aws-cdk/commit/678737607cea769109aa8315520a71bc47eb50ef))
* **lambda-python:** Pipenv projects no longer work for Python 3.6 ([#20019](https://github.com/aws/aws-cdk/issues/20019)) ([c5dcdeb](https://github.com/aws/aws-cdk/commit/c5dcdeb2742fc8f0d41a7211d74934e20a7442c2))
* **lambda-python:** Pipenv projects no longer work for Python 3.6 ([#20019](https://github.com/aws/aws-cdk/issues/20019)) ([5024021](https://github.com/aws/aws-cdk/commit/5024021bec9952ca7b1e3d82e2c257f124c6300c))

## [2.21.1-alpha.0](https://github.com/aws/aws-cdk/compare/v2.21.0-alpha.0...v2.21.1-alpha.0) (2022-04-22)

## [2.21.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.20.0-alpha.0...v2.21.0-alpha.0) (2022-04-22)


### Features

* **apigatewayv2:** set throttling on stages ([#19776](https://github.com/aws/aws-cdk/issues/19776)) ([3cabd10](https://github.com/aws/aws-cdk/commit/3cabd105288789c03d1a8d508637b2d7f46407a4)), closes [#19626](https://github.com/aws/aws-cdk/issues/19626)
* **integ-runner:** add missing features from the integ manifest ([#19969](https://github.com/aws/aws-cdk/issues/19969)) ([2ca5050](https://github.com/aws/aws-cdk/commit/2ca5050865f94e033fda850961439d8fcb01f468))
* **integ-runner:** integ-runner enhancements ([#19865](https://github.com/aws/aws-cdk/issues/19865)) ([697fdbe](https://github.com/aws/aws-cdk/commit/697fdbe71642c93492c38e834e654ed736a9edb4))
* **integ-runner:** test update path when running tests ([#19915](https://github.com/aws/aws-cdk/issues/19915)) ([d0ace8f](https://github.com/aws/aws-cdk/commit/d0ace8f2db53d56cdb670979c7c173ee17b6bcd8))
* **integ-tests:** Add `IntegTestCase` ([#19829](https://github.com/aws/aws-cdk/issues/19829)) ([ad249c9](https://github.com/aws/aws-cdk/commit/ad249c9943c2d602b2b077435935731f723db715))
* **iotevents:** support comparison operators ([#19329](https://github.com/aws/aws-cdk/issues/19329)) ([95cb3f3](https://github.com/aws/aws-cdk/commit/95cb3f3c7a4c98ebf4a4818af2f4e725fc16aa29))


### Bug Fixes

* **integ-runner:** enable all feature flags by default ([#19955](https://github.com/aws/aws-cdk/issues/19955)) ([ca3920d](https://github.com/aws/aws-cdk/commit/ca3920dbd588ebd9c68f17bfbf420713cf42790a))
* **lambda-python:** Pipenv projects no longer work for Python 3.6 ([#20019](https://github.com/aws/aws-cdk/issues/20019)) ([08cfc2d](https://github.com/aws/aws-cdk/commit/08cfc2d5a2e3727e692311b244b1fcb6c3b3f5f7))

## [2.20.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.19.0-alpha.0...v2.20.0-alpha.0) (2022-04-07)


### Features

* **synthetics:** new puppeteer 3.5 runtime ([#19673](https://github.com/aws/aws-cdk/issues/19673)) ([ce2b91b](https://github.com/aws/aws-cdk/commit/ce2b91b319da0221adffcdda54321b860db2a56d)), closes [#19634](https://github.com/aws/aws-cdk/issues/19634)

## [2.19.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.18.0-alpha.0...v2.19.0-alpha.0) (2022-03-31)


### Features

* **kinesisanalytics-flink:** Add metrics to Flink applications ([#19599](https://github.com/aws/aws-cdk/issues/19599)) ([dab6aca](https://github.com/aws/aws-cdk/commit/dab6aca5005c8d6d180aada699a4cebc2ef5aefa))

## [2.18.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.17.0-alpha.0...v2.18.0-alpha.0) (2022-03-28)


### Features

* **appsync:** support custom domain mappings ([#19368](https://github.com/aws/aws-cdk/issues/19368)) ([8c7a4ac](https://github.com/aws/aws-cdk/commit/8c7a4acbd58975a8f1c4e4ca180ca9a3ea2c750d)), closes [#18040](https://github.com/aws/aws-cdk/issues/18040)
* **synthetics:** add support for puppeteer 3.4 runtime ([#19429](https://github.com/aws/aws-cdk/issues/19429)) ([024b890](https://github.com/aws/aws-cdk/commit/024b890c67392e255ea8e82c1aa58bcc6bcf6f86)), closes [#19382](https://github.com/aws/aws-cdk/issues/19382)

## [2.17.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.16.0-alpha.0...v2.17.0-alpha.0) (2022-03-17)


### Features

* **appsync:** add OpenSearch domain data source ([#16529](https://github.com/aws/aws-cdk/issues/16529)) ([922a9dc](https://github.com/aws/aws-cdk/commit/922a9dcf07174334ac67b9fcbacb01aafdfd9c6a)), closes [#16528](https://github.com/aws/aws-cdk/issues/16528)
* **iotevents:** support SetVariable action ([#19305](https://github.com/aws/aws-cdk/issues/19305)) ([c222b12](https://github.com/aws/aws-cdk/commit/c222b122206e00dc9932639efd54d78a16ebf6d3))
* **synthetics:** add vpc configuration ([#18447](https://github.com/aws/aws-cdk/issues/18447)) ([c991e92](https://github.com/aws/aws-cdk/commit/c991e92453034330b68daa5b5721119e770b6109)), closes [#11865](https://github.com/aws/aws-cdk/issues/11865) [#9954](https://github.com/aws/aws-cdk/issues/9954)

## [2.16.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.15.0-alpha.0...v2.16.0-alpha.0) (2022-03-11)


### Features

* **aws-s3objectlambda:** add L2 construct for S3 Object Lambda ([#15833](https://github.com/aws/aws-cdk/issues/15833)) ([fe9f750](https://github.com/aws/aws-cdk/commit/fe9f750bd9dd9974b9ae6f73c78fcd12ab2edd91)), closes [#13675](https://github.com/aws/aws-cdk/issues/13675)
* **iotevents:** support actions ([#18869](https://github.com/aws/aws-cdk/issues/18869)) ([e01654e](https://github.com/aws/aws-cdk/commit/e01654e792708ee283d7a31e1370d0d1ae383171))
* **iotevents:** support setting Events on input and exit for State ([#19249](https://github.com/aws/aws-cdk/issues/19249)) ([ffa9e0d](https://github.com/aws/aws-cdk/commit/ffa9e0d287d0a86e1e0eb7dc2dec16d9f3e84450))
* **servicecatalog:** Service Catalog is now in Developer Preview ([#19204](https://github.com/aws/aws-cdk/issues/19204)) ([6dfc254](https://github.com/aws/aws-cdk/commit/6dfc254e1925597b4ef2ece9c132b1a0e580dd6d))


### Bug Fixes

* **apigatewayv2-integrations:** in case of multiple routes, only one execute permission is created ([#18716](https://github.com/aws/aws-cdk/issues/18716)) ([1e352ca](https://github.com/aws/aws-cdk/commit/1e352ca2ab458bfe4e1de6cf431166654ce9aa58))
* **lambda-python:** asset bundling fails on windows ([#19270](https://github.com/aws/aws-cdk/issues/19270)) ([0da57da](https://github.com/aws/aws-cdk/commit/0da57da9606d982788350a6257f0f0ed6e9fd92a)), closes [#18861](https://github.com/aws/aws-cdk/issues/18861)
* **lambda-python:** docker image gets built even when we don't need to bundle assets  ([#16192](https://github.com/aws/aws-cdk/issues/16192)) ([5dc61ea](https://github.com/aws/aws-cdk/commit/5dc61eabc0ea3e6294f83db5deb8528461a1d5bc)), closes [#14747](https://github.com/aws/aws-cdk/issues/14747)

## [2.15.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.14.0-alpha.0...v2.15.0-alpha.0) (2022-03-01)


### Bug Fixes

* **aws-lambda-python:** skip default docker build when image passed ([#19143](https://github.com/aws/aws-cdk/issues/19143)) ([7300f2e](https://github.com/aws/aws-cdk/commit/7300f2eee9e1593eef271d7a953edf80a8962e08)), closes [#18082](https://github.com/aws/aws-cdk/issues/18082)

## [2.14.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.13.0-alpha.0...v2.14.0-alpha.0) (2022-02-25)


### Features

* **apigatewayv2:** Import existing WebSocketApi from attributes ([#18958](https://github.com/aws/aws-cdk/issues/18958)) ([f203845](https://github.com/aws/aws-cdk/commit/f203845d26ae8333f467f1cb91ad965697087d85))

## [2.13.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.12.0-alpha.0...v2.13.0-alpha.0) (2022-02-18)


### Features

* **iot-actions:** add SNS publish action ([#18839](https://github.com/aws/aws-cdk/issues/18839)) ([3a39f6b](https://github.com/aws/aws-cdk/commit/3a39f6bf34eb428c527db1c614ed682c582821fb)), closes [#17700](https://github.com/aws/aws-cdk/issues/17700)
* **iotevents:** create new module for IoT Events actions ([#18956](https://github.com/aws/aws-cdk/issues/18956)) ([3533ea9](https://github.com/aws/aws-cdk/commit/3533ea9cb7ec7fd9e230abd27556a87d3559bdb8)), closes [/github.com/aws/aws-cdk/pull/18869#discussion_r802719713](https://github.com/aws//github.com/aws/aws-cdk/pull/18869/issues/discussion_r802719713)


### Bug Fixes

* **synthetics:** generated role has incorrect permissions for cloudwatch logs ([#18946](https://github.com/aws/aws-cdk/issues/18946)) ([f8bb85f](https://github.com/aws/aws-cdk/commit/f8bb85fad8f659a2b72d5d05d7a94c97765a76f8)), closes [#18910](https://github.com/aws/aws-cdk/issues/18910)

## [2.12.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.11.0-alpha.0...v2.12.0-alpha.0) (2022-02-08)


### Features

* **iotevents:** support transition events ([#18768](https://github.com/aws/aws-cdk/issues/18768)) ([ccc1988](https://github.com/aws/aws-cdk/commit/ccc198864f92620857da09c68013123e9cd3f01d)), closes [#17711](https://github.com/aws/aws-cdk/issues/17711)

## [2.11.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.10.0-alpha.0...v2.11.0-alpha.0) (2022-02-08)


### Features

* **amplify:** support performance mode in Branch ([#18598](https://github.com/aws/aws-cdk/issues/18598)) ([bdeb8eb](https://github.com/aws/aws-cdk/commit/bdeb8eb604f5012ce3180d2f6d887fed1834e4f4)), closes [#18557](https://github.com/aws/aws-cdk/issues/18557)
* **iot:** add Action to republish MQTT messages to another MQTT topic ([#18661](https://github.com/aws/aws-cdk/issues/18661)) ([7ac1215](https://github.com/aws/aws-cdk/commit/7ac121546776cae972bbfb89c2a11949762e7c47))
* **iotevents:** add grant method to Input class ([#18617](https://github.com/aws/aws-cdk/issues/18617)) ([e89688e](https://github.com/aws/aws-cdk/commit/e89688ec1dd7a3b072d23287cddcb73bccc16fd4))


### Bug Fixes

* **aws-appsync:** Strip unsupported characters from Lambda DataSource ([#18765](https://github.com/aws/aws-cdk/issues/18765)) ([bb8d6f6](https://github.com/aws/aws-cdk/commit/bb8d6f6bf5941b76ef0590c99fe8e26440e09c18))

## [2.10.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.9.0-alpha.0...v2.10.0-alpha.0) (2022-01-29)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **servicecatalog:** `TagOptions` now have `scope` and `props` argument in constructor, and data is now passed via a `allowedValueForTags` field in props

### Features

* **iotevents:** add DetectorModel L2 Construct ([#18049](https://github.com/aws/aws-cdk/issues/18049)) ([d0960f1](https://github.com/aws/aws-cdk/commit/d0960f181e5f66daa1eb53be2190b7e62bd66030)), closes [#17711](https://github.com/aws/aws-cdk/issues/17711) [#17711](https://github.com/aws/aws-cdk/issues/17711)
* **iotevents:** allow setting description, evaluation method and key of DetectorModel ([#18644](https://github.com/aws/aws-cdk/issues/18644)) ([2eeaebc](https://github.com/aws/aws-cdk/commit/2eeaebc3cdc9c5c7ef3fa312b3d1abca265dcbb6))
* **lambda-python:** support setting environment vars for bundling ([#18635](https://github.com/aws/aws-cdk/issues/18635)) ([30e2233](https://github.com/aws/aws-cdk/commit/30e223333fef0b0d7f12287dab170a34e092d7fa))
* **servicecatalog:** Create TagOptions Construct ([#18314](https://github.com/aws/aws-cdk/issues/18314)) ([903c4b6](https://github.com/aws/aws-cdk/commit/903c4b6e4adf676fae42265a048dddd0e1386542)), closes [#17753](https://github.com/aws/aws-cdk/issues/17753)


### Bug Fixes

* **apigatewayv2:** websocket api: allow all methods in grant manage connections ([#18544](https://github.com/aws/aws-cdk/issues/18544)) ([41c8a3f](https://github.com/aws/aws-cdk/commit/41c8a3fa6b50a94affb65286d862056050d02e84)), closes [#18410](https://github.com/aws/aws-cdk/issues/18410)
* **synthetics:** correct getbucketlocation policy ([#13573](https://github.com/aws/aws-cdk/issues/13573)) ([e743525](https://github.com/aws/aws-cdk/commit/e743525b6379371110d737bb360f637c41d30ca1)), closes [#13572](https://github.com/aws/aws-cdk/issues/13572)

## [2.9.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.8.0-alpha.0...v2.9.0-alpha.0) (2022-01-26)


### Features

* **aws-neptune:** add autoMinorVersionUpgrade to cluster props ([#18394](https://github.com/aws/aws-cdk/issues/18394)) ([8b5320a](https://github.com/aws/aws-cdk/commit/8b5320ac5e5c320db46bc74f33b3841977dd3a5d)), closes [#17545](https://github.com/aws/aws-cdk/issues/17545)
* **iot:** add Action to put record to Kinesis Data stream ([#18321](https://github.com/aws/aws-cdk/issues/18321)) ([1480213](https://github.com/aws/aws-cdk/commit/1480213a032549ab7319e0c3a66e02e9b6a9c4ab)), closes [#17703](https://github.com/aws/aws-cdk/issues/17703)

## [2.8.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.7.0-alpha.0...v2.8.0-alpha.0) (2022-01-13)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **apigatewayv2:** `HttpIntegrationType.LAMBDA_PROXY` has been renamed to `HttpIntegrationType.AWS_PROXY`

### Features

* **apigatewayv2:** HttpRouteIntegration supports AWS services integrations ([#18154](https://github.com/aws/aws-cdk/issues/18154)) ([a8094c7](https://github.com/aws/aws-cdk/commit/a8094c7d9970557077f560ccd24882216094ee3c)), closes [#16287](https://github.com/aws/aws-cdk/issues/16287)
* **apigatewayv2:** support for mock integration type ([#18129](https://github.com/aws/aws-cdk/issues/18129)) ([7779c14](https://github.com/aws/aws-cdk/commit/7779c147c7445d9e8ccafa9b732521c9021a6234)), closes [#15008](https://github.com/aws/aws-cdk/issues/15008)

## [2.7.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.6.0-alpha.0...v2.7.0-alpha.0) (2022-01-12)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **iot:** the class `FirehoseStreamAction` has been renamed to `FirehosePutRecordAction`
* **apigatewayv2-authorizers:** `WebSocketLambdaAuthorizerProps.identitySource` default changes from `['$request.header.Authorization']` to `['route.request.header.Authorization']`.

### Features

* **apigatewayv2:** websocket api: api keys ([#16636](https://github.com/aws/aws-cdk/issues/16636)) ([24f8f74](https://github.com/aws/aws-cdk/commit/24f8f74ebec023f5e3f5bd2bdfc89575a53b38f3))


### Bug Fixes

* **apigatewayv2-authorizers:** incorrect `identitySource` default for `WebSocketLambdaAuthorizer` ([#18315](https://github.com/aws/aws-cdk/issues/18315)) ([74eee1e](https://github.com/aws/aws-cdk/commit/74eee1e5b8fa404dde129f001b986d615f435c73)), closes [#16886](https://github.com/aws/aws-cdk/issues/16886) [/docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-apigatewayv2-authorizer.html#cfn-apigatewayv2](https://github.com/aws//docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-apigatewayv2-authorizer.html/issues/cfn-apigatewayv2) [#18307](https://github.com/aws/aws-cdk/issues/18307)
* **iot:** `FirehoseStreamAction` is now called `FirehosePutRecordAction` ([#18356](https://github.com/aws/aws-cdk/issues/18356)) ([c016a9f](https://github.com/aws/aws-cdk/commit/c016a9fcf51f2415e6e0fcca2255da384c8abbc1)), closes [/github.com/aws/aws-cdk/pull/18321#discussion_r781620195](https://github.com/aws//github.com/aws/aws-cdk/pull/18321/issues/discussion_r781620195)
* **lambda-python:** asset files are generated inside the 'asset-input' folder ([#18306](https://github.com/aws/aws-cdk/issues/18306)) ([b00b44e](https://github.com/aws/aws-cdk/commit/b00b44efd6e402744725e711906b456a28cebc5b))
* **lambda-python:** asset files are generated inside the 'asset-input' folder ([#18306](https://github.com/aws/aws-cdk/issues/18306)) ([aff607a](https://github.com/aws/aws-cdk/commit/aff607a65e061ade5c3ec9e29f82fdaa8b57f638))
* **lambda-python:** bundle asset files correctly ([#18335](https://github.com/aws/aws-cdk/issues/18335)) ([3822c85](https://github.com/aws/aws-cdk/commit/3822c855cf92ee0cd4539dee33e55f57d995bf89)), closes [/github.com/aws/aws-cdk/pull/18306#discussion_r780186564](https://github.com/aws//github.com/aws/aws-cdk/pull/18306/issues/discussion_r780186564) [#18301](https://github.com/aws/aws-cdk/issues/18301) [/github.com/aws/aws-cdk/pull/18082#issuecomment-1008442363](https://github.com/aws//github.com/aws/aws-cdk/pull/18082/issues/issuecomment-1008442363)

## [2.6.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.5.0-alpha.0...v2.6.0-alpha.0) (2022-01-12)


### Bug Fixes

* **lambda-python:** asset files are generated inside the 'asset-input' folder (backport [#18306](https://github.com/aws/aws-cdk/issues/18306)) ([#18341](https://github.com/aws/aws-cdk/issues/18341)) ([a1715e4](https://github.com/aws/aws-cdk/commit/a1715e42944ba8a1d06513d288f2d28ca94eeccf))

## [2.5.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.4.0-alpha.0...v2.5.0-alpha.0) (2022-01-09)

## [2.4.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.3.0-alpha.0...v2.4.0-alpha.0) (2022-01-06)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **lambda-python:** `assetHashType` and `assetHash` properties moved to new `bundling` property.
* **lambda-python:** Runtime is now required for `LambdaPython`
* **appsync:** The `CachingConfig#ttl` property is now required.

[1]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-appsync-resolver-cachingconfig.html#cfn-appsync-resolver-cachingconfig-ttl

### Features

* **amplify:** Add Amplify asset deployment resource  ([#16922](https://github.com/aws/aws-cdk/issues/16922)) ([499ba85](https://github.com/aws/aws-cdk/commit/499ba857e75aa54aa90606f22b984692c8271152)), closes [#16208](https://github.com/aws/aws-cdk/issues/16208)
* **apigatewayv2:** http api - IAM authorizer support ([#17519](https://github.com/aws/aws-cdk/issues/17519)) ([fd8e0e3](https://github.com/aws/aws-cdk/commit/fd8e0e33816cb46678f7d1beac80b1623cdb6bac)), closes [#15123](https://github.com/aws/aws-cdk/issues/15123) [/github.com/aws/aws-cdk/pull/14853#discussion_r648952691](https://github.com/aws//github.com/aws/aws-cdk/pull/14853/issues/discussion_r648952691) [#10534](https://github.com/aws/aws-cdk/issues/10534)
* **apigatewayv2:** Lambda authorizer for WebSocket API  ([#16886](https://github.com/aws/aws-cdk/issues/16886)) ([67cce37](https://github.com/aws/aws-cdk/commit/67cce37f8ea3e6096e44a926fe61441dfcbc685b)), closes [#13869](https://github.com/aws/aws-cdk/issues/13869)
* **glue:** support partition index on tables ([#17998](https://github.com/aws/aws-cdk/issues/17998)) ([c071367](https://github.com/aws/aws-cdk/commit/c071367def4382c630057546c74fa56f00d9294c)), closes [#17589](https://github.com/aws/aws-cdk/issues/17589)
* **iot:** Action to send messages to SQS queues ([#18087](https://github.com/aws/aws-cdk/issues/18087)) ([37537fe](https://github.com/aws/aws-cdk/commit/37537fe1c1b016ae226bf7bc4ceeb128d6124872)), closes [#17699](https://github.com/aws/aws-cdk/issues/17699)
* **iot:** add Action to set a CloudWatch alarm ([#18021](https://github.com/aws/aws-cdk/issues/18021)) ([de2369c](https://github.com/aws/aws-cdk/commit/de2369c1d64260ed47cccfc2619320123af64a0f)), closes [#17705](https://github.com/aws/aws-cdk/issues/17705)
* **lambda-python:** support for providing a custom bundling docker image ([#18082](https://github.com/aws/aws-cdk/issues/18082)) ([c3c4a97](https://github.com/aws/aws-cdk/commit/c3c4a97e65071fcab35212be82dea7b29454786f)), closes [#10298](https://github.com/aws/aws-cdk/issues/10298) [#12949](https://github.com/aws/aws-cdk/issues/12949) [#15391](https://github.com/aws/aws-cdk/issues/15391) [#16234](https://github.com/aws/aws-cdk/issues/16234) [#15306](https://github.com/aws/aws-cdk/issues/15306)
* **msk:** add Kafka versions 2.6.3, 2.7.1 and 2.7.2 ([#18191](https://github.com/aws/aws-cdk/issues/18191)) ([8832df1](https://github.com/aws/aws-cdk/commit/8832df1d7497ef67b9ec62110d2f371ffe4781aa))


### Bug Fixes

* **amplify:** deploy asset Custom Resource points to TS file ([#18166](https://github.com/aws/aws-cdk/issues/18166)) ([a1508af](https://github.com/aws/aws-cdk/commit/a1508afab55c3ba0aa88b6aa85ca947badacb4a9))
* **appsync:** `ttl` property of `CachingConfig` is not required ([#17981](https://github.com/aws/aws-cdk/issues/17981)) ([73e5fec](https://github.com/aws/aws-cdk/commit/73e5fec36cb149cf666320afbe63308c968c62dd))
* **lambda-python:** runtime is now required ([#18143](https://github.com/aws/aws-cdk/issues/18143)) ([98f1bb1](https://github.com/aws/aws-cdk/commit/98f1bb147624a942773d191344c8d7242adb8d04)), closes [#10248](https://github.com/aws/aws-cdk/issues/10248)

## [2.3.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.2.0-alpha.0...v2.3.0-alpha.0) (2021-12-22)

## [2.2.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.1.0-alpha.0...v2.2.0-alpha.0) (2021-12-15)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **glue:** the grantRead API previously included 'glue:BatchDeletePartition', and now it does not.




### Features

* **iotevents:** add IoT Events input L2 Construct ([#17847](https://github.com/aws/aws-cdk/issues/17847)) ([9f03dc4](https://github.com/aws/aws-cdk/commit/9f03dc4c5b75225942037fb6c8fa7d6abf35fe11)), closes [/github.com/aws/aws-cdk/issues/17711#issuecomment-986153267](https://github.com/aws//github.com/aws/aws-cdk/issues/17711/issues/issuecomment-986153267)


### Bug Fixes

* **appsync:** empty caching config is created when not provided ([#17947](https://github.com/aws/aws-cdk/issues/17947)) ([3a9f206](https://github.com/aws/aws-cdk/commit/3a9f20669cc8338d05f9ef8684aa7e50748baa3d))
* **glue:** remove `batchDeletePartition` from `grantRead()` permissions ([#17941](https://github.com/aws/aws-cdk/issues/17941)) ([3d64f9b](https://github.com/aws/aws-cdk/commit/3d64f9b8c07e83d4a085e3eaf69629a866bc20d0)), closes [#17935](https://github.com/aws/aws-cdk/issues/17935) [#15116](https://github.com/aws/aws-cdk/issues/15116)

## [2.1.0-alpha.0](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.11...v2.1.0-alpha.0) (2021-12-08)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **apigatewayv2-authorizers:** The default value for the prop `authorizerName`
in `HttpJwtAuthorizerProps` has changed.
* **apigatewayv2-authorizers:** `HttpJwtAuthorizer` now takes the
  construct id and the target jwt issuer as part of its constructor.
* **apigatewayv2-authorizers:** `HttpLambdaAuthorizer` now takes
  the construct id and the target lambda function handler as part of
  its constructor.
* **apigatewayv2-authorizers:** The default value for the prop
  `authorizerName` in `HttpUserPoolAuthorizerProps` has changed.
* **apigatewayv2:** The `HttpIntegration` and `WebSocketIntegration`
classes require an "id" parameter to be provided during its initialization.
* **apigatewayv2-integrations:** The `LambdaWebSocketIntegration` is now
  renamed to `WebSocketLambdaIntegration`. The new class accepts the
  handler to the target lambda function directly in its constructor.
* **apigatewayv2-integrations:** `HttpProxyIntegration` and
  `HttpProxyIntegrationProps` are now renamed to `HttpUrlIntegration`
  and `HttpUrlIntegrationProps` respectively. The new class accepts the
  target url directly in its constructor.
* **apigatewayv2-integrations:** `LambdaProxyIntegration` and
  `LambdaProxyIntegrationProps` are now renamed to
  `HttpLambdaIntegration` and `HttpLambdaIntegrationProps` respectively.
  The new class accepts the lambda function handler directly in its
  constructor.
* **apigatewayv2-integrations:** `HttpAlbIntegration` now accepts the
  ELB listener directly in its constructor.
* **apigatewayv2-integrations:** `HttpNlbIntegration` now accepts the
  ELB listener directly in its constructor.
* **apigatewayv2-integrations:** `HttpServiceDiscoveryIntegration` now
  accepts the service discovery Service directly in its constructor.
* **apigatewayv2-authorizers:** `UserPoolAuthorizerProps` is now
  renamed to `HttpUserPoolAuthorizerProps`.
* **apigatewayv2:** The interface `IHttpRouteIntegration` is replaced by
the abstract class `HttpRouteIntegration`.
* **apigatewayv2:** The interface `IWebSocketRouteIntegration` is now
  replaced by the abstract class `WebSocketRouteIntegration`.
* **apigatewayv2:** Previously, we allowed the usage of integration
  classes to be used with routes defined in multiple `HttpApi` instances
  (or `WebSocketApi` instances). This is now disallowed, and separate
  instances must be created for each instance of `HttpApi` or
  `WebSocketApi`.

### Features

* **iot:** add Action to capture CloudWatch metrics ([#17503](https://github.com/aws/aws-cdk/issues/17503)) ([ec4187c](https://github.com/aws/aws-cdk/commit/ec4187c26d68df970d72d0e766d7d27b42e8b784)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **neptune:** add engine version 1.1.0.0 and instance types t4g, r6g ([#17669](https://github.com/aws/aws-cdk/issues/17669)) ([83e669d](https://github.com/aws/aws-cdk/commit/83e669dcdae9390990598236c75015832af766b4))
* **servicecatalog:** Add TagOptions to a CloudformationProduct ([#17672](https://github.com/aws/aws-cdk/issues/17672)) ([2d19e15](https://github.com/aws/aws-cdk/commit/2d19e1535586d2b006d43da787ffbb0fad8b4978))


### Bug Fixes

* **apigatewayv2:** integration class does not render an integration resource ([#17729](https://github.com/aws/aws-cdk/issues/17729)) ([3b5b97a](https://github.com/aws/aws-cdk/commit/3b5b97ac1f972f53240798df19af43d85ebf6f13)), closes [#13213](https://github.com/aws/aws-cdk/issues/13213)
* **apprunner:** startCommand and environment are ignored in imageConfiguration  ([#16939](https://github.com/aws/aws-cdk/issues/16939)) ([d911c58](https://github.com/aws/aws-cdk/commit/d911c5878c59498a2d0e14ff536e0f8f9f503bfe)), closes [#16812](https://github.com/aws/aws-cdk/issues/16812)
* **appsync:** add caching config to AppSync resolvers ([#17815](https://github.com/aws/aws-cdk/issues/17815)) ([52b535b](https://github.com/aws/aws-cdk/commit/52b535bda5f26b07377fcdfca63a75c62eb5f883))
* **appsync:** remove 'id' suffix to union definition key ([#17787](https://github.com/aws/aws-cdk/issues/17787)) ([86e7780](https://github.com/aws/aws-cdk/commit/86e77806391dc3fe8cd254fec773320cdb425dec)), closes [#17771](https://github.com/aws/aws-cdk/issues/17771)
* **assert:** support multiline strings with `stringLike()` ([#17692](https://github.com/aws/aws-cdk/issues/17692)) ([37596e6](https://github.com/aws/aws-cdk/commit/37596e6be4cf05432dcba3838955484e512beca6))


### Miscellaneous Chores

* **apigatewayv2:** integration api re-organization ([#17752](https://github.com/aws/aws-cdk/issues/17752)) ([29039e8](https://github.com/aws/aws-cdk/commit/29039e8bd13a4fdb7f84254038b3331c179273fd))
* **apigatewayv2-authorizers:** re-organize authorizer api ([#17772](https://github.com/aws/aws-cdk/issues/17772)) ([719f33e](https://github.com/aws/aws-cdk/commit/719f33e20c723f161fc35230fafd7e99bca66a61))

## [2.0.0-alpha.11](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.10...v2.0.0-alpha.11) (2021-12-02)

## [2.0.0-alpha.10](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.9...v2.0.0-alpha.10) (2021-11-26)

## [2.0.0-alpha.9](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.8...v2.0.0-alpha.9) (2021-11-25)

## [2.0.0-alpha.8](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.7...v2.0.0-alpha.8) (2021-11-23)


### Features

* **apigatewayv2:** domain endpoint type, security policy and endpoint migration ([#17518](https://github.com/aws/aws-cdk/issues/17518)) ([261b331](https://github.com/aws/aws-cdk/commit/261b331e89be01dc996d153c91b4018e7ddfda29))
* **apigatewayv2:** http api - mTLS support  ([#17284](https://github.com/aws/aws-cdk/issues/17284)) ([54be156](https://github.com/aws/aws-cdk/commit/54be1567546ffd52e83fbe52531f901c0b6c29c9)), closes [#12559](https://github.com/aws/aws-cdk/issues/12559)
* **apigatewayv2:** websocket api: grant manage connections ([#16872](https://github.com/aws/aws-cdk/issues/16872)) ([10dfa60](https://github.com/aws/aws-cdk/commit/10dfa60a693db6e38a1188effc6eeebc2b5c49b8)), closes [#14828](https://github.com/aws/aws-cdk/issues/14828)
* **iot:** add Action to put objects in S3 Buckets ([#17307](https://github.com/aws/aws-cdk/issues/17307)) ([49b87db](https://github.com/aws/aws-cdk/commit/49b87dbfe5a37abad8880e0325f7aa8218705407)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **iot:** add Action to put records to a Firehose stream ([#17466](https://github.com/aws/aws-cdk/issues/17466)) ([7cb5f2c](https://github.com/aws/aws-cdk/commit/7cb5f2cc8402aad223eb5c50cdf5ee2e0d56150e)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **msk:** add Kafka version 2.6.2 ([#17497](https://github.com/aws/aws-cdk/issues/17497)) ([5f1f476](https://github.com/aws/aws-cdk/commit/5f1f4762e964345741426fa1242320a5fc117338))
* **redshift:** Add support for distStyle, distKey, sortStyle and sortKey to Table ([#17135](https://github.com/aws/aws-cdk/issues/17135)) ([a137cd1](https://github.com/aws/aws-cdk/commit/a137cd13a90cc3bfdb8207bd8764e2eb05836126)), closes [#17125](https://github.com/aws/aws-cdk/issues/17125)
* **servicecatalog:** support local launch role name in launch role constraint ([#17371](https://github.com/aws/aws-cdk/issues/17371)) ([b307b69](https://github.com/aws/aws-cdk/commit/b307b6996ed13b1f2dedeb41d29409183becb969))


### Bug Fixes

* **iot:** unable to add the same lambda function to two TopicRule Actions ([#17521](https://github.com/aws/aws-cdk/issues/17521)) ([eda1640](https://github.com/aws/aws-cdk/commit/eda1640fcaf6375d7edc5f8edcb5d69c82d130a1)), closes [#17508](https://github.com/aws/aws-cdk/issues/17508)
* **redshift:** tableNameSuffix evaluation ([#17213](https://github.com/aws/aws-cdk/issues/17213)) ([f7c3217](https://github.com/aws/aws-cdk/commit/f7c3217a731804f014526e10b414a9e7f27d575b)), closes [#17064](https://github.com/aws/aws-cdk/issues/17064)

## [2.0.0-alpha.7](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.6...v2.0.0-alpha.7) (2021-11-17)

## [2.0.0-alpha.6](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.5...v2.0.0-alpha.6) (2021-11-10)

## [2.0.0-alpha.5](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.4...v2.0.0-alpha.5) (2021-11-09)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **apigatewayv2-authorizers:** `userPoolClient` property in `UserPoolAuthorizerProps`
is now renamed to `userPoolClients`.

### Features

* **apigatewayv2-authorizers:** http api - allow multiple user pool clients per HttpUserPoolAuthorizer  ([#16903](https://github.com/aws/aws-cdk/issues/16903)) ([747eb7c](https://github.com/aws/aws-cdk/commit/747eb7cf5dba4514241103ffebc49e03261d25a9)), closes [#15431](https://github.com/aws/aws-cdk/issues/15431)
* **iot:** allow setting `description` and `enabled` of TopicRule ([#17225](https://github.com/aws/aws-cdk/issues/17225)) ([a9aae09](https://github.com/aws/aws-cdk/commit/a9aae097daad475dd57bbf4842956327a6d5a220)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **iot:** allow setting `errorAction` of TopicRule  ([#17287](https://github.com/aws/aws-cdk/issues/17287)) ([e412308](https://github.com/aws/aws-cdk/commit/e412308bc81ede16b079077cfa4774ceaa2fadeb)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **iot:** allow setting Actions of TopicRule ([#17110](https://github.com/aws/aws-cdk/issues/17110)) ([0cabb9f](https://github.com/aws/aws-cdk/commit/0cabb9f2d2f50c03337cd6f35bf47fc54ada3a21)), closes [#16681](https://github.com/aws/aws-cdk/issues/16681) [/github.com/aws/aws-cdk/pull/16681#discussion_r733912215](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/discussion_r733912215)
* **iot-actions:** Add the action to put CloudWatch Logs ([#17228](https://github.com/aws/aws-cdk/issues/17228)) ([a7c869e](https://github.com/aws/aws-cdk/commit/a7c869e6d57932389df572cd7f104a4c9ea8f8a5)), closes [/github.com/aws/aws-cdk/pull/16681#issuecomment-942233029](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/issuecomment-942233029)
* **servicecatalog:** allow creating a CFN Product Version with CDK code ([#17144](https://github.com/aws/aws-cdk/issues/17144)) ([f8d0ef5](https://github.com/aws/aws-cdk/commit/f8d0ef550df07e43aeab35dde4406c92f7551ed0))
* **synthetics:** add static cron method to schedule class ([#17250](https://github.com/aws/aws-cdk/issues/17250)) ([1ab9b26](https://github.com/aws/aws-cdk/commit/1ab9b265e9899ffcd093b3600d658c8a6519cc69)), closes [#16402](https://github.com/aws/aws-cdk/issues/16402)

## [2.0.0-alpha.4](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.3...v2.0.0-alpha.4) (2021-10-27)


### Features

* **amplify:** Add support for custom headers in the App ([#17102](https://github.com/aws/aws-cdk/issues/17102)) ([9f3abd7](https://github.com/aws/aws-cdk/commit/9f3abd745c98a65e7314528f40d08ea2ecbe19a6)), closes [#17084](https://github.com/aws/aws-cdk/issues/17084)
* **synthetics:** add syn-nodejs-puppeteer-3.3 runtime ([#17132](https://github.com/aws/aws-cdk/issues/17132)) ([8343bec](https://github.com/aws/aws-cdk/commit/8343beccbee06f453b63387f54768b320fe01339))


### Bug Fixes

* **redshift:** cluster uses key ARN instead of key ID ([#17108](https://github.com/aws/aws-cdk/issues/17108)) ([bdf30c6](https://github.com/aws/aws-cdk/commit/bdf30c696b04b26a8e41548839d5c4cf61d471cc)), closes [#17032](https://github.com/aws/aws-cdk/issues/17032)

## [2.0.0-alpha.3](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.2...v2.0.0-alpha.3) (2021-10-25)


### Features

* **iot:** create new aws-iot-actions module ([#17112](https://github.com/aws/aws-cdk/issues/17112)) ([06838e6](https://github.com/aws/aws-cdk/commit/06838e66db0c9a48e2aa7da1e7707fda335bb62c)), closes [#16681](https://github.com/aws/aws-cdk/issues/16681) [/github.com/aws/aws-cdk/pull/16681#discussion_r733912215](https://github.com/aws//github.com/aws/aws-cdk/pull/16681/issues/discussion_r733912215)

## [2.0.0-alpha.2](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.1...v2.0.0-alpha.2) (2021-10-22)


### Features

* **apigatewayv2-integrations:** http api - support for request parameter mapping ([#15630](https://github.com/aws/aws-cdk/issues/15630)) ([0452aed](https://github.com/aws/aws-cdk/commit/0452aed2f00198e05bd65b1d20246f7de0b24e20))
* **iot:** add the TopicRule L2 construct ([#16681](https://github.com/aws/aws-cdk/issues/16681)) ([86f85ce](https://github.com/aws/aws-cdk/commit/86f85ce10f78b86133f9dab9851e56d03fb28cc0)), closes [#16602](https://github.com/aws/aws-cdk/issues/16602)
* **msk:** add Kafka version 2.8.1 ([#16881](https://github.com/aws/aws-cdk/issues/16881)) ([7db5c8c](https://github.com/aws/aws-cdk/commit/7db5c8cdafe7b9b22b6b40cb25ed8bd1946301f4))


### Bug Fixes

* **apigatewayv2:** unable to retrieve domain url for default stage ([#16854](https://github.com/aws/aws-cdk/issues/16854)) ([c6db91e](https://github.com/aws/aws-cdk/commit/c6db91eee2cb658ce347c7ac6d6e3c95bc5977dc)), closes [#16638](https://github.com/aws/aws-cdk/issues/16638)

## [2.0.0-alpha.1](https://github.com/aws/aws-cdk/compare/v2.0.0-alpha.0...v2.0.0-alpha.1) (2021-10-13)


### ⚠ BREAKING CHANGES TO EXPERIMENTAL FEATURES

* **assertions:** Starting this release, the `assertions` module will be
published to Maven with the name 'assertions' instead of
'cdk-assertions'.
* **assertions:** `Match.absentProperty()` becomes `Match.absent()`, and its type changes from `string` to `Matcher`.
* **assertions:** The `templateMatches()` API previously performed
an exact match. The default behavior has been updated to be
"object-like".
* **assertions:** the `findResources()` API previously returned a list of resources, but now returns a map of logical id to resource.
* **assertions:** the `findOutputs()` API previously returned a list of outputs, but now returns a map of logical id to output.
* **assertions:** the `findMappings()` API previously returned a list of mappings, but now returns a map of logical id to mapping.

### Features

* **appsync:** Lambda Authorizer for AppSync GraphqlApi ([#16743](https://github.com/aws/aws-cdk/issues/16743)) ([bdbe8b6](https://github.com/aws/aws-cdk/commit/bdbe8b6cf6ab1ae261dddeb39576749e768183b3)), closes [#16380](https://github.com/aws/aws-cdk/issues/16380)
* **assertions:** capture matching value ([#16426](https://github.com/aws/aws-cdk/issues/16426)) ([cc74f92](https://github.com/aws/aws-cdk/commit/cc74f92f275a338cb53caa7d6f124ab0dd960f0b))
* **assertions:** findXxx() APIs now includes the logical id as part of its result ([#16454](https://github.com/aws/aws-cdk/issues/16454)) ([532a72b](https://github.com/aws/aws-cdk/commit/532a72b133e6ebd0c7b8b7c65b273bb0e6f3293c))
* **assertions:** match into serialized json ([#16456](https://github.com/aws/aws-cdk/issues/16456)) ([fed30fc](https://github.com/aws/aws-cdk/commit/fed30fc815bac1006003524ac6232778f3c3babe))
* **assertions:** matcher support for `templateMatches()` API ([#16789](https://github.com/aws/aws-cdk/issues/16789)) ([0fb2179](https://github.com/aws/aws-cdk/commit/0fb21799b0da3185c2d4ba91a8ef9729c71fbd5a))
* **aws-apprunner:** support the Service L2 construct ([#15810](https://github.com/aws/aws-cdk/issues/15810)) ([3cea941](https://github.com/aws/aws-cdk/commit/3cea9419b6c02b3b5eb952b7e03b5a132e5e9630)), closes [#14813](https://github.com/aws/aws-cdk/issues/14813)
* **batch:** fargate support for jobs ([#15848](https://github.com/aws/aws-cdk/issues/15848)) ([066bcb1](https://github.com/aws/aws-cdk/commit/066bcb1e5d53192bd465190c8a4f81c5838987f4)), closes [#13591](https://github.com/aws/aws-cdk/issues/13591) [#13590](https://github.com/aws/aws-cdk/issues/13590) [#13591](https://github.com/aws/aws-cdk/issues/13591)
* **glue:** Job construct ([#12506](https://github.com/aws/aws-cdk/issues/12506)) ([fc74110](https://github.com/aws/aws-cdk/commit/fc74110ff7eae544d9cfc11b2f6779169f17d145)), closes [#12443](https://github.com/aws/aws-cdk/issues/12443)
* **neptune:** add engine version 1.0.5.0 ([#16394](https://github.com/aws/aws-cdk/issues/16394)) ([deaac4a](https://github.com/aws/aws-cdk/commit/deaac4a16e957bd046f24a6c26d735fc4cf980bd)), closes [#16388](https://github.com/aws/aws-cdk/issues/16388)
* **redshift:** manage database users and tables via cdk ([#15931](https://github.com/aws/aws-cdk/issues/15931)) ([a9d5118](https://github.com/aws/aws-cdk/commit/a9d51185a144cd4962c85227ae5b904510399fa4)), closes [#9815](https://github.com/aws/aws-cdk/issues/9815)


### Bug Fixes

* **apigatewayv2:** ApiMapping does not depend on DomainName ([#16201](https://github.com/aws/aws-cdk/issues/16201)) ([1e247d8](https://github.com/aws/aws-cdk/commit/1e247d89adbc09ff79b87753fcd78b238a6752e8)), closes [#15464](https://github.com/aws/aws-cdk/issues/15464)
* **assertions:** `hasResourceProperties` is incompatible with `Match.not` and `Match.absent` ([#16678](https://github.com/aws/aws-cdk/issues/16678)) ([6f0a507](https://github.com/aws/aws-cdk/commit/6f0a5076b1e074fd33ed118af8e48b72d7593418)), closes [#16626](https://github.com/aws/aws-cdk/issues/16626)
* **aws-servicecatalog:** Allow users to create multiple product versions from assets.  ([#16914](https://github.com/aws/aws-cdk/issues/16914)) ([958d755](https://github.com/aws/aws-cdk/commit/958d755ff7acaf016e3f8969bf5ab07d4dc2977b))
* **route53resolver:** FirewallDomainList throws with wildcard domains ([#16538](https://github.com/aws/aws-cdk/issues/16538)) ([643e5ee](https://github.com/aws/aws-cdk/commit/643e5ee519095968a758942220f1e3a6c20f54b3)), closes [#16527](https://github.com/aws/aws-cdk/issues/16527)


### Miscellaneous Chores

* **assertions:** consistent naming in maven ([#16921](https://github.com/aws/aws-cdk/issues/16921)) ([0dcd9ec](https://github.com/aws/aws-cdk/commit/0dcd9eca3a1014c39f92d9e052b67974fc751af0))
* **assertions:** replace `absentProperty()` with `absent()` and support it as a `Matcher` type ([#16653](https://github.com/aws/aws-cdk/issues/16653)) ([c980185](https://github.com/aws/aws-cdk/commit/c980185142c58821b7ae7ef0b88c6c98ca8f0246))
