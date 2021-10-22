# Google Cloud Client Libraries for Go

[![GoDoc](https://pkg.go.dev/cloud.google.com/go?status.svg)](https://pkg.go.dev/cloud.google.com/go)

Go packages for [Google Cloud Platform](https://cloud.google.com) services.

``` go
import "cloud.google.com/go"
```

To install the packages on your system, *do not clone the repo*. Instead:

1. Change to your project directory:

   ```
   cd /my/cloud/project
   ```
1. Get the package you want to use. Some products have their own module, so it's
   best to `go get` the package(s) you want to use:

   ```
   $ go get cloud.google.com/go/firestore # Replace with the package you want to use.
   ```

**NOTE:** Some of these packages are under development, and may occasionally
make backwards-incompatible changes.

**NOTE:** Github repo is a mirror of [https://code.googlesource.com/gocloud](https://code.googlesource.com/gocloud).

## Supported APIs

Google API                                      | Status       | Package
------------------------------------------------|--------------|-----------------------------------------------------------
[Asset][cloud-asset]                            | stable       | [`cloud.google.com/go/asset/apiv1`](https://pkg.go.dev/cloud.google.com/go/asset/v1beta)
[Automl][cloud-automl]                          | stable       | [`cloud.google.com/go/automl/apiv1`](https://pkg.go.dev/cloud.google.com/go/automl/apiv1)
[BigQuery][cloud-bigquery]                      | stable       | [`cloud.google.com/go/bigquery`](https://pkg.go.dev/cloud.google.com/go/bigquery)
[Bigtable][cloud-bigtable]                      | stable       | [`cloud.google.com/go/bigtable`](https://pkg.go.dev/cloud.google.com/go/bigtable)
[Cloudbuild][cloud-build]                       | stable       | [`cloud.google.com/go/cloudbuild/apiv1`](https://pkg.go.dev/cloud.google.com/go/cloudbuild/apiv1)
[Cloudtasks][cloud-tasks]                       | stable       | [`cloud.google.com/go/cloudtasks/apiv2`](https://pkg.go.dev/cloud.google.com/go/cloudtasks/apiv2)
[Container][cloud-container]                    | stable       | [`cloud.google.com/go/container/apiv1`](https://pkg.go.dev/cloud.google.com/go/container/apiv1)
[ContainerAnalysis][cloud-containeranalysis]    | beta         | [`cloud.google.com/go/containeranalysis/apiv1`](https://pkg.go.dev/cloud.google.com/go/containeranalysis/apiv1)
[Dataproc][cloud-dataproc]                      | stable       | [`cloud.google.com/go/dataproc/apiv1`](https://pkg.go.dev/cloud.google.com/go/dataproc/apiv1)
[Datastore][cloud-datastore]                    | stable       | [`cloud.google.com/go/datastore`](https://pkg.go.dev/cloud.google.com/go/datastore)
[Debugger][cloud-debugger]                      | stable       | [`cloud.google.com/go/debugger/apiv2`](https://pkg.go.dev/cloud.google.com/go/debugger/apiv2)
[Dialogflow][cloud-dialogflow]                  | stable       | [`cloud.google.com/go/dialogflow/apiv2`](https://pkg.go.dev/cloud.google.com/go/dialogflow/apiv2)
[Data Loss Prevention][cloud-dlp]               | stable       | [`cloud.google.com/go/dlp/apiv2`](https://pkg.go.dev/cloud.google.com/go/dlp/apiv2)
[ErrorReporting][cloud-errors]                  | alpha        | [`cloud.google.com/go/errorreporting`](https://pkg.go.dev/cloud.google.com/go/errorreporting)
[Firestore][cloud-firestore]                    | stable       | [`cloud.google.com/go/firestore`](https://pkg.go.dev/cloud.google.com/go/firestore)
[IAM][cloud-iam]                                | stable       | [`cloud.google.com/go/iam`](https://pkg.go.dev/cloud.google.com/go/iam)
[IoT][cloud-iot]                                | stable       | [`cloud.google.com/go/iot/apiv1`](https://pkg.go.dev/cloud.google.com/go/iot/apiv1)
[IRM][cloud-irm]                                | alpha        | [`cloud.google.com/go/irm/apiv1alpha2`](https://pkg.go.dev/cloud.google.com/go/irm/apiv1alpha2)
[KMS][cloud-kms]                                | stable       | [`cloud.google.com/go/kms/apiv1`](https://pkg.go.dev/cloud.google.com/go/kms/apiv1)
[Natural Language][cloud-natural-language]      | stable       | [`cloud.google.com/go/language/apiv1`](https://pkg.go.dev/cloud.google.com/go/language/apiv1)
[Logging][cloud-logging]                        | stable       | [`cloud.google.com/go/logging`](https://pkg.go.dev/cloud.google.com/go/logging)
[Memorystore][cloud-memorystore]                | alpha        | [`cloud.google.com/go/redis/apiv1`](https://pkg.go.dev/cloud.google.com/go/redis/apiv1)
[Monitoring][cloud-monitoring]                  | stable       | [`cloud.google.com/go/monitoring/apiv3`](https://pkg.go.dev/cloud.google.com/go/monitoring/apiv3)
[OS Login][cloud-oslogin]                       | stable       | [`cloud.google.com/go/oslogin/apiv1`](https://pkg.go.dev/cloud.google.com/go/oslogin/apiv1)
[Pub/Sub][cloud-pubsub]                         | stable       | [`cloud.google.com/go/pubsub`](https://pkg.go.dev/cloud.google.com/go/pubsub)
[Phishing Protection][cloud-phishingprotection] | alpha        | [`cloud.google.com/go/phishingprotection/apiv1beta1`](https://pkg.go.dev/cloud.google.com/go/phishingprotection/apiv1beta1)
[reCAPTCHA Enterprise][cloud-recaptcha]         | alpha        | [`cloud.google.com/go/recaptchaenterprise/apiv1beta1`](https://pkg.go.dev/cloud.google.com/go/recaptchaenterprise/apiv1beta1)
[Recommender][cloud-recommender]                | beta         | [`cloud.google.com/go/recommender/apiv1beta1`](https://pkg.go.dev/cloud.google.com/go/recommender/apiv1beta1)
[Scheduler][cloud-scheduler]                    | stable       | [`cloud.google.com/go/scheduler/apiv1`](https://pkg.go.dev/cloud.google.com/go/scheduler/apiv1)
[Securitycenter][cloud-securitycenter]          | stable       | [`cloud.google.com/go/securitycenter/apiv1`](https://pkg.go.dev/cloud.google.com/go/securitycenter/apiv1)
[Spanner][cloud-spanner]                        | stable       | [`cloud.google.com/go/spanner`](https://pkg.go.dev/cloud.google.com/go/spanner)
[Speech][cloud-speech]                          | stable       | [`cloud.google.com/go/speech/apiv1`](https://pkg.go.dev/cloud.google.com/go/speech/apiv1)
[Storage][cloud-storage]                        | stable       | [`cloud.google.com/go/storage`](https://pkg.go.dev/cloud.google.com/go/storage)
[Talent][cloud-talent]                          | alpha        | [`cloud.google.com/go/talent/apiv4beta1`](https://pkg.go.dev/cloud.google.com/go/talent/apiv4beta1)
[Text To Speech][cloud-texttospeech]            | stable       | [`cloud.google.com/go/texttospeech/apiv1`](https://pkg.go.dev/cloud.google.com/go/texttospeech/apiv1)
[Trace][cloud-trace]                            | stable       | [`cloud.google.com/go/trace/apiv2`](https://pkg.go.dev/cloud.google.com/go/trace/apiv2)
[Translate][cloud-translate]                    | stable       | [`cloud.google.com/go/translate`](https://pkg.go.dev/cloud.google.com/go/translate)
[Video Intelligence][cloud-video]               | beta         | [`cloud.google.com/go/videointelligence/apiv1beta2`](https://pkg.go.dev/cloud.google.com/go/videointelligence/apiv1beta2)
[Vision][cloud-vision]                          | stable       | [`cloud.google.com/go/vision/apiv1`](https://pkg.go.dev/cloud.google.com/go/vision/apiv1)
[Webrisk][cloud-webrisk]                        | alpha        | [`cloud.google.com/go/webrisk/apiv1beta1`](https://pkg.go.dev/cloud.google.com/go/webrisk/apiv1beta1)

> **Alpha status**: the API is still being actively developed. As a
> result, it might change in backward-incompatible ways and is not recommended
> for production use.
>
> **Beta status**: the API is largely complete, but still has outstanding
> features and bugs to be addressed. There may be minor backwards-incompatible
> changes where necessary.
>
> **Stable status**: the API is mature and ready for production use. We will
> continue addressing bugs and feature requests.

Documentation and examples are available at [pkg.go.dev/cloud.google.com/go](https://pkg.go.dev/cloud.google.com/go)

## Go Versions Supported

We support the two most recent major versions of Go. If Google App Engine uses
an older version, we support that as well.

## Authorization

By default, each API will use [Google Application Default Credentials](https://developers.google.com/identity/protocols/application-default-credentials)
for authorization credentials used in calling the API endpoints. This will allow your
application to run in many environments without requiring explicit configuration.

[snip]:# (auth)
```go
client, err := storage.NewClient(ctx)
```

To authorize using a
[JSON key file](https://cloud.google.com/iam/docs/managing-service-account-keys),
pass
[`option.WithCredentialsFile`](https://pkg.go.dev/google.golang.org/api/option#WithCredentialsFile)
to the `NewClient` function of the desired package. For example:

[snip]:# (auth-JSON)
```go
client, err := storage.NewClient(ctx, option.WithCredentialsFile("path/to/keyfile.json"))
```

You can exert more control over authorization by using the
[`golang.org/x/oauth2`](https://pkg.go.dev/golang.org/x/oauth2) package to
create an `oauth2.TokenSource`. Then pass
[`option.WithTokenSource`](https://pkg.go.dev/google.golang.org/api/option#WithTokenSource)
to the `NewClient` function:
[snip]:# (auth-ts)
```go
tokenSource := ...
client, err := storage.NewClient(ctx, option.WithTokenSource(tokenSource))
```

## Contributing

Contributions are welcome. Please, see the
[CONTRIBUTING](https://github.com/GoogleCloudPlatform/google-cloud-go/blob/master/CONTRIBUTING.md)
document for details. We're using Gerrit for our code reviews. Please don't open pull
requests against this repo, new pull requests will be automatically closed.

Please note that this project is released with a Contributor Code of Conduct.
By participating in this project you agree to abide by its terms.
See [Contributor Code of Conduct](https://github.com/GoogleCloudPlatform/google-cloud-go/blob/master/CONTRIBUTING.md#contributor-code-of-conduct)
for more information.

[cloud-asset]: https://cloud.google.com/security-command-center/docs/how-to-asset-inventory
[cloud-automl]: https://cloud.google.com/automl
[cloud-build]: https://cloud.google.com/cloud-build/
[cloud-bigquery]: https://cloud.google.com/bigquery/
[cloud-bigtable]: https://cloud.google.com/bigtable/
[cloud-container]: https://cloud.google.com/containers/
[cloud-containeranalysis]: https://cloud.google.com/container-registry/docs/container-analysis
[cloud-dataproc]: https://cloud.google.com/dataproc/
[cloud-datastore]: https://cloud.google.com/datastore/
[cloud-dialogflow]: https://cloud.google.com/dialogflow-enterprise/
[cloud-debugger]: https://cloud.google.com/debugger/
[cloud-dlp]: https://cloud.google.com/dlp/
[cloud-errors]: https://cloud.google.com/error-reporting/
[cloud-firestore]: https://cloud.google.com/firestore/
[cloud-iam]: https://cloud.google.com/iam/
[cloud-iot]: https://cloud.google.com/iot-core/
[cloud-irm]: https://cloud.google.com/incident-response/docs/concepts
[cloud-kms]: https://cloud.google.com/kms/
[cloud-pubsub]: https://cloud.google.com/pubsub/
[cloud-storage]: https://cloud.google.com/storage/
[cloud-language]: https://cloud.google.com/natural-language
[cloud-logging]: https://cloud.google.com/logging/
[cloud-natural-language]: https://cloud.google.com/natural-language/
[cloud-memorystore]: https://cloud.google.com/memorystore/
[cloud-monitoring]: https://cloud.google.com/monitoring/
[cloud-oslogin]: https://cloud.google.com/compute/docs/oslogin/rest
[cloud-phishingprotection]: https://cloud.google.com/phishing-protection/
[cloud-securitycenter]: https://cloud.google.com/security-command-center/
[cloud-scheduler]: https://cloud.google.com/scheduler
[cloud-spanner]: https://cloud.google.com/spanner/
[cloud-speech]: https://cloud.google.com/speech
[cloud-talent]: https://cloud.google.com/solutions/talent-solution/
[cloud-tasks]: https://cloud.google.com/tasks/
[cloud-texttospeech]: https://cloud.google.com/texttospeech/
[cloud-talent]: https://cloud.google.com/solutions/talent-solution/
[cloud-trace]: https://cloud.google.com/trace/
[cloud-translate]: https://cloud.google.com/translate
[cloud-recaptcha]: https://cloud.google.com/recaptcha-enterprise/
[cloud-recommender]: https://cloud.google.com/recommendations/
[cloud-video]: https://cloud.google.com/video-intelligence/
[cloud-vision]: https://cloud.google.com/vision
[cloud-webrisk]: https://cloud.google.com/web-risk/
