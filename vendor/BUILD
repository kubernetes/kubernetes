package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "cgo_genrule",
    "go_binary",
    "go_library",
    "go_test",
)

go_library(
    name = "bitbucket.org/bertimus9/systemstat",
    srcs = [
        "bitbucket.org/bertimus9/systemstat/systemstat.go",
        "bitbucket.org/bertimus9/systemstat/systemstat_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "bitbucket.org/ww/goautoneg",
    srcs = ["bitbucket.org/ww/goautoneg/autoneg.go"],
    tags = ["automanaged"],
)

go_library(
    name = "cloud.google.com/go/compute/metadata",
    srcs = ["cloud.google.com/go/compute/metadata/metadata.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:cloud.google.com/go/internal",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
    ],
)

go_library(
    name = "cloud.google.com/go/internal",
    srcs = ["cloud.google.com/go/internal/cloud.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Azure/azure-sdk-for-go/arm/compute",
    srcs = [
        "github.com/Azure/azure-sdk-for-go/arm/compute/availabilitysets.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/client.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/models.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/usageoperations.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/version.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachineextensionimages.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachineextensions.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachineimages.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachines.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachinescalesets.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachinescalesetvms.go",
        "github.com/Azure/azure-sdk-for-go/arm/compute/virtualmachinesizes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/Azure/go-autorest/autorest/date",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
        "//vendor:github.com/Azure/go-autorest/autorest/validation",
    ],
)

go_library(
    name = "github.com/Azure/azure-sdk-for-go/arm/containerregistry",
    srcs = [
        "github.com/Azure/azure-sdk-for-go/arm/containerregistry/client.go",
        "github.com/Azure/azure-sdk-for-go/arm/containerregistry/models.go",
        "github.com/Azure/azure-sdk-for-go/arm/containerregistry/registries.go",
        "github.com/Azure/azure-sdk-for-go/arm/containerregistry/version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/Azure/go-autorest/autorest/date",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
        "//vendor:github.com/Azure/go-autorest/autorest/validation",
    ],
)

go_library(
    name = "github.com/Azure/azure-sdk-for-go/arm/network",
    srcs = [
        "github.com/Azure/azure-sdk-for-go/arm/network/applicationgateways.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/client.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/expressroutecircuitauthorizations.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/expressroutecircuitpeerings.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/expressroutecircuits.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/expressrouteserviceproviders.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/interfaces.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/loadbalancers.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/localnetworkgateways.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/models.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/publicipaddresses.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/routes.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/routetables.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/securitygroups.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/securityrules.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/subnets.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/usages.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/version.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/virtualnetworkgatewayconnections.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/virtualnetworkgateways.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/virtualnetworkpeerings.go",
        "github.com/Azure/azure-sdk-for-go/arm/network/virtualnetworks.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
        "//vendor:github.com/Azure/go-autorest/autorest/validation",
    ],
)

go_library(
    name = "github.com/Azure/azure-sdk-for-go/arm/storage",
    srcs = [
        "github.com/Azure/azure-sdk-for-go/arm/storage/accounts.go",
        "github.com/Azure/azure-sdk-for-go/arm/storage/client.go",
        "github.com/Azure/azure-sdk-for-go/arm/storage/models.go",
        "github.com/Azure/azure-sdk-for-go/arm/storage/usageoperations.go",
        "github.com/Azure/azure-sdk-for-go/arm/storage/version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/Azure/go-autorest/autorest/date",
        "//vendor:github.com/Azure/go-autorest/autorest/validation",
    ],
)

go_library(
    name = "github.com/Azure/azure-sdk-for-go/storage",
    srcs = [
        "github.com/Azure/azure-sdk-for-go/storage/blob.go",
        "github.com/Azure/azure-sdk-for-go/storage/client.go",
        "github.com/Azure/azure-sdk-for-go/storage/file.go",
        "github.com/Azure/azure-sdk-for-go/storage/queue.go",
        "github.com/Azure/azure-sdk-for-go/storage/table.go",
        "github.com/Azure/azure-sdk-for-go/storage/table_entities.go",
        "github.com/Azure/azure-sdk-for-go/storage/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Azure/go-ansiterm",
    srcs = [
        "github.com/Azure/go-ansiterm/constants.go",
        "github.com/Azure/go-ansiterm/context.go",
        "github.com/Azure/go-ansiterm/csi_entry_state.go",
        "github.com/Azure/go-ansiterm/csi_param_state.go",
        "github.com/Azure/go-ansiterm/escape_intermediate_state.go",
        "github.com/Azure/go-ansiterm/escape_state.go",
        "github.com/Azure/go-ansiterm/event_handler.go",
        "github.com/Azure/go-ansiterm/ground_state.go",
        "github.com/Azure/go-ansiterm/osc_string_state.go",
        "github.com/Azure/go-ansiterm/parser.go",
        "github.com/Azure/go-ansiterm/parser_action_helpers.go",
        "github.com/Azure/go-ansiterm/parser_actions.go",
        "github.com/Azure/go-ansiterm/parser_test_helpers.go",
        "github.com/Azure/go-ansiterm/parser_test_utilities.go",
        "github.com/Azure/go-ansiterm/states.go",
        "github.com/Azure/go-ansiterm/test_event_handler.go",
        "github.com/Azure/go-ansiterm/utilities.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/Sirupsen/logrus"],
)

go_library(
    name = "github.com/Azure/go-autorest/autorest",
    srcs = [
        "github.com/Azure/go-autorest/autorest/autorest.go",
        "github.com/Azure/go-autorest/autorest/client.go",
        "github.com/Azure/go-autorest/autorest/error.go",
        "github.com/Azure/go-autorest/autorest/preparer.go",
        "github.com/Azure/go-autorest/autorest/responder.go",
        "github.com/Azure/go-autorest/autorest/sender.go",
        "github.com/Azure/go-autorest/autorest/utility.go",
        "github.com/Azure/go-autorest/autorest/version.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Azure/go-autorest/autorest/azure",
    srcs = [
        "github.com/Azure/go-autorest/autorest/azure/async.go",
        "github.com/Azure/go-autorest/autorest/azure/azure.go",
        "github.com/Azure/go-autorest/autorest/azure/config.go",
        "github.com/Azure/go-autorest/autorest/azure/devicetoken.go",
        "github.com/Azure/go-autorest/autorest/azure/environments.go",
        "github.com/Azure/go-autorest/autorest/azure/persist.go",
        "github.com/Azure/go-autorest/autorest/azure/token.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/date",
        "//vendor:github.com/dgrijalva/jwt-go",
    ],
)

go_library(
    name = "github.com/Azure/go-autorest/autorest/date",
    srcs = [
        "github.com/Azure/go-autorest/autorest/date/date.go",
        "github.com/Azure/go-autorest/autorest/date/time.go",
        "github.com/Azure/go-autorest/autorest/date/timerfc1123.go",
        "github.com/Azure/go-autorest/autorest/date/utility.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Azure/go-autorest/autorest/to",
    srcs = ["github.com/Azure/go-autorest/autorest/to/convert.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Azure/go-autorest/autorest/validation",
    srcs = ["github.com/Azure/go-autorest/autorest/validation/validation.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/MakeNowJust/heredoc",
    srcs = ["github.com/MakeNowJust/heredoc/heredoc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Microsoft/go-winio",
    srcs = [
        "github.com/Microsoft/go-winio/backup.go",
        "github.com/Microsoft/go-winio/file.go",
        "github.com/Microsoft/go-winio/fileinfo.go",
        "github.com/Microsoft/go-winio/pipe.go",
        "github.com/Microsoft/go-winio/privilege.go",
        "github.com/Microsoft/go-winio/reparse.go",
        "github.com/Microsoft/go-winio/sd.go",
        "github.com/Microsoft/go-winio/syscall.go",
        "github.com/Microsoft/go-winio/zsyscall.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/PuerkitoBio/purell",
    srcs = ["github.com/PuerkitoBio/purell/purell.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/PuerkitoBio/urlesc",
        "//vendor:golang.org/x/net/idna",
        "//vendor:golang.org/x/text/secure/precis",
        "//vendor:golang.org/x/text/unicode/norm",
    ],
)

go_library(
    name = "github.com/PuerkitoBio/urlesc",
    srcs = ["github.com/PuerkitoBio/urlesc/urlesc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/Sirupsen/logrus",
    srcs = [
        "github.com/Sirupsen/logrus/entry.go",
        "github.com/Sirupsen/logrus/exported.go",
        "github.com/Sirupsen/logrus/formatter.go",
        "github.com/Sirupsen/logrus/hooks.go",
        "github.com/Sirupsen/logrus/json_formatter.go",
        "github.com/Sirupsen/logrus/logger.go",
        "github.com/Sirupsen/logrus/logrus.go",
        "github.com/Sirupsen/logrus/terminal_linux.go",
        "github.com/Sirupsen/logrus/terminal_notwindows.go",
        "github.com/Sirupsen/logrus/text_formatter.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/abbot/go-http-auth",
    srcs = [
        "github.com/abbot/go-http-auth/auth.go",
        "github.com/abbot/go-http-auth/basic.go",
        "github.com/abbot/go-http-auth/digest.go",
        "github.com/abbot/go-http-auth/md5crypt.go",
        "github.com/abbot/go-http-auth/misc.go",
        "github.com/abbot/go-http-auth/users.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/appc/spec/schema",
    srcs = [
        "github.com/appc/spec/schema/doc.go",
        "github.com/appc/spec/schema/image.go",
        "github.com/appc/spec/schema/kind.go",
        "github.com/appc/spec/schema/pod.go",
        "github.com/appc/spec/schema/version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/appc/spec/schema/types",
        "//vendor:go4.org/errorutil",
    ],
)

go_library(
    name = "github.com/appc/spec/schema/common",
    srcs = ["github.com/appc/spec/schema/common/common.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/appc/spec/schema/types",
    srcs = [
        "github.com/appc/spec/schema/types/acidentifier.go",
        "github.com/appc/spec/schema/types/ackind.go",
        "github.com/appc/spec/schema/types/acname.go",
        "github.com/appc/spec/schema/types/annotations.go",
        "github.com/appc/spec/schema/types/app.go",
        "github.com/appc/spec/schema/types/date.go",
        "github.com/appc/spec/schema/types/dependencies.go",
        "github.com/appc/spec/schema/types/doc.go",
        "github.com/appc/spec/schema/types/environment.go",
        "github.com/appc/spec/schema/types/errors.go",
        "github.com/appc/spec/schema/types/event_handler.go",
        "github.com/appc/spec/schema/types/exec.go",
        "github.com/appc/spec/schema/types/hash.go",
        "github.com/appc/spec/schema/types/isolator.go",
        "github.com/appc/spec/schema/types/isolator_linux_specific.go",
        "github.com/appc/spec/schema/types/isolator_resources.go",
        "github.com/appc/spec/schema/types/isolator_unix.go",
        "github.com/appc/spec/schema/types/labels.go",
        "github.com/appc/spec/schema/types/mountpoint.go",
        "github.com/appc/spec/schema/types/port.go",
        "github.com/appc/spec/schema/types/semver.go",
        "github.com/appc/spec/schema/types/url.go",
        "github.com/appc/spec/schema/types/user_annotations.go",
        "github.com/appc/spec/schema/types/user_labels.go",
        "github.com/appc/spec/schema/types/uuid.go",
        "github.com/appc/spec/schema/types/volume.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/appc/spec/schema/common",
        "//vendor:github.com/appc/spec/schema/types/resource",
        "//vendor:github.com/coreos/go-semver/semver",
    ],
)

go_library(
    name = "github.com/asaskevich/govalidator",
    srcs = [
        "github.com/asaskevich/govalidator/arrays.go",
        "github.com/asaskevich/govalidator/converter.go",
        "github.com/asaskevich/govalidator/error.go",
        "github.com/asaskevich/govalidator/numerics.go",
        "github.com/asaskevich/govalidator/patterns.go",
        "github.com/asaskevich/govalidator/types.go",
        "github.com/asaskevich/govalidator/utils.go",
        "github.com/asaskevich/govalidator/validator.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/config.go",
        "github.com/aws/aws-sdk-go/aws/convert_types.go",
        "github.com/aws/aws-sdk-go/aws/errors.go",
        "github.com/aws/aws-sdk-go/aws/logger.go",
        "github.com/aws/aws-sdk-go/aws/types.go",
        "github.com/aws/aws-sdk-go/aws/version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/endpoints",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/awserr",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/awserr/error.go",
        "github.com/aws/aws-sdk-go/aws/awserr/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/awsutil",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/awsutil/copy.go",
        "github.com/aws/aws-sdk-go/aws/awsutil/equal.go",
        "github.com/aws/aws-sdk-go/aws/awsutil/path_value.go",
        "github.com/aws/aws-sdk-go/aws/awsutil/prettify.go",
        "github.com/aws/aws-sdk-go/aws/awsutil/string_value.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/jmespath/go-jmespath"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/client",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/client/client.go",
        "github.com/aws/aws-sdk-go/aws/client/default_retryer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/client/metadata",
    srcs = ["github.com/aws/aws-sdk-go/aws/client/metadata/client_info.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/corehandlers",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/corehandlers/handlers.go",
        "github.com/aws/aws-sdk-go/aws/corehandlers/param_validator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/credentials",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/credentials/chain_provider.go",
        "github.com/aws/aws-sdk-go/aws/credentials/credentials.go",
        "github.com/aws/aws-sdk-go/aws/credentials/env_provider.go",
        "github.com/aws/aws-sdk-go/aws/credentials/shared_credentials_provider.go",
        "github.com/aws/aws-sdk-go/aws/credentials/static_provider.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/go-ini/ini",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds",
    srcs = ["github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds/ec2_role_provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/ec2metadata",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/credentials/endpointcreds",
    srcs = ["github.com/aws/aws-sdk-go/aws/credentials/endpointcreds/provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/credentials/stscreds",
    srcs = ["github.com/aws/aws-sdk-go/aws/credentials/stscreds/assume_role_provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/service/sts",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/defaults",
    srcs = ["github.com/aws/aws-sdk-go/aws/defaults/defaults.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/corehandlers",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials/endpointcreds",
        "//vendor:github.com/aws/aws-sdk-go/aws/ec2metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/endpoints",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/ec2metadata",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/ec2metadata/api.go",
        "github.com/aws/aws-sdk-go/aws/ec2metadata/service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/endpoints",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/endpoints/decode.go",
        "github.com/aws/aws-sdk-go/aws/endpoints/defaults.go",
        "github.com/aws/aws-sdk-go/aws/endpoints/doc.go",
        "github.com/aws/aws-sdk-go/aws/endpoints/endpoints.go",
        "github.com/aws/aws-sdk-go/aws/endpoints/v3model.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/aws/aws-sdk-go/aws/awserr"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/request",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/request/handlers.go",
        "github.com/aws/aws-sdk-go/aws/request/http_request.go",
        "github.com/aws/aws-sdk-go/aws/request/offset_reader.go",
        "github.com/aws/aws-sdk-go/aws/request/request.go",
        "github.com/aws/aws-sdk-go/aws/request/request_1_7.go",
        "github.com/aws/aws-sdk-go/aws/request/request_pagination.go",
        "github.com/aws/aws-sdk-go/aws/request/retryer.go",
        "github.com/aws/aws-sdk-go/aws/request/validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/session",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/session/doc.go",
        "github.com/aws/aws-sdk-go/aws/session/env_config.go",
        "github.com/aws/aws-sdk-go/aws/session/session.go",
        "github.com/aws/aws-sdk-go/aws/session/shared_config.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/corehandlers",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials/stscreds",
        "//vendor:github.com/aws/aws-sdk-go/aws/defaults",
        "//vendor:github.com/aws/aws-sdk-go/aws/endpoints",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/go-ini/ini",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/aws/signer/v4",
    srcs = [
        "github.com/aws/aws-sdk-go/aws/signer/v4/header_rules.go",
        "github.com/aws/aws-sdk-go/aws/signer/v4/uri_path.go",
        "github.com/aws/aws-sdk-go/aws/signer/v4/v4.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/credentials",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/rest",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/idempotency.go",
        "github.com/aws/aws-sdk-go/private/protocol/unmarshal.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/aws/aws-sdk-go/aws/request"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/ec2query",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/ec2query/build.go",
        "github.com/aws/aws-sdk-go/private/protocol/ec2query/unmarshal.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query/queryutil",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/json/jsonutil",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/json/jsonutil/build.go",
        "github.com/aws/aws-sdk-go/private/protocol/json/jsonutil/unmarshal.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/aws/aws-sdk-go/private/protocol"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/jsonrpc",
    srcs = ["github.com/aws/aws-sdk-go/private/protocol/jsonrpc/jsonrpc.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/json/jsonutil",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/rest",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/query",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/query/build.go",
        "github.com/aws/aws-sdk-go/private/protocol/query/unmarshal.go",
        "github.com/aws/aws-sdk-go/private/protocol/query/unmarshal_error.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query/queryutil",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/query/queryutil",
    srcs = ["github.com/aws/aws-sdk-go/private/protocol/query/queryutil/queryutil.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/aws/aws-sdk-go/private/protocol"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/rest",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/rest/build.go",
        "github.com/aws/aws-sdk-go/private/protocol/rest/payload.go",
        "github.com/aws/aws-sdk-go/private/protocol/rest/unmarshal.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/restxml",
    srcs = ["github.com/aws/aws-sdk-go/private/protocol/restxml/restxml.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/rest",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    srcs = [
        "github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil/build.go",
        "github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil/unmarshal.go",
        "github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil/xml_to_struct.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/aws/aws-sdk-go/private/protocol"],
)

go_library(
    name = "github.com/aws/aws-sdk-go/private/waiter",
    srcs = ["github.com/aws/aws-sdk-go/private/waiter/waiter.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/autoscaling",
    srcs = [
        "github.com/aws/aws-sdk-go/service/autoscaling/api.go",
        "github.com/aws/aws-sdk-go/service/autoscaling/service.go",
        "github.com/aws/aws-sdk-go/service/autoscaling/waiters.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query",
        "//vendor:github.com/aws/aws-sdk-go/private/waiter",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/ec2",
    srcs = [
        "github.com/aws/aws-sdk-go/service/ec2/api.go",
        "github.com/aws/aws-sdk-go/service/ec2/customizations.go",
        "github.com/aws/aws-sdk-go/service/ec2/service.go",
        "github.com/aws/aws-sdk-go/service/ec2/waiters.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/endpoints",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/ec2query",
        "//vendor:github.com/aws/aws-sdk-go/private/waiter",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/ecr",
    srcs = [
        "github.com/aws/aws-sdk-go/service/ecr/api.go",
        "github.com/aws/aws-sdk-go/service/ecr/service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/jsonrpc",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/elb",
    srcs = [
        "github.com/aws/aws-sdk-go/service/elb/api.go",
        "github.com/aws/aws-sdk-go/service/elb/service.go",
        "github.com/aws/aws-sdk-go/service/elb/waiters.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query",
        "//vendor:github.com/aws/aws-sdk-go/private/waiter",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/route53",
    srcs = [
        "github.com/aws/aws-sdk-go/service/route53/api.go",
        "github.com/aws/aws-sdk-go/service/route53/customizations.go",
        "github.com/aws/aws-sdk-go/service/route53/service.go",
        "github.com/aws/aws-sdk-go/service/route53/unmarshal_error.go",
        "github.com/aws/aws-sdk-go/service/route53/waiters.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awserr",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/restxml",
        "//vendor:github.com/aws/aws-sdk-go/private/waiter",
    ],
)

go_library(
    name = "github.com/aws/aws-sdk-go/service/sts",
    srcs = [
        "github.com/aws/aws-sdk-go/service/sts/api.go",
        "github.com/aws/aws-sdk-go/service/sts/customizations.go",
        "github.com/aws/aws-sdk-go/service/sts/service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/awsutil",
        "//vendor:github.com/aws/aws-sdk-go/aws/client",
        "//vendor:github.com/aws/aws-sdk-go/aws/client/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/signer/v4",
        "//vendor:github.com/aws/aws-sdk-go/private/protocol/query",
    ],
)

go_library(
    name = "github.com/beorn7/perks/quantile",
    srcs = ["github.com/beorn7/perks/quantile/stream.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/blang/semver",
    srcs = [
        "github.com/blang/semver/json.go",
        "github.com/blang/semver/semver.go",
        "github.com/blang/semver/sort.go",
        "github.com/blang/semver/sql.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/boltdb/bolt",
    srcs = [
        "github.com/boltdb/bolt/bolt_amd64.go",
        "github.com/boltdb/bolt/bolt_linux.go",
        "github.com/boltdb/bolt/bolt_unix.go",
        "github.com/boltdb/bolt/bucket.go",
        "github.com/boltdb/bolt/cursor.go",
        "github.com/boltdb/bolt/db.go",
        "github.com/boltdb/bolt/doc.go",
        "github.com/boltdb/bolt/errors.go",
        "github.com/boltdb/bolt/freelist.go",
        "github.com/boltdb/bolt/node.go",
        "github.com/boltdb/bolt/page.go",
        "github.com/boltdb/bolt/tx.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/chai2010/gettext-go/gettext",
    srcs = [
        "github.com/chai2010/gettext-go/gettext/caller.go",
        "github.com/chai2010/gettext-go/gettext/doc.go",
        "github.com/chai2010/gettext-go/gettext/domain.go",
        "github.com/chai2010/gettext-go/gettext/domain_helper.go",
        "github.com/chai2010/gettext-go/gettext/fs.go",
        "github.com/chai2010/gettext-go/gettext/gettext.go",
        "github.com/chai2010/gettext-go/gettext/local.go",
        "github.com/chai2010/gettext-go/gettext/tr.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/chai2010/gettext-go/gettext/mo",
        "//vendor:github.com/chai2010/gettext-go/gettext/plural",
        "//vendor:github.com/chai2010/gettext-go/gettext/po",
    ],
)

go_library(
    name = "github.com/chai2010/gettext-go/gettext/mo",
    srcs = [
        "github.com/chai2010/gettext-go/gettext/mo/doc.go",
        "github.com/chai2010/gettext-go/gettext/mo/encoder.go",
        "github.com/chai2010/gettext-go/gettext/mo/file.go",
        "github.com/chai2010/gettext-go/gettext/mo/header.go",
        "github.com/chai2010/gettext-go/gettext/mo/message.go",
        "github.com/chai2010/gettext-go/gettext/mo/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/chai2010/gettext-go/gettext/plural",
    srcs = [
        "github.com/chai2010/gettext-go/gettext/plural/doc.go",
        "github.com/chai2010/gettext-go/gettext/plural/formula.go",
        "github.com/chai2010/gettext-go/gettext/plural/table.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/chai2010/gettext-go/gettext/po",
    srcs = [
        "github.com/chai2010/gettext-go/gettext/po/comment.go",
        "github.com/chai2010/gettext-go/gettext/po/doc.go",
        "github.com/chai2010/gettext-go/gettext/po/file.go",
        "github.com/chai2010/gettext-go/gettext/po/header.go",
        "github.com/chai2010/gettext-go/gettext/po/line_reader.go",
        "github.com/chai2010/gettext-go/gettext/po/message.go",
        "github.com/chai2010/gettext-go/gettext/po/re.go",
        "github.com/chai2010/gettext-go/gettext/po/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/auth",
    srcs = ["github.com/cloudflare/cfssl/auth/auth.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/certdb",
    srcs = ["github.com/cloudflare/cfssl/certdb/certdb.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/config",
    srcs = ["github.com/cloudflare/cfssl/config/config.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cloudflare/cfssl/auth",
        "//vendor:github.com/cloudflare/cfssl/errors",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/log",
        "//vendor:github.com/cloudflare/cfssl/ocsp/config",
    ],
)

go_library(
    name = "github.com/cloudflare/cfssl/crypto/pkcs7",
    srcs = ["github.com/cloudflare/cfssl/crypto/pkcs7/pkcs7.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/cloudflare/cfssl/errors"],
)

go_library(
    name = "github.com/cloudflare/cfssl/csr",
    srcs = ["github.com/cloudflare/cfssl/csr/csr.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cloudflare/cfssl/errors",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/log",
    ],
)

go_library(
    name = "github.com/cloudflare/cfssl/errors",
    srcs = [
        "github.com/cloudflare/cfssl/errors/doc.go",
        "github.com/cloudflare/cfssl/errors/error.go",
        "github.com/cloudflare/cfssl/errors/http.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/helpers",
    srcs = ["github.com/cloudflare/cfssl/helpers/helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cloudflare/cfssl/crypto/pkcs7",
        "//vendor:github.com/cloudflare/cfssl/errors",
        "//vendor:github.com/cloudflare/cfssl/helpers/derhelpers",
        "//vendor:github.com/cloudflare/cfssl/log",
        "//vendor:golang.org/x/crypto/pkcs12",
    ],
)

go_library(
    name = "github.com/cloudflare/cfssl/helpers/derhelpers",
    srcs = ["github.com/cloudflare/cfssl/helpers/derhelpers/derhelpers.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/cloudflare/cfssl/errors"],
)

go_library(
    name = "github.com/cloudflare/cfssl/info",
    srcs = ["github.com/cloudflare/cfssl/info/info.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/log",
    srcs = ["github.com/cloudflare/cfssl/log/log.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/ocsp/config",
    srcs = ["github.com/cloudflare/cfssl/ocsp/config/config.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/cloudflare/cfssl/signer",
    srcs = ["github.com/cloudflare/cfssl/signer/signer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cloudflare/cfssl/certdb",
        "//vendor:github.com/cloudflare/cfssl/config",
        "//vendor:github.com/cloudflare/cfssl/csr",
        "//vendor:github.com/cloudflare/cfssl/errors",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/info",
    ],
)

go_library(
    name = "github.com/cloudflare/cfssl/signer/local",
    srcs = ["github.com/cloudflare/cfssl/signer/local/local.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cloudflare/cfssl/certdb",
        "//vendor:github.com/cloudflare/cfssl/config",
        "//vendor:github.com/cloudflare/cfssl/errors",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/info",
        "//vendor:github.com/cloudflare/cfssl/log",
        "//vendor:github.com/cloudflare/cfssl/signer",
        "//vendor:github.com/google/certificate-transparency/go",
        "//vendor:github.com/google/certificate-transparency/go/client",
    ],
)

go_library(
    name = "github.com/clusterhq/flocker-go",
    srcs = [
        "github.com/clusterhq/flocker-go/client.go",
        "github.com/clusterhq/flocker-go/doc.go",
        "github.com/clusterhq/flocker-go/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/codegangsta/negroni",
    srcs = [
        "github.com/codegangsta/negroni/doc.go",
        "github.com/codegangsta/negroni/logger.go",
        "github.com/codegangsta/negroni/negroni.go",
        "github.com/codegangsta/negroni/recovery.go",
        "github.com/codegangsta/negroni/response_writer.go",
        "github.com/codegangsta/negroni/static.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/containernetworking/cni/libcni",
    srcs = [
        "github.com/containernetworking/cni/libcni/api.go",
        "github.com/containernetworking/cni/libcni/conf.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/containernetworking/cni/pkg/invoke",
        "//vendor:github.com/containernetworking/cni/pkg/types",
    ],
)

go_library(
    name = "github.com/containernetworking/cni/pkg/invoke",
    srcs = [
        "github.com/containernetworking/cni/pkg/invoke/args.go",
        "github.com/containernetworking/cni/pkg/invoke/delegate.go",
        "github.com/containernetworking/cni/pkg/invoke/exec.go",
        "github.com/containernetworking/cni/pkg/invoke/find.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/containernetworking/cni/pkg/types"],
)

go_library(
    name = "github.com/containernetworking/cni/pkg/types",
    srcs = [
        "github.com/containernetworking/cni/pkg/types/args.go",
        "github.com/containernetworking/cni/pkg/types/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/alarm",
    srcs = ["github.com/coreos/etcd/alarm/alarms.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/auth",
    srcs = [
        "github.com/coreos/etcd/auth/doc.go",
        "github.com/coreos/etcd/auth/range_perm_cache.go",
        "github.com/coreos/etcd/auth/simple_token.go",
        "github.com/coreos/etcd/auth/store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/auth/authpb",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:golang.org/x/crypto/bcrypt",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/auth/authpb",
    srcs = ["github.com/coreos/etcd/auth/authpb/auth.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/etcd/client",
    srcs = [
        "github.com/coreos/etcd/client/auth_role.go",
        "github.com/coreos/etcd/client/auth_user.go",
        "github.com/coreos/etcd/client/cancelreq.go",
        "github.com/coreos/etcd/client/client.go",
        "github.com/coreos/etcd/client/cluster_error.go",
        "github.com/coreos/etcd/client/curl.go",
        "github.com/coreos/etcd/client/discover.go",
        "github.com/coreos/etcd/client/doc.go",
        "github.com/coreos/etcd/client/keys.generated.go",
        "github.com/coreos/etcd/client/keys.go",
        "github.com/coreos/etcd/client/members.go",
        "github.com/coreos/etcd/client/srv.go",
        "github.com/coreos/etcd/client/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/pathutil",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/clientv3",
    srcs = [
        "github.com/coreos/etcd/clientv3/auth.go",
        "github.com/coreos/etcd/clientv3/balancer.go",
        "github.com/coreos/etcd/clientv3/client.go",
        "github.com/coreos/etcd/clientv3/cluster.go",
        "github.com/coreos/etcd/clientv3/compact_op.go",
        "github.com/coreos/etcd/clientv3/compare.go",
        "github.com/coreos/etcd/clientv3/config.go",
        "github.com/coreos/etcd/clientv3/doc.go",
        "github.com/coreos/etcd/clientv3/kv.go",
        "github.com/coreos/etcd/clientv3/lease.go",
        "github.com/coreos/etcd/clientv3/logger.go",
        "github.com/coreos/etcd/clientv3/maintenance.go",
        "github.com/coreos/etcd/clientv3/op.go",
        "github.com/coreos/etcd/clientv3/retry.go",
        "github.com/coreos/etcd/clientv3/sort.go",
        "github.com/coreos/etcd/clientv3/txn.go",
        "github.com/coreos/etcd/clientv3/watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/auth/authpb",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/coreos/etcd/pkg/tlsutil",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/credentials",
        "//vendor:google.golang.org/grpc/grpclog",
        "//vendor:google.golang.org/grpc/metadata",
    ],
)

go_library(
    name = "github.com/coreos/etcd/compactor",
    srcs = [
        "github.com/coreos/etcd/compactor/compactor.go",
        "github.com/coreos/etcd/compactor/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/mvcc",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/jonboulle/clockwork",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/discovery",
    srcs = [
        "github.com/coreos/etcd/discovery/discovery.go",
        "github.com/coreos/etcd/discovery/srv.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/jonboulle/clockwork",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/error",
    srcs = ["github.com/coreos/etcd/error/error.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver",
    srcs = [
        "github.com/coreos/etcd/etcdserver/apply.go",
        "github.com/coreos/etcd/etcdserver/apply_auth.go",
        "github.com/coreos/etcd/etcdserver/apply_v2.go",
        "github.com/coreos/etcd/etcdserver/cluster_util.go",
        "github.com/coreos/etcd/etcdserver/config.go",
        "github.com/coreos/etcd/etcdserver/consistent_index.go",
        "github.com/coreos/etcd/etcdserver/doc.go",
        "github.com/coreos/etcd/etcdserver/errors.go",
        "github.com/coreos/etcd/etcdserver/metrics.go",
        "github.com/coreos/etcd/etcdserver/quota.go",
        "github.com/coreos/etcd/etcdserver/raft.go",
        "github.com/coreos/etcd/etcdserver/server.go",
        "github.com/coreos/etcd/etcdserver/snapshot_merge.go",
        "github.com/coreos/etcd/etcdserver/storage.go",
        "github.com/coreos/etcd/etcdserver/util.go",
        "github.com/coreos/etcd/etcdserver/v2_server.go",
        "github.com/coreos/etcd/etcdserver/v3_server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/alarm",
        "//vendor:github.com/coreos/etcd/auth",
        "//vendor:github.com/coreos/etcd/compactor",
        "//vendor:github.com/coreos/etcd/discovery",
        "//vendor:github.com/coreos/etcd/etcdserver/api",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http/httptypes",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/etcdserver/membership",
        "//vendor:github.com/coreos/etcd/etcdserver/stats",
        "//vendor:github.com/coreos/etcd/lease",
        "//vendor:github.com/coreos/etcd/lease/leasehttp",
        "//vendor:github.com/coreos/etcd/mvcc",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/coreos/etcd/pkg/contention",
        "//vendor:github.com/coreos/etcd/pkg/fileutil",
        "//vendor:github.com/coreos/etcd/pkg/httputil",
        "//vendor:github.com/coreos/etcd/pkg/idutil",
        "//vendor:github.com/coreos/etcd/pkg/netutil",
        "//vendor:github.com/coreos/etcd/pkg/pbutil",
        "//vendor:github.com/coreos/etcd/pkg/runtime",
        "//vendor:github.com/coreos/etcd/pkg/schedule",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/pkg/wait",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/rafthttp",
        "//vendor:github.com/coreos/etcd/snap",
        "//vendor:github.com/coreos/etcd/store",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/etcd/wal",
        "//vendor:github.com/coreos/etcd/wal/walpb",
        "//vendor:github.com/coreos/go-semver/semver",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc/metadata",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/api",
    srcs = [
        "github.com/coreos/etcd/etcdserver/api/capability.go",
        "github.com/coreos/etcd/etcdserver/api/cluster.go",
        "github.com/coreos/etcd/etcdserver/api/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/etcdserver/membership",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/go-semver/semver",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/api/v2http",
    srcs = [
        "github.com/coreos/etcd/etcdserver/api/v2http/capability.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/client.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/client_auth.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/doc.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/http.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/metrics.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/peer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/error",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http/httptypes",
        "//vendor:github.com/coreos/etcd/etcdserver/auth",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/etcdserver/membership",
        "//vendor:github.com/coreos/etcd/etcdserver/stats",
        "//vendor:github.com/coreos/etcd/lease/leasehttp",
        "//vendor:github.com/coreos/etcd/pkg/logutil",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/rafthttp",
        "//vendor:github.com/coreos/etcd/store",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/jonboulle/clockwork",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/api/v2http/httptypes",
    srcs = [
        "github.com/coreos/etcd/etcdserver/api/v2http/httptypes/errors.go",
        "github.com/coreos/etcd/etcdserver/api/v2http/httptypes/member.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/api/v3rpc",
    srcs = [
        "github.com/coreos/etcd/etcdserver/api/v3rpc/auth.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/codec.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/grpc.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/header.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/interceptor.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/key.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/lease.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/maintenance.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/member.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/metrics.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/quota.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/util.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/auth",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/etcdserver/membership",
        "//vendor:github.com/coreos/etcd/lease",
        "//vendor:github.com/coreos/etcd/mvcc",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:google.golang.org/grpc/credentials",
        "//vendor:google.golang.org/grpc/grpclog",
        "//vendor:google.golang.org/grpc/metadata",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
    srcs = [
        "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes/doc.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes/error.go",
        "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes/md.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/codes",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/auth",
    srcs = [
        "github.com/coreos/etcd/etcdserver/auth/auth.go",
        "github.com/coreos/etcd/etcdserver/auth/auth_requests.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/error",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:golang.org/x/crypto/bcrypt",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/etcdserverpb",
    srcs = [
        "github.com/coreos/etcd/etcdserver/etcdserverpb/etcdserver.pb.go",
        "github.com/coreos/etcd/etcdserver/etcdserverpb/raft_internal.pb.go",
        "github.com/coreos/etcd/etcdserver/etcdserverpb/rpc.pb.go",
        "github.com/coreos/etcd/etcdserver/etcdserverpb/rpc.pb.gw.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/auth/authpb",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/grpc-ecosystem/grpc-gateway/runtime",
        "//vendor:github.com/grpc-ecosystem/grpc-gateway/utilities",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:google.golang.org/grpc/grpclog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/membership",
    srcs = [
        "github.com/coreos/etcd/etcdserver/membership/cluster.go",
        "github.com/coreos/etcd/etcdserver/membership/errors.go",
        "github.com/coreos/etcd/etcdserver/membership/member.go",
        "github.com/coreos/etcd/etcdserver/membership/store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/error",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/etcd/pkg/netutil",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/store",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/go-semver/semver",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/etcdserver/stats",
    srcs = [
        "github.com/coreos/etcd/etcdserver/stats/leader.go",
        "github.com/coreos/etcd/etcdserver/stats/queue.go",
        "github.com/coreos/etcd/etcdserver/stats/server.go",
        "github.com/coreos/etcd/etcdserver/stats/stats.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/integration",
    srcs = [
        "github.com/coreos/etcd/integration/bridge.go",
        "github.com/coreos/etcd/integration/cluster.go",
        "github.com/coreos/etcd/integration/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc",
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/pkg/testutil",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/rafthttp",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)

go_library(
    name = "github.com/coreos/etcd/lease",
    srcs = [
        "github.com/coreos/etcd/lease/doc.go",
        "github.com/coreos/etcd/lease/lessor.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/lease/leasepb",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
    ],
)

go_library(
    name = "github.com/coreos/etcd/lease/leasehttp",
    srcs = [
        "github.com/coreos/etcd/lease/leasehttp/doc.go",
        "github.com/coreos/etcd/lease/leasehttp/http.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/etcdserver/etcdserverpb",
        "//vendor:github.com/coreos/etcd/lease",
    ],
)

go_library(
    name = "github.com/coreos/etcd/lease/leasepb",
    srcs = ["github.com/coreos/etcd/lease/leasepb/lease.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/etcd/mvcc",
    srcs = [
        "github.com/coreos/etcd/mvcc/doc.go",
        "github.com/coreos/etcd/mvcc/index.go",
        "github.com/coreos/etcd/mvcc/key_index.go",
        "github.com/coreos/etcd/mvcc/kv.go",
        "github.com/coreos/etcd/mvcc/kvstore.go",
        "github.com/coreos/etcd/mvcc/kvstore_compaction.go",
        "github.com/coreos/etcd/mvcc/metrics.go",
        "github.com/coreos/etcd/mvcc/revision.go",
        "github.com/coreos/etcd/mvcc/util.go",
        "github.com/coreos/etcd/mvcc/watchable_store.go",
        "github.com/coreos/etcd/mvcc/watcher.go",
        "github.com/coreos/etcd/mvcc/watcher_group.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/lease",
        "//vendor:github.com/coreos/etcd/mvcc/backend",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/coreos/etcd/pkg/adt",
        "//vendor:github.com/coreos/etcd/pkg/schedule",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/google/btree",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/mvcc/backend",
    srcs = [
        "github.com/coreos/etcd/mvcc/backend/backend.go",
        "github.com/coreos/etcd/mvcc/backend/batch_tx.go",
        "github.com/coreos/etcd/mvcc/backend/boltoption_linux.go",
        "github.com/coreos/etcd/mvcc/backend/doc.go",
        "github.com/coreos/etcd/mvcc/backend/metrics.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/boltdb/bolt",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/coreos/etcd/mvcc/mvccpb",
    srcs = ["github.com/coreos/etcd/mvcc/mvccpb/kv.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/adt",
    srcs = [
        "github.com/coreos/etcd/pkg/adt/doc.go",
        "github.com/coreos/etcd/pkg/adt/interval_tree.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/contention",
    srcs = [
        "github.com/coreos/etcd/pkg/contention/contention.go",
        "github.com/coreos/etcd/pkg/contention/doc.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/crc",
    srcs = ["github.com/coreos/etcd/pkg/crc/crc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/fileutil",
    srcs = [
        "github.com/coreos/etcd/pkg/fileutil/dir_unix.go",
        "github.com/coreos/etcd/pkg/fileutil/fileutil.go",
        "github.com/coreos/etcd/pkg/fileutil/lock.go",
        "github.com/coreos/etcd/pkg/fileutil/lock_flock.go",
        "github.com/coreos/etcd/pkg/fileutil/lock_linux.go",
        "github.com/coreos/etcd/pkg/fileutil/preallocate.go",
        "github.com/coreos/etcd/pkg/fileutil/preallocate_unix.go",
        "github.com/coreos/etcd/pkg/fileutil/purge.go",
        "github.com/coreos/etcd/pkg/fileutil/sync_linux.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/capnslog"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/httputil",
    srcs = ["github.com/coreos/etcd/pkg/httputil/httputil.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/idutil",
    srcs = ["github.com/coreos/etcd/pkg/idutil/id.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/ioutil",
    srcs = [
        "github.com/coreos/etcd/pkg/ioutil/pagewriter.go",
        "github.com/coreos/etcd/pkg/ioutil/readcloser.go",
        "github.com/coreos/etcd/pkg/ioutil/reader.go",
        "github.com/coreos/etcd/pkg/ioutil/util.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/etcd/pkg/fileutil"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/logutil",
    srcs = ["github.com/coreos/etcd/pkg/logutil/merge_logger.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/capnslog"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/netutil",
    srcs = [
        "github.com/coreos/etcd/pkg/netutil/isolate_linux.go",
        "github.com/coreos/etcd/pkg/netutil/netutil.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
    ],
)

go_library(
    name = "github.com/coreos/etcd/pkg/pathutil",
    srcs = ["github.com/coreos/etcd/pkg/pathutil/path.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/pbutil",
    srcs = ["github.com/coreos/etcd/pkg/pbutil/pbutil.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/capnslog"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/runtime",
    srcs = ["github.com/coreos/etcd/pkg/runtime/fds_linux.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/schedule",
    srcs = [
        "github.com/coreos/etcd/pkg/schedule/doc.go",
        "github.com/coreos/etcd/pkg/schedule/schedule.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/testutil",
    srcs = [
        "github.com/coreos/etcd/pkg/testutil/leak.go",
        "github.com/coreos/etcd/pkg/testutil/pauseable_handler.go",
        "github.com/coreos/etcd/pkg/testutil/recorder.go",
        "github.com/coreos/etcd/pkg/testutil/testutil.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/tlsutil",
    srcs = [
        "github.com/coreos/etcd/pkg/tlsutil/doc.go",
        "github.com/coreos/etcd/pkg/tlsutil/tlsutil.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/transport",
    srcs = [
        "github.com/coreos/etcd/pkg/transport/doc.go",
        "github.com/coreos/etcd/pkg/transport/keepalive_listener.go",
        "github.com/coreos/etcd/pkg/transport/limit_listen.go",
        "github.com/coreos/etcd/pkg/transport/listener.go",
        "github.com/coreos/etcd/pkg/transport/timeout_conn.go",
        "github.com/coreos/etcd/pkg/transport/timeout_dialer.go",
        "github.com/coreos/etcd/pkg/transport/timeout_listener.go",
        "github.com/coreos/etcd/pkg/transport/timeout_transport.go",
        "github.com/coreos/etcd/pkg/transport/tls.go",
        "github.com/coreos/etcd/pkg/transport/transport.go",
        "github.com/coreos/etcd/pkg/transport/unix_listener.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/fileutil",
        "//vendor:github.com/coreos/etcd/pkg/tlsutil",
    ],
)

go_library(
    name = "github.com/coreos/etcd/pkg/types",
    srcs = [
        "github.com/coreos/etcd/pkg/types/doc.go",
        "github.com/coreos/etcd/pkg/types/id.go",
        "github.com/coreos/etcd/pkg/types/set.go",
        "github.com/coreos/etcd/pkg/types/slice.go",
        "github.com/coreos/etcd/pkg/types/urls.go",
        "github.com/coreos/etcd/pkg/types/urlsmap.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/pkg/wait",
    srcs = [
        "github.com/coreos/etcd/pkg/wait/wait.go",
        "github.com/coreos/etcd/pkg/wait/wait_time.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/etcd/raft",
    srcs = [
        "github.com/coreos/etcd/raft/doc.go",
        "github.com/coreos/etcd/raft/log.go",
        "github.com/coreos/etcd/raft/log_unstable.go",
        "github.com/coreos/etcd/raft/logger.go",
        "github.com/coreos/etcd/raft/node.go",
        "github.com/coreos/etcd/raft/progress.go",
        "github.com/coreos/etcd/raft/raft.go",
        "github.com/coreos/etcd/raft/rawnode.go",
        "github.com/coreos/etcd/raft/status.go",
        "github.com/coreos/etcd/raft/storage.go",
        "github.com/coreos/etcd/raft/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/raft/raftpb",
    srcs = ["github.com/coreos/etcd/raft/raftpb/raft.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/etcd/rafthttp",
    srcs = [
        "github.com/coreos/etcd/rafthttp/coder.go",
        "github.com/coreos/etcd/rafthttp/doc.go",
        "github.com/coreos/etcd/rafthttp/http.go",
        "github.com/coreos/etcd/rafthttp/metrics.go",
        "github.com/coreos/etcd/rafthttp/msg_codec.go",
        "github.com/coreos/etcd/rafthttp/msgappv2_codec.go",
        "github.com/coreos/etcd/rafthttp/peer.go",
        "github.com/coreos/etcd/rafthttp/peer_status.go",
        "github.com/coreos/etcd/rafthttp/pipeline.go",
        "github.com/coreos/etcd/rafthttp/probing_status.go",
        "github.com/coreos/etcd/rafthttp/remote.go",
        "github.com/coreos/etcd/rafthttp/snapshot_sender.go",
        "github.com/coreos/etcd/rafthttp/stream.go",
        "github.com/coreos/etcd/rafthttp/transport.go",
        "github.com/coreos/etcd/rafthttp/urlpick.go",
        "github.com/coreos/etcd/rafthttp/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/etcdserver/stats",
        "//vendor:github.com/coreos/etcd/pkg/httputil",
        "//vendor:github.com/coreos/etcd/pkg/ioutil",
        "//vendor:github.com/coreos/etcd/pkg/logutil",
        "//vendor:github.com/coreos/etcd/pkg/pbutil",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/snap",
        "//vendor:github.com/coreos/etcd/version",
        "//vendor:github.com/coreos/go-semver/semver",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:github.com/xiang90/probing",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/coreos/etcd/snap",
    srcs = [
        "github.com/coreos/etcd/snap/db.go",
        "github.com/coreos/etcd/snap/message.go",
        "github.com/coreos/etcd/snap/metrics.go",
        "github.com/coreos/etcd/snap/snapshotter.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/fileutil",
        "//vendor:github.com/coreos/etcd/pkg/ioutil",
        "//vendor:github.com/coreos/etcd/pkg/pbutil",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/snap/snappb",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/coreos/etcd/snap/snappb",
    srcs = ["github.com/coreos/etcd/snap/snappb/snap.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/etcd/store",
    srcs = [
        "github.com/coreos/etcd/store/doc.go",
        "github.com/coreos/etcd/store/event.go",
        "github.com/coreos/etcd/store/event_history.go",
        "github.com/coreos/etcd/store/event_queue.go",
        "github.com/coreos/etcd/store/metrics.go",
        "github.com/coreos/etcd/store/node.go",
        "github.com/coreos/etcd/store/node_extern.go",
        "github.com/coreos/etcd/store/stats.go",
        "github.com/coreos/etcd/store/store.go",
        "github.com/coreos/etcd/store/ttl_key_heap.go",
        "github.com/coreos/etcd/store/watcher.go",
        "github.com/coreos/etcd/store/watcher_hub.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/error",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/jonboulle/clockwork",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/coreos/etcd/version",
    srcs = ["github.com/coreos/etcd/version/version.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/fileutil",
        "//vendor:github.com/coreos/etcd/pkg/types",
    ],
)

go_library(
    name = "github.com/coreos/etcd/wal",
    srcs = [
        "github.com/coreos/etcd/wal/decoder.go",
        "github.com/coreos/etcd/wal/doc.go",
        "github.com/coreos/etcd/wal/encoder.go",
        "github.com/coreos/etcd/wal/file_pipeline.go",
        "github.com/coreos/etcd/wal/metrics.go",
        "github.com/coreos/etcd/wal/repair.go",
        "github.com/coreos/etcd/wal/util.go",
        "github.com/coreos/etcd/wal/wal.go",
        "github.com/coreos/etcd/wal/wal_unix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/pkg/crc",
        "//vendor:github.com/coreos/etcd/pkg/fileutil",
        "//vendor:github.com/coreos/etcd/pkg/ioutil",
        "//vendor:github.com/coreos/etcd/pkg/pbutil",
        "//vendor:github.com/coreos/etcd/raft",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/wal/walpb",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/coreos/etcd/wal/walpb",
    srcs = [
        "github.com/coreos/etcd/wal/walpb/record.go",
        "github.com/coreos/etcd/wal/walpb/record.pb.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/coreos/go-oidc/http",
    srcs = [
        "github.com/coreos/go-oidc/http/client.go",
        "github.com/coreos/go-oidc/http/http.go",
        "github.com/coreos/go-oidc/http/url.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-oidc/jose",
    srcs = [
        "github.com/coreos/go-oidc/jose/claims.go",
        "github.com/coreos/go-oidc/jose/jose.go",
        "github.com/coreos/go-oidc/jose/jwk.go",
        "github.com/coreos/go-oidc/jose/jws.go",
        "github.com/coreos/go-oidc/jose/jwt.go",
        "github.com/coreos/go-oidc/jose/sig.go",
        "github.com/coreos/go-oidc/jose/sig_hmac.go",
        "github.com/coreos/go-oidc/jose/sig_rsa.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-oidc/key",
    srcs = [
        "github.com/coreos/go-oidc/key/key.go",
        "github.com/coreos/go-oidc/key/manager.go",
        "github.com/coreos/go-oidc/key/repo.go",
        "github.com/coreos/go-oidc/key/rotate.go",
        "github.com/coreos/go-oidc/key/sync.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/pkg/health",
        "//vendor:github.com/coreos/pkg/timeutil",
        "//vendor:github.com/jonboulle/clockwork",
    ],
)

go_library(
    name = "github.com/coreos/go-oidc/oauth2",
    srcs = [
        "github.com/coreos/go-oidc/oauth2/error.go",
        "github.com/coreos/go-oidc/oauth2/oauth2.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/go-oidc/http"],
)

go_library(
    name = "github.com/coreos/go-oidc/oidc",
    srcs = [
        "github.com/coreos/go-oidc/oidc/client.go",
        "github.com/coreos/go-oidc/oidc/identity.go",
        "github.com/coreos/go-oidc/oidc/interface.go",
        "github.com/coreos/go-oidc/oidc/key.go",
        "github.com/coreos/go-oidc/oidc/provider.go",
        "github.com/coreos/go-oidc/oidc/transport.go",
        "github.com/coreos/go-oidc/oidc/util.go",
        "github.com/coreos/go-oidc/oidc/verification.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/http",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oauth2",
        "//vendor:github.com/coreos/pkg/timeutil",
        "//vendor:github.com/jonboulle/clockwork",
    ],
)

go_library(
    name = "github.com/coreos/go-semver/semver",
    srcs = [
        "github.com/coreos/go-semver/semver/semver.go",
        "github.com/coreos/go-semver/semver/sort.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-systemd/daemon",
    srcs = ["github.com/coreos/go-systemd/daemon/sdnotify.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-systemd/dbus",
    srcs = [
        "github.com/coreos/go-systemd/dbus/dbus.go",
        "github.com/coreos/go-systemd/dbus/methods.go",
        "github.com/coreos/go-systemd/dbus/properties.go",
        "github.com/coreos/go-systemd/dbus/set.go",
        "github.com/coreos/go-systemd/dbus/subscription.go",
        "github.com/coreos/go-systemd/dbus/subscription_set.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/godbus/dbus"],
)

go_library(
    name = "github.com/coreos/go-systemd/journal",
    srcs = ["github.com/coreos/go-systemd/journal/journal.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-systemd/unit",
    srcs = [
        "github.com/coreos/go-systemd/unit/deserialize.go",
        "github.com/coreos/go-systemd/unit/escape.go",
        "github.com/coreos/go-systemd/unit/option.go",
        "github.com/coreos/go-systemd/unit/serialize.go",
    ],
    tags = ["automanaged"],
)

cgo_genrule(
    name = "github.com/coreos/go-systemd/util_cgo",
    srcs = ["github.com/coreos/go-systemd/util/util.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/go-systemd/util",
    srcs = [],
    library = ":github.com/coreos/go-systemd/util_cgo",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/dlopen"],
)

go_library(
    name = "github.com/coreos/pkg/capnslog",
    srcs = [
        "github.com/coreos/pkg/capnslog/formatters.go",
        "github.com/coreos/pkg/capnslog/glog_formatter.go",
        "github.com/coreos/pkg/capnslog/init.go",
        "github.com/coreos/pkg/capnslog/journald_formatter.go",
        "github.com/coreos/pkg/capnslog/log_hijack.go",
        "github.com/coreos/pkg/capnslog/logmap.go",
        "github.com/coreos/pkg/capnslog/pkg_logger.go",
        "github.com/coreos/pkg/capnslog/syslog_formatter.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/go-systemd/journal"],
)

cgo_genrule(
    name = "github.com/coreos/pkg/dlopen_cgo",
    srcs = [
        "github.com/coreos/pkg/dlopen/dlopen.go",
        "github.com/coreos/pkg/dlopen/dlopen_example.go",
    ],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/pkg/dlopen",
    srcs = [],
    library = ":github.com/coreos/pkg/dlopen_cgo",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/pkg/health",
    srcs = ["github.com/coreos/pkg/health/health.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/httputil"],
)

go_library(
    name = "github.com/coreos/pkg/httputil",
    srcs = [
        "github.com/coreos/pkg/httputil/cookie.go",
        "github.com/coreos/pkg/httputil/json.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/pkg/timeutil",
    srcs = ["github.com/coreos/pkg/timeutil/backoff.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/coreos/rkt/api/v1alpha",
    srcs = ["github.com/coreos/rkt/api/v1alpha/api.pb.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)

go_library(
    name = "github.com/cpuguy83/go-md2man/md2man",
    srcs = [
        "github.com/cpuguy83/go-md2man/md2man/md2man.go",
        "github.com/cpuguy83/go-md2man/md2man/roff.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/russross/blackfriday"],
)

go_library(
    name = "github.com/davecgh/go-spew/spew",
    srcs = [
        "github.com/davecgh/go-spew/spew/bypass.go",
        "github.com/davecgh/go-spew/spew/common.go",
        "github.com/davecgh/go-spew/spew/config.go",
        "github.com/davecgh/go-spew/spew/doc.go",
        "github.com/davecgh/go-spew/spew/dump.go",
        "github.com/davecgh/go-spew/spew/format.go",
        "github.com/davecgh/go-spew/spew/spew.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/daviddengcn/go-colortext",
    srcs = [
        "github.com/daviddengcn/go-colortext/ct.go",
        "github.com/daviddengcn/go-colortext/ct_ansi.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/dgrijalva/jwt-go",
    srcs = [
        "github.com/dgrijalva/jwt-go/claims.go",
        "github.com/dgrijalva/jwt-go/doc.go",
        "github.com/dgrijalva/jwt-go/ecdsa.go",
        "github.com/dgrijalva/jwt-go/ecdsa_utils.go",
        "github.com/dgrijalva/jwt-go/errors.go",
        "github.com/dgrijalva/jwt-go/hmac.go",
        "github.com/dgrijalva/jwt-go/map_claims.go",
        "github.com/dgrijalva/jwt-go/none.go",
        "github.com/dgrijalva/jwt-go/parser.go",
        "github.com/dgrijalva/jwt-go/rsa.go",
        "github.com/dgrijalva/jwt-go/rsa_pss.go",
        "github.com/dgrijalva/jwt-go/rsa_utils.go",
        "github.com/dgrijalva/jwt-go/signing_method.go",
        "github.com/dgrijalva/jwt-go/token.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/distribution/digest",
    srcs = [
        "github.com/docker/distribution/digest/digest.go",
        "github.com/docker/distribution/digest/digester.go",
        "github.com/docker/distribution/digest/doc.go",
        "github.com/docker/distribution/digest/set.go",
        "github.com/docker/distribution/digest/verifiers.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/distribution/reference",
    srcs = [
        "github.com/docker/distribution/reference/reference.go",
        "github.com/docker/distribution/reference/regexp.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/distribution/digest"],
)

go_library(
    name = "github.com/docker/docker/pkg/jsonlog",
    srcs = [
        "github.com/docker/docker/pkg/jsonlog/jsonlog.go",
        "github.com/docker/docker/pkg/jsonlog/jsonlog_marshalling.go",
        "github.com/docker/docker/pkg/jsonlog/jsonlogbytes.go",
        "github.com/docker/docker/pkg/jsonlog/time_marshalling.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/docker/pkg/jsonmessage",
    srcs = ["github.com/docker/docker/pkg/jsonmessage/jsonmessage.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/docker/pkg/jsonlog",
        "//vendor:github.com/docker/docker/pkg/term",
        "//vendor:github.com/docker/go-units",
    ],
)

go_library(
    name = "github.com/docker/docker/pkg/longpath",
    srcs = ["github.com/docker/docker/pkg/longpath/longpath.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/docker/pkg/mount",
    srcs = [
        "github.com/docker/docker/pkg/mount/flags.go",
        "github.com/docker/docker/pkg/mount/flags_linux.go",
        "github.com/docker/docker/pkg/mount/mount.go",
        "github.com/docker/docker/pkg/mount/mounter_linux.go",
        "github.com/docker/docker/pkg/mount/mountinfo.go",
        "github.com/docker/docker/pkg/mount/mountinfo_linux.go",
        "github.com/docker/docker/pkg/mount/sharedsubtree_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/docker/pkg/stdcopy",
    srcs = ["github.com/docker/docker/pkg/stdcopy/stdcopy.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/Sirupsen/logrus"],
)

go_library(
    name = "github.com/docker/docker/pkg/symlink",
    srcs = [
        "github.com/docker/docker/pkg/symlink/fs.go",
        "github.com/docker/docker/pkg/symlink/fs_unix.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/docker/pkg/system"],
)

go_library(
    name = "github.com/docker/docker/pkg/system",
    srcs = [
        "github.com/docker/docker/pkg/system/chtimes.go",
        "github.com/docker/docker/pkg/system/chtimes_unix.go",
        "github.com/docker/docker/pkg/system/errors.go",
        "github.com/docker/docker/pkg/system/filesys.go",
        "github.com/docker/docker/pkg/system/lstat.go",
        "github.com/docker/docker/pkg/system/meminfo.go",
        "github.com/docker/docker/pkg/system/meminfo_linux.go",
        "github.com/docker/docker/pkg/system/mknod.go",
        "github.com/docker/docker/pkg/system/path_unix.go",
        "github.com/docker/docker/pkg/system/stat.go",
        "github.com/docker/docker/pkg/system/stat_linux.go",
        "github.com/docker/docker/pkg/system/syscall_unix.go",
        "github.com/docker/docker/pkg/system/umask.go",
        "github.com/docker/docker/pkg/system/utimes_linux.go",
        "github.com/docker/docker/pkg/system/xattrs_linux.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/go-units"],
)

cgo_genrule(
    name = "github.com/docker/docker/pkg/term_cgo",
    srcs = ["github.com/docker/docker/pkg/term/tc_linux_cgo.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/docker/pkg/term",
    srcs = [
        "github.com/docker/docker/pkg/term/ascii.go",
        "github.com/docker/docker/pkg/term/term.go",
    ],
    library = ":github.com/docker/docker/pkg/term_cgo",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/docker/pkg/term/windows",
    srcs = ["github.com/docker/docker/pkg/term/windows/windows.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/client",
    srcs = [
        "github.com/docker/engine-api/client/client.go",
        "github.com/docker/engine-api/client/client_unix.go",
        "github.com/docker/engine-api/client/container_attach.go",
        "github.com/docker/engine-api/client/container_commit.go",
        "github.com/docker/engine-api/client/container_copy.go",
        "github.com/docker/engine-api/client/container_create.go",
        "github.com/docker/engine-api/client/container_diff.go",
        "github.com/docker/engine-api/client/container_exec.go",
        "github.com/docker/engine-api/client/container_export.go",
        "github.com/docker/engine-api/client/container_inspect.go",
        "github.com/docker/engine-api/client/container_kill.go",
        "github.com/docker/engine-api/client/container_list.go",
        "github.com/docker/engine-api/client/container_logs.go",
        "github.com/docker/engine-api/client/container_pause.go",
        "github.com/docker/engine-api/client/container_remove.go",
        "github.com/docker/engine-api/client/container_rename.go",
        "github.com/docker/engine-api/client/container_resize.go",
        "github.com/docker/engine-api/client/container_restart.go",
        "github.com/docker/engine-api/client/container_start.go",
        "github.com/docker/engine-api/client/container_stats.go",
        "github.com/docker/engine-api/client/container_stop.go",
        "github.com/docker/engine-api/client/container_top.go",
        "github.com/docker/engine-api/client/container_unpause.go",
        "github.com/docker/engine-api/client/container_update.go",
        "github.com/docker/engine-api/client/container_wait.go",
        "github.com/docker/engine-api/client/errors.go",
        "github.com/docker/engine-api/client/events.go",
        "github.com/docker/engine-api/client/hijack.go",
        "github.com/docker/engine-api/client/image_build.go",
        "github.com/docker/engine-api/client/image_create.go",
        "github.com/docker/engine-api/client/image_history.go",
        "github.com/docker/engine-api/client/image_import.go",
        "github.com/docker/engine-api/client/image_inspect.go",
        "github.com/docker/engine-api/client/image_list.go",
        "github.com/docker/engine-api/client/image_load.go",
        "github.com/docker/engine-api/client/image_pull.go",
        "github.com/docker/engine-api/client/image_push.go",
        "github.com/docker/engine-api/client/image_remove.go",
        "github.com/docker/engine-api/client/image_save.go",
        "github.com/docker/engine-api/client/image_search.go",
        "github.com/docker/engine-api/client/image_tag.go",
        "github.com/docker/engine-api/client/info.go",
        "github.com/docker/engine-api/client/interface.go",
        "github.com/docker/engine-api/client/login.go",
        "github.com/docker/engine-api/client/network_connect.go",
        "github.com/docker/engine-api/client/network_create.go",
        "github.com/docker/engine-api/client/network_disconnect.go",
        "github.com/docker/engine-api/client/network_inspect.go",
        "github.com/docker/engine-api/client/network_list.go",
        "github.com/docker/engine-api/client/network_remove.go",
        "github.com/docker/engine-api/client/request.go",
        "github.com/docker/engine-api/client/version.go",
        "github.com/docker/engine-api/client/volume_create.go",
        "github.com/docker/engine-api/client/volume_inspect.go",
        "github.com/docker/engine-api/client/volume_list.go",
        "github.com/docker/engine-api/client/volume_remove.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/distribution/reference",
        "//vendor:github.com/docker/engine-api/client/transport",
        "//vendor:github.com/docker/engine-api/client/transport/cancellable",
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/docker/engine-api/types/container",
        "//vendor:github.com/docker/engine-api/types/filters",
        "//vendor:github.com/docker/engine-api/types/network",
        "//vendor:github.com/docker/engine-api/types/reference",
        "//vendor:github.com/docker/engine-api/types/registry",
        "//vendor:github.com/docker/engine-api/types/time",
        "//vendor:github.com/docker/go-connections/sockets",
        "//vendor:github.com/docker/go-connections/tlsconfig",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/docker/engine-api/client/transport",
    srcs = [
        "github.com/docker/engine-api/client/transport/client.go",
        "github.com/docker/engine-api/client/transport/transport.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/go-connections/sockets"],
)

go_library(
    name = "github.com/docker/engine-api/client/transport/cancellable",
    srcs = [
        "github.com/docker/engine-api/client/transport/cancellable/canceler.go",
        "github.com/docker/engine-api/client/transport/cancellable/cancellable.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/engine-api/client/transport",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/docker/engine-api/types",
    srcs = [
        "github.com/docker/engine-api/types/auth.go",
        "github.com/docker/engine-api/types/client.go",
        "github.com/docker/engine-api/types/configs.go",
        "github.com/docker/engine-api/types/seccomp.go",
        "github.com/docker/engine-api/types/stats.go",
        "github.com/docker/engine-api/types/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/engine-api/types/container",
        "//vendor:github.com/docker/engine-api/types/filters",
        "//vendor:github.com/docker/engine-api/types/network",
        "//vendor:github.com/docker/engine-api/types/registry",
        "//vendor:github.com/docker/go-connections/nat",
        "//vendor:github.com/docker/go-units",
    ],
)

go_library(
    name = "github.com/docker/engine-api/types/blkiodev",
    srcs = ["github.com/docker/engine-api/types/blkiodev/blkio.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/types/container",
    srcs = [
        "github.com/docker/engine-api/types/container/config.go",
        "github.com/docker/engine-api/types/container/host_config.go",
        "github.com/docker/engine-api/types/container/hostconfig_unix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/engine-api/types/blkiodev",
        "//vendor:github.com/docker/engine-api/types/strslice",
        "//vendor:github.com/docker/go-connections/nat",
        "//vendor:github.com/docker/go-units",
    ],
)

go_library(
    name = "github.com/docker/engine-api/types/filters",
    srcs = ["github.com/docker/engine-api/types/filters/parse.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/engine-api/types/versions"],
)

go_library(
    name = "github.com/docker/engine-api/types/network",
    srcs = ["github.com/docker/engine-api/types/network/network.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/types/reference",
    srcs = ["github.com/docker/engine-api/types/reference/image_reference.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/distribution/reference"],
)

go_library(
    name = "github.com/docker/engine-api/types/registry",
    srcs = ["github.com/docker/engine-api/types/registry/registry.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/types/strslice",
    srcs = ["github.com/docker/engine-api/types/strslice/strslice.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/types/time",
    srcs = ["github.com/docker/engine-api/types/time/timestamp.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/engine-api/types/versions",
    srcs = ["github.com/docker/engine-api/types/versions/compare.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/go-connections/nat",
    srcs = [
        "github.com/docker/go-connections/nat/nat.go",
        "github.com/docker/go-connections/nat/parse.go",
        "github.com/docker/go-connections/nat/sort.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/go-connections/sockets",
    srcs = [
        "github.com/docker/go-connections/sockets/inmem_socket.go",
        "github.com/docker/go-connections/sockets/proxy.go",
        "github.com/docker/go-connections/sockets/sockets.go",
        "github.com/docker/go-connections/sockets/sockets_unix.go",
        "github.com/docker/go-connections/sockets/tcp_socket.go",
        "github.com/docker/go-connections/sockets/unix_socket.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Sirupsen/logrus",
        "//vendor:github.com/opencontainers/runc/libcontainer/user",
        "//vendor:golang.org/x/net/proxy",
    ],
)

go_library(
    name = "github.com/docker/go-connections/tlsconfig",
    srcs = [
        "github.com/docker/go-connections/tlsconfig/config.go",
        "github.com/docker/go-connections/tlsconfig/config_client_ciphers.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/Sirupsen/logrus"],
)

go_library(
    name = "github.com/docker/go-units",
    srcs = [
        "github.com/docker/go-units/duration.go",
        "github.com/docker/go-units/size.go",
        "github.com/docker/go-units/ulimit.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/docker/spdystream",
    srcs = [
        "github.com/docker/spdystream/connection.go",
        "github.com/docker/spdystream/handlers.go",
        "github.com/docker/spdystream/priority.go",
        "github.com/docker/spdystream/stream.go",
        "github.com/docker/spdystream/utils.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/spdystream/spdy"],
)

go_library(
    name = "github.com/docker/spdystream/spdy",
    srcs = [
        "github.com/docker/spdystream/spdy/dictionary.go",
        "github.com/docker/spdystream/spdy/read.go",
        "github.com/docker/spdystream/spdy/types.go",
        "github.com/docker/spdystream/spdy/write.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/elazarl/go-bindata-assetfs",
    srcs = [
        "github.com/elazarl/go-bindata-assetfs/assetfs.go",
        "github.com/elazarl/go-bindata-assetfs/doc.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/elazarl/goproxy",
    srcs = [
        "github.com/elazarl/goproxy/actions.go",
        "github.com/elazarl/goproxy/certs.go",
        "github.com/elazarl/goproxy/chunked.go",
        "github.com/elazarl/goproxy/counterecryptor.go",
        "github.com/elazarl/goproxy/ctx.go",
        "github.com/elazarl/goproxy/dispatcher.go",
        "github.com/elazarl/goproxy/doc.go",
        "github.com/elazarl/goproxy/https.go",
        "github.com/elazarl/goproxy/proxy.go",
        "github.com/elazarl/goproxy/responses.go",
        "github.com/elazarl/goproxy/signer.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/emicklei/go-restful",
    srcs = [
        "github.com/emicklei/go-restful/compress.go",
        "github.com/emicklei/go-restful/compressor_cache.go",
        "github.com/emicklei/go-restful/compressor_pools.go",
        "github.com/emicklei/go-restful/compressors.go",
        "github.com/emicklei/go-restful/constants.go",
        "github.com/emicklei/go-restful/container.go",
        "github.com/emicklei/go-restful/cors_filter.go",
        "github.com/emicklei/go-restful/curly.go",
        "github.com/emicklei/go-restful/curly_route.go",
        "github.com/emicklei/go-restful/doc.go",
        "github.com/emicklei/go-restful/entity_accessors.go",
        "github.com/emicklei/go-restful/filter.go",
        "github.com/emicklei/go-restful/jsr311.go",
        "github.com/emicklei/go-restful/logger.go",
        "github.com/emicklei/go-restful/mime.go",
        "github.com/emicklei/go-restful/options_filter.go",
        "github.com/emicklei/go-restful/parameter.go",
        "github.com/emicklei/go-restful/path_expression.go",
        "github.com/emicklei/go-restful/request.go",
        "github.com/emicklei/go-restful/response.go",
        "github.com/emicklei/go-restful/route.go",
        "github.com/emicklei/go-restful/route_builder.go",
        "github.com/emicklei/go-restful/router.go",
        "github.com/emicklei/go-restful/service_error.go",
        "github.com/emicklei/go-restful/web_service.go",
        "github.com/emicklei/go-restful/web_service_container.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/emicklei/go-restful/log"],
)

go_library(
    name = "github.com/emicklei/go-restful/log",
    srcs = ["github.com/emicklei/go-restful/log/log.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/emicklei/go-restful/swagger",
    srcs = [
        "github.com/emicklei/go-restful/swagger/api_declaration_list.go",
        "github.com/emicklei/go-restful/swagger/config.go",
        "github.com/emicklei/go-restful/swagger/model_builder.go",
        "github.com/emicklei/go-restful/swagger/model_list.go",
        "github.com/emicklei/go-restful/swagger/model_property_ext.go",
        "github.com/emicklei/go-restful/swagger/model_property_list.go",
        "github.com/emicklei/go-restful/swagger/ordered_route_map.go",
        "github.com/emicklei/go-restful/swagger/swagger.go",
        "github.com/emicklei/go-restful/swagger/swagger_builder.go",
        "github.com/emicklei/go-restful/swagger/swagger_webservice.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/log",
    ],
)

go_library(
    name = "github.com/evanphx/json-patch",
    srcs = [
        "github.com/evanphx/json-patch/merge.go",
        "github.com/evanphx/json-patch/patch.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/exponent-io/jsonpath",
    srcs = [
        "github.com/exponent-io/jsonpath/decoder.go",
        "github.com/exponent-io/jsonpath/path.go",
        "github.com/exponent-io/jsonpath/pathaction.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/fsnotify/fsnotify",
    srcs = [
        "github.com/fsnotify/fsnotify/fsnotify.go",
        "github.com/fsnotify/fsnotify/inotify.go",
        "github.com/fsnotify/fsnotify/inotify_poller.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/sys/unix"],
)

go_library(
    name = "github.com/garyburd/redigo/internal",
    srcs = ["github.com/garyburd/redigo/internal/commandinfo.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/garyburd/redigo/redis",
    srcs = [
        "github.com/garyburd/redigo/redis/conn.go",
        "github.com/garyburd/redigo/redis/doc.go",
        "github.com/garyburd/redigo/redis/log.go",
        "github.com/garyburd/redigo/redis/pool.go",
        "github.com/garyburd/redigo/redis/pubsub.go",
        "github.com/garyburd/redigo/redis/redis.go",
        "github.com/garyburd/redigo/redis/reply.go",
        "github.com/garyburd/redigo/redis/scan.go",
        "github.com/garyburd/redigo/redis/script.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/garyburd/redigo/internal"],
)

go_library(
    name = "github.com/ghodss/yaml",
    srcs = [
        "github.com/ghodss/yaml/fields.go",
        "github.com/ghodss/yaml/yaml.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:gopkg.in/yaml.v2"],
)

go_library(
    name = "github.com/go-ini/ini",
    srcs = [
        "github.com/go-ini/ini/ini.go",
        "github.com/go-ini/ini/struct.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/go-openapi/analysis",
    srcs = ["github.com/go-openapi/analysis/analyzer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/jsonpointer",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "github.com/go-openapi/errors",
    srcs = [
        "github.com/go-openapi/errors/api.go",
        "github.com/go-openapi/errors/auth.go",
        "github.com/go-openapi/errors/headers.go",
        "github.com/go-openapi/errors/middleware.go",
        "github.com/go-openapi/errors/parsing.go",
        "github.com/go-openapi/errors/schema.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/go-openapi/jsonpointer",
    srcs = ["github.com/go-openapi/jsonpointer/pointer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/go-openapi/swag"],
)

go_library(
    name = "github.com/go-openapi/jsonreference",
    srcs = ["github.com/go-openapi/jsonreference/reference.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/PuerkitoBio/purell",
        "//vendor:github.com/go-openapi/jsonpointer",
    ],
)

go_library(
    name = "github.com/go-openapi/loads",
    srcs = ["github.com/go-openapi/loads/spec.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/analysis",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "github.com/go-openapi/runtime",
    srcs = [
        "github.com/go-openapi/runtime/bytestream.go",
        "github.com/go-openapi/runtime/client_auth_info.go",
        "github.com/go-openapi/runtime/client_operation.go",
        "github.com/go-openapi/runtime/client_request.go",
        "github.com/go-openapi/runtime/client_response.go",
        "github.com/go-openapi/runtime/constants.go",
        "github.com/go-openapi/runtime/discard.go",
        "github.com/go-openapi/runtime/headers.go",
        "github.com/go-openapi/runtime/interfaces.go",
        "github.com/go-openapi/runtime/json.go",
        "github.com/go-openapi/runtime/request.go",
        "github.com/go-openapi/runtime/statuses.go",
        "github.com/go-openapi/runtime/text.go",
        "github.com/go-openapi/runtime/values.go",
        "github.com/go-openapi/runtime/xml.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/errors",
        "//vendor:github.com/go-openapi/strfmt",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "github.com/go-openapi/spec",
    srcs = [
        "github.com/go-openapi/spec/bindata.go",
        "github.com/go-openapi/spec/contact_info.go",
        "github.com/go-openapi/spec/expander.go",
        "github.com/go-openapi/spec/external_docs.go",
        "github.com/go-openapi/spec/header.go",
        "github.com/go-openapi/spec/info.go",
        "github.com/go-openapi/spec/items.go",
        "github.com/go-openapi/spec/license.go",
        "github.com/go-openapi/spec/operation.go",
        "github.com/go-openapi/spec/parameter.go",
        "github.com/go-openapi/spec/path_item.go",
        "github.com/go-openapi/spec/paths.go",
        "github.com/go-openapi/spec/ref.go",
        "github.com/go-openapi/spec/response.go",
        "github.com/go-openapi/spec/responses.go",
        "github.com/go-openapi/spec/schema.go",
        "github.com/go-openapi/spec/security_scheme.go",
        "github.com/go-openapi/spec/spec.go",
        "github.com/go-openapi/spec/swagger.go",
        "github.com/go-openapi/spec/tag.go",
        "github.com/go-openapi/spec/xml_object.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/jsonpointer",
        "//vendor:github.com/go-openapi/jsonreference",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "github.com/go-openapi/strfmt",
    srcs = [
        "github.com/go-openapi/strfmt/date.go",
        "github.com/go-openapi/strfmt/default.go",
        "github.com/go-openapi/strfmt/doc.go",
        "github.com/go-openapi/strfmt/duration.go",
        "github.com/go-openapi/strfmt/format.go",
        "github.com/go-openapi/strfmt/time.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/asaskevich/govalidator",
        "//vendor:github.com/go-openapi/errors",
        "//vendor:github.com/mailru/easyjson/jlexer",
        "//vendor:github.com/mailru/easyjson/jwriter",
    ],
)

go_library(
    name = "github.com/go-openapi/swag",
    srcs = [
        "github.com/go-openapi/swag/convert.go",
        "github.com/go-openapi/swag/convert_types.go",
        "github.com/go-openapi/swag/json.go",
        "github.com/go-openapi/swag/loading.go",
        "github.com/go-openapi/swag/net.go",
        "github.com/go-openapi/swag/path.go",
        "github.com/go-openapi/swag/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mailru/easyjson/jlexer",
        "//vendor:github.com/mailru/easyjson/jwriter",
    ],
)

go_library(
    name = "github.com/go-openapi/validate",
    srcs = [
        "github.com/go-openapi/validate/formats.go",
        "github.com/go-openapi/validate/object_validator.go",
        "github.com/go-openapi/validate/result.go",
        "github.com/go-openapi/validate/schema.go",
        "github.com/go-openapi/validate/schema_props.go",
        "github.com/go-openapi/validate/slice_validator.go",
        "github.com/go-openapi/validate/spec.go",
        "github.com/go-openapi/validate/type.go",
        "github.com/go-openapi/validate/validator.go",
        "github.com/go-openapi/validate/values.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/analysis",
        "//vendor:github.com/go-openapi/errors",
        "//vendor:github.com/go-openapi/jsonpointer",
        "//vendor:github.com/go-openapi/loads",
        "//vendor:github.com/go-openapi/runtime",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/go-openapi/strfmt",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "github.com/godbus/dbus",
    srcs = [
        "github.com/godbus/dbus/auth.go",
        "github.com/godbus/dbus/auth_external.go",
        "github.com/godbus/dbus/auth_sha1.go",
        "github.com/godbus/dbus/call.go",
        "github.com/godbus/dbus/conn.go",
        "github.com/godbus/dbus/conn_other.go",
        "github.com/godbus/dbus/dbus.go",
        "github.com/godbus/dbus/decoder.go",
        "github.com/godbus/dbus/doc.go",
        "github.com/godbus/dbus/encoder.go",
        "github.com/godbus/dbus/export.go",
        "github.com/godbus/dbus/homedir.go",
        "github.com/godbus/dbus/homedir_dynamic.go",
        "github.com/godbus/dbus/message.go",
        "github.com/godbus/dbus/object.go",
        "github.com/godbus/dbus/sig.go",
        "github.com/godbus/dbus/transport_generic.go",
        "github.com/godbus/dbus/transport_unix.go",
        "github.com/godbus/dbus/transport_unixcred_linux.go",
        "github.com/godbus/dbus/variant.go",
        "github.com/godbus/dbus/variant_lexer.go",
        "github.com/godbus/dbus/variant_parser.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/gogo/protobuf/gogoproto",
    srcs = [
        "github.com/gogo/protobuf/gogoproto/doc.go",
        "github.com/gogo/protobuf/gogoproto/gogo.pb.go",
        "github.com/gogo/protobuf/gogoproto/helper.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/compare",
    srcs = [
        "github.com/gogo/protobuf/plugin/compare/compare.go",
        "github.com/gogo/protobuf/plugin/compare/comparetest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/vanity",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/defaultcheck",
    srcs = ["github.com/gogo/protobuf/plugin/defaultcheck/defaultcheck.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/description",
    srcs = [
        "github.com/gogo/protobuf/plugin/description/description.go",
        "github.com/gogo/protobuf/plugin/description/descriptiontest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/embedcheck",
    srcs = ["github.com/gogo/protobuf/plugin/embedcheck/embedcheck.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/enumstringer",
    srcs = ["github.com/gogo/protobuf/plugin/enumstringer/enumstringer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/equal",
    srcs = [
        "github.com/gogo/protobuf/plugin/equal/equal.go",
        "github.com/gogo/protobuf/plugin/equal/equaltest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/vanity",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/face",
    srcs = [
        "github.com/gogo/protobuf/plugin/face/face.go",
        "github.com/gogo/protobuf/plugin/face/facetest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/gostring",
    srcs = [
        "github.com/gogo/protobuf/plugin/gostring/gostring.go",
        "github.com/gogo/protobuf/plugin/gostring/gostringtest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/marshalto",
    srcs = ["github.com/gogo/protobuf/plugin/marshalto/marshalto.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/vanity",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/oneofcheck",
    srcs = ["github.com/gogo/protobuf/plugin/oneofcheck/oneofcheck.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/populate",
    srcs = ["github.com/gogo/protobuf/plugin/populate/populate.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/vanity",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/size",
    srcs = [
        "github.com/gogo/protobuf/plugin/size/size.go",
        "github.com/gogo/protobuf/plugin/size/sizetest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/vanity",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/stringer",
    srcs = [
        "github.com/gogo/protobuf/plugin/stringer/stringer.go",
        "github.com/gogo/protobuf/plugin/stringer/stringertest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/testgen",
    srcs = ["github.com/gogo/protobuf/plugin/testgen/testgen.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/union",
    srcs = [
        "github.com/gogo/protobuf/plugin/union/union.go",
        "github.com/gogo/protobuf/plugin/union/uniontest.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/plugin/unmarshal",
    srcs = ["github.com/gogo/protobuf/plugin/unmarshal/unmarshal.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/proto",
    srcs = [
        "github.com/gogo/protobuf/proto/clone.go",
        "github.com/gogo/protobuf/proto/decode.go",
        "github.com/gogo/protobuf/proto/decode_gogo.go",
        "github.com/gogo/protobuf/proto/encode.go",
        "github.com/gogo/protobuf/proto/encode_gogo.go",
        "github.com/gogo/protobuf/proto/equal.go",
        "github.com/gogo/protobuf/proto/extensions.go",
        "github.com/gogo/protobuf/proto/extensions_gogo.go",
        "github.com/gogo/protobuf/proto/lib.go",
        "github.com/gogo/protobuf/proto/lib_gogo.go",
        "github.com/gogo/protobuf/proto/message_set.go",
        "github.com/gogo/protobuf/proto/pointer_unsafe.go",
        "github.com/gogo/protobuf/proto/pointer_unsafe_gogo.go",
        "github.com/gogo/protobuf/proto/properties.go",
        "github.com/gogo/protobuf/proto/properties_gogo.go",
        "github.com/gogo/protobuf/proto/skip_gogo.go",
        "github.com/gogo/protobuf/proto/text.go",
        "github.com/gogo/protobuf/proto/text_gogo.go",
        "github.com/gogo/protobuf/proto/text_parser.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
    srcs = [
        "github.com/gogo/protobuf/protoc-gen-gogo/descriptor/descriptor.pb.go",
        "github.com/gogo/protobuf/protoc-gen-gogo/descriptor/gostring.go",
        "github.com/gogo/protobuf/protoc-gen-gogo/descriptor/helper.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/gogo/protobuf/proto"],
)

go_library(
    name = "github.com/gogo/protobuf/protoc-gen-gogo/generator",
    srcs = [
        "github.com/gogo/protobuf/protoc-gen-gogo/generator/generator.go",
        "github.com/gogo/protobuf/protoc-gen-gogo/generator/helper.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/plugin",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/protoc-gen-gogo/grpc",
    srcs = ["github.com/gogo/protobuf/protoc-gen-gogo/grpc/grpc.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/protoc-gen-gogo/plugin",
    srcs = ["github.com/gogo/protobuf/protoc-gen-gogo/plugin/plugin.pb.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/sortkeys",
    srcs = ["github.com/gogo/protobuf/sortkeys/sortkeys.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/gogo/protobuf/vanity",
    srcs = [
        "github.com/gogo/protobuf/vanity/enum.go",
        "github.com/gogo/protobuf/vanity/field.go",
        "github.com/gogo/protobuf/vanity/file.go",
        "github.com/gogo/protobuf/vanity/foreach.go",
        "github.com/gogo/protobuf/vanity/msg.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/gogoproto",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/descriptor",
    ],
)

go_library(
    name = "github.com/gogo/protobuf/vanity/command",
    srcs = ["github.com/gogo/protobuf/vanity/command/command.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/plugin/compare",
        "//vendor:github.com/gogo/protobuf/plugin/defaultcheck",
        "//vendor:github.com/gogo/protobuf/plugin/description",
        "//vendor:github.com/gogo/protobuf/plugin/embedcheck",
        "//vendor:github.com/gogo/protobuf/plugin/enumstringer",
        "//vendor:github.com/gogo/protobuf/plugin/equal",
        "//vendor:github.com/gogo/protobuf/plugin/face",
        "//vendor:github.com/gogo/protobuf/plugin/gostring",
        "//vendor:github.com/gogo/protobuf/plugin/marshalto",
        "//vendor:github.com/gogo/protobuf/plugin/oneofcheck",
        "//vendor:github.com/gogo/protobuf/plugin/populate",
        "//vendor:github.com/gogo/protobuf/plugin/size",
        "//vendor:github.com/gogo/protobuf/plugin/stringer",
        "//vendor:github.com/gogo/protobuf/plugin/testgen",
        "//vendor:github.com/gogo/protobuf/plugin/union",
        "//vendor:github.com/gogo/protobuf/plugin/unmarshal",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/generator",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/grpc",
        "//vendor:github.com/gogo/protobuf/protoc-gen-gogo/plugin",
    ],
)

go_library(
    name = "github.com/golang/glog",
    srcs = [
        "github.com/golang/glog/glog.go",
        "github.com/golang/glog/glog_file.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/golang/groupcache/lru",
    srcs = ["github.com/golang/groupcache/lru/lru.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/golang/mock/gomock",
    srcs = [
        "github.com/golang/mock/gomock/call.go",
        "github.com/golang/mock/gomock/callset.go",
        "github.com/golang/mock/gomock/controller.go",
        "github.com/golang/mock/gomock/matchers.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/golang/protobuf/jsonpb",
    srcs = ["github.com/golang/protobuf/jsonpb/jsonpb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/golang/protobuf/proto",
    srcs = [
        "github.com/golang/protobuf/proto/clone.go",
        "github.com/golang/protobuf/proto/decode.go",
        "github.com/golang/protobuf/proto/encode.go",
        "github.com/golang/protobuf/proto/equal.go",
        "github.com/golang/protobuf/proto/extensions.go",
        "github.com/golang/protobuf/proto/lib.go",
        "github.com/golang/protobuf/proto/message_set.go",
        "github.com/golang/protobuf/proto/pointer_unsafe.go",
        "github.com/golang/protobuf/proto/properties.go",
        "github.com/golang/protobuf/proto/text.go",
        "github.com/golang/protobuf/proto/text_parser.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/btree",
    srcs = ["github.com/google/btree/btree.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/api",
    srcs = [
        "github.com/google/cadvisor/api/handler.go",
        "github.com/google/cadvisor/api/versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/events",
        "//vendor:github.com/google/cadvisor/http/mux",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/google/cadvisor/manager",
    ],
)

go_library(
    name = "github.com/google/cadvisor/cache/memory",
    srcs = ["github.com/google/cadvisor/cache/memory/memory.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/storage",
        "//vendor:github.com/google/cadvisor/utils",
    ],
)

go_library(
    name = "github.com/google/cadvisor/client/v2",
    srcs = ["github.com/google/cadvisor/client/v2/client.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
    ],
)

go_library(
    name = "github.com/google/cadvisor/collector",
    srcs = [
        "github.com/google/cadvisor/collector/collector_manager.go",
        "github.com/google/cadvisor/collector/config.go",
        "github.com/google/cadvisor/collector/fakes.go",
        "github.com/google/cadvisor/collector/generic_collector.go",
        "github.com/google/cadvisor/collector/prometheus_collector.go",
        "github.com/google/cadvisor/collector/types.go",
        "github.com/google/cadvisor/collector/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/prometheus/client_model/go",
        "//vendor:github.com/prometheus/common/expfmt",
        "//vendor:github.com/prometheus/common/model",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container",
    srcs = [
        "github.com/google/cadvisor/container/container.go",
        "github.com/google/cadvisor/container/factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/manager/watcher",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/common",
    srcs = [
        "github.com/google/cadvisor/container/common/container_hints.go",
        "github.com/google/cadvisor/container/common/fsHandler.go",
        "github.com/google/cadvisor/container/common/helpers.go",
        "github.com/google/cadvisor/container/common/inotify_watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/utils",
        "//vendor:golang.org/x/exp/inotify",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/docker",
    srcs = [
        "github.com/google/cadvisor/container/docker/client.go",
        "github.com/google/cadvisor/container/docker/docker.go",
        "github.com/google/cadvisor/container/docker/factory.go",
        "github.com/google/cadvisor/container/docker/handler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/docker/engine-api/client",
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/docker/engine-api/types/container",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/container/common",
        "//vendor:github.com/google/cadvisor/container/libcontainer",
        "//vendor:github.com/google/cadvisor/devicemapper",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/machine",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:github.com/google/cadvisor/utils/docker",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/libcontainer",
    srcs = ["github.com/google/cadvisor/container/libcontainer/helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/opencontainers/runc/libcontainer",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/raw",
    srcs = [
        "github.com/google/cadvisor/container/raw/factory.go",
        "github.com/google/cadvisor/container/raw/handler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/container/common",
        "//vendor:github.com/google/cadvisor/container/libcontainer",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/machine",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/rkt",
    srcs = [
        "github.com/google/cadvisor/container/rkt/client.go",
        "github.com/google/cadvisor/container/rkt/factory.go",
        "github.com/google/cadvisor/container/rkt/handler.go",
        "github.com/google/cadvisor/container/rkt/helpers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/coreos/rkt/api/v1alpha",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/container/common",
        "//vendor:github.com/google/cadvisor/container/libcontainer",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)

go_library(
    name = "github.com/google/cadvisor/container/systemd",
    srcs = ["github.com/google/cadvisor/container/systemd/factory.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/manager/watcher",
    ],
)

go_library(
    name = "github.com/google/cadvisor/devicemapper",
    srcs = [
        "github.com/google/cadvisor/devicemapper/dmsetup_client.go",
        "github.com/google/cadvisor/devicemapper/doc.go",
        "github.com/google/cadvisor/devicemapper/thin_ls_client.go",
        "github.com/google/cadvisor/devicemapper/thin_pool_watcher.go",
        "github.com/google/cadvisor/devicemapper/util.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_library(
    name = "github.com/google/cadvisor/events",
    srcs = ["github.com/google/cadvisor/events/handler.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/utils",
    ],
)

go_library(
    name = "github.com/google/cadvisor/fs",
    srcs = [
        "github.com/google/cadvisor/fs/fs.go",
        "github.com/google/cadvisor/fs/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/docker/pkg/mount",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/devicemapper",
        "//vendor:github.com/google/cadvisor/utils/docker",
        "//vendor:github.com/mistifyio/go-zfs",
    ],
)

go_library(
    name = "github.com/google/cadvisor/healthz",
    srcs = ["github.com/google/cadvisor/healthz/healthz.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/google/cadvisor/http/mux"],
)

go_library(
    name = "github.com/google/cadvisor/http",
    srcs = ["github.com/google/cadvisor/http/handlers.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/abbot/go-http-auth",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/api",
        "//vendor:github.com/google/cadvisor/healthz",
        "//vendor:github.com/google/cadvisor/http/mux",
        "//vendor:github.com/google/cadvisor/manager",
        "//vendor:github.com/google/cadvisor/metrics",
        "//vendor:github.com/google/cadvisor/pages",
        "//vendor:github.com/google/cadvisor/pages/static",
        "//vendor:github.com/google/cadvisor/validate",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/google/cadvisor/http/mux",
    srcs = ["github.com/google/cadvisor/http/mux/mux.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/info/v1",
    srcs = [
        "github.com/google/cadvisor/info/v1/container.go",
        "github.com/google/cadvisor/info/v1/docker.go",
        "github.com/google/cadvisor/info/v1/machine.go",
        "github.com/google/cadvisor/info/v1/metric.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/info/v2",
    srcs = [
        "github.com/google/cadvisor/info/v2/container.go",
        "github.com/google/cadvisor/info/v2/conversion.go",
        "github.com/google/cadvisor/info/v2/machine.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
    ],
)

go_library(
    name = "github.com/google/cadvisor/machine",
    srcs = [
        "github.com/google/cadvisor/machine/info.go",
        "github.com/google/cadvisor/machine/machine.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/utils",
        "//vendor:github.com/google/cadvisor/utils/cloudinfo",
        "//vendor:github.com/google/cadvisor/utils/sysfs",
        "//vendor:github.com/google/cadvisor/utils/sysinfo",
    ],
)

go_library(
    name = "github.com/google/cadvisor/manager",
    srcs = [
        "github.com/google/cadvisor/manager/container.go",
        "github.com/google/cadvisor/manager/manager.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/go-units",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/cache/memory",
        "//vendor:github.com/google/cadvisor/collector",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/container/docker",
        "//vendor:github.com/google/cadvisor/container/raw",
        "//vendor:github.com/google/cadvisor/container/rkt",
        "//vendor:github.com/google/cadvisor/container/systemd",
        "//vendor:github.com/google/cadvisor/events",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/google/cadvisor/machine",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:github.com/google/cadvisor/manager/watcher/raw",
        "//vendor:github.com/google/cadvisor/manager/watcher/rkt",
        "//vendor:github.com/google/cadvisor/summary",
        "//vendor:github.com/google/cadvisor/utils/cpuload",
        "//vendor:github.com/google/cadvisor/utils/oomparser",
        "//vendor:github.com/google/cadvisor/utils/sysfs",
        "//vendor:github.com/google/cadvisor/version",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
    ],
)

go_library(
    name = "github.com/google/cadvisor/manager/watcher",
    srcs = ["github.com/google/cadvisor/manager/watcher/watcher.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/manager/watcher/raw",
    srcs = ["github.com/google/cadvisor/manager/watcher/raw/raw.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container/common",
        "//vendor:github.com/google/cadvisor/container/libcontainer",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:golang.org/x/exp/inotify",
    ],
)

go_library(
    name = "github.com/google/cadvisor/manager/watcher/rkt",
    srcs = ["github.com/google/cadvisor/manager/watcher/rkt/rkt.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/rkt/api/v1alpha",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container/rkt",
        "//vendor:github.com/google/cadvisor/manager/watcher",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/google/cadvisor/metrics",
    srcs = ["github.com/google/cadvisor/metrics/prometheus.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_library(
    name = "github.com/google/cadvisor/pages",
    srcs = [
        "github.com/google/cadvisor/pages/containers.go",
        "github.com/google/cadvisor/pages/docker.go",
        "github.com/google/cadvisor/pages/pages.go",
        "github.com/google/cadvisor/pages/templates.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/abbot/go-http-auth",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/container/docker",
        "//vendor:github.com/google/cadvisor/http/mux",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/manager",
    ],
)

go_library(
    name = "github.com/google/cadvisor/pages/static",
    srcs = [
        "github.com/google/cadvisor/pages/static/assets.go",
        "github.com/google/cadvisor/pages/static/static.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/storage",
    srcs = [
        "github.com/google/cadvisor/storage/common_flags.go",
        "github.com/google/cadvisor/storage/storage.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/google/cadvisor/info/v1"],
)

go_library(
    name = "github.com/google/cadvisor/summary",
    srcs = [
        "github.com/google/cadvisor/summary/buffer.go",
        "github.com/google/cadvisor/summary/percentiles.go",
        "github.com/google/cadvisor/summary/summary.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
    ],
)

go_library(
    name = "github.com/google/cadvisor/utils",
    srcs = [
        "github.com/google/cadvisor/utils/path.go",
        "github.com/google/cadvisor/utils/timed_store.go",
        "github.com/google/cadvisor/utils/utils.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/utils/cloudinfo",
    srcs = [
        "github.com/google/cadvisor/utils/cloudinfo/aws.go",
        "github.com/google/cadvisor/utils/cloudinfo/azure.go",
        "github.com/google/cadvisor/utils/cloudinfo/cloudinfo.go",
        "github.com/google/cadvisor/utils/cloudinfo/gce.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/ec2metadata",
        "//vendor:github.com/aws/aws-sdk-go/aws/session",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
    ],
)

go_library(
    name = "github.com/google/cadvisor/utils/cpuload",
    srcs = ["github.com/google/cadvisor/utils/cpuload/cpuload.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/utils/cpuload/netlink",
    ],
)

cgo_genrule(
    name = "github.com/google/cadvisor/utils/cpuload/netlink_cgo",
    srcs = ["github.com/google/cadvisor/utils/cpuload/netlink/defs.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/utils/cpuload/netlink",
    srcs = [
        "github.com/google/cadvisor/utils/cpuload/netlink/conn.go",
        "github.com/google/cadvisor/utils/cpuload/netlink/netlink.go",
        "github.com/google/cadvisor/utils/cpuload/netlink/reader.go",
    ],
    library = ":github.com/google/cadvisor/utils/cpuload/netlink_cgo",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
    ],
)

go_library(
    name = "github.com/google/cadvisor/utils/docker",
    srcs = ["github.com/google/cadvisor/utils/docker/docker.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/engine-api/types"],
)

go_library(
    name = "github.com/google/cadvisor/utils/oomparser",
    srcs = ["github.com/google/cadvisor/utils/oomparser/oomparser.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/utils",
        "//vendor:github.com/google/cadvisor/utils/tail",
    ],
)

go_library(
    name = "github.com/google/cadvisor/utils/sysfs",
    srcs = ["github.com/google/cadvisor/utils/sysfs/sysfs.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/cadvisor/utils/sysinfo",
    srcs = ["github.com/google/cadvisor/utils/sysinfo/sysinfo.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/utils/sysfs",
    ],
)

go_library(
    name = "github.com/google/cadvisor/utils/tail",
    srcs = ["github.com/google/cadvisor/utils/tail/tail.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/exp/inotify",
    ],
)

go_library(
    name = "github.com/google/cadvisor/validate",
    srcs = ["github.com/google/cadvisor/validate/validate.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/cadvisor/container/docker",
        "//vendor:github.com/google/cadvisor/manager",
        "//vendor:github.com/google/cadvisor/utils",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
    ],
)

go_library(
    name = "github.com/google/cadvisor/version",
    srcs = ["github.com/google/cadvisor/version/version.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/certificate-transparency/go",
    srcs = [
        "github.com/google/certificate-transparency/go/serialization.go",
        "github.com/google/certificate-transparency/go/signatures.go",
        "github.com/google/certificate-transparency/go/types.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/google/certificate-transparency/go/x509"],
)

go_library(
    name = "github.com/google/certificate-transparency/go/asn1",
    srcs = [
        "github.com/google/certificate-transparency/go/asn1/asn1.go",
        "github.com/google/certificate-transparency/go/asn1/common.go",
        "github.com/google/certificate-transparency/go/asn1/marshal.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/google/certificate-transparency/go/client",
    srcs = ["github.com/google/certificate-transparency/go/client/logclient.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/certificate-transparency/go",
        "//vendor:github.com/mreiferson/go-httpclient",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/google/certificate-transparency/go/x509",
    srcs = [
        "github.com/google/certificate-transparency/go/x509/cert_pool.go",
        "github.com/google/certificate-transparency/go/x509/pem_decrypt.go",
        "github.com/google/certificate-transparency/go/x509/pkcs1.go",
        "github.com/google/certificate-transparency/go/x509/pkcs8.go",
        "github.com/google/certificate-transparency/go/x509/root.go",
        "github.com/google/certificate-transparency/go/x509/root_unix.go",
        "github.com/google/certificate-transparency/go/x509/sec1.go",
        "github.com/google/certificate-transparency/go/x509/verify.go",
        "github.com/google/certificate-transparency/go/x509/x509.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/certificate-transparency/go/asn1",
        "//vendor:github.com/google/certificate-transparency/go/x509/pkix",
    ],
)

go_library(
    name = "github.com/google/certificate-transparency/go/x509/pkix",
    srcs = ["github.com/google/certificate-transparency/go/x509/pkix/pkix.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/google/certificate-transparency/go/asn1"],
)

go_library(
    name = "github.com/google/gofuzz",
    srcs = [
        "github.com/google/gofuzz/doc.go",
        "github.com/google/gofuzz/fuzz.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/gorilla/context",
    srcs = [
        "github.com/gorilla/context/context.go",
        "github.com/gorilla/context/doc.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/gorilla/mux",
    srcs = [
        "github.com/gorilla/mux/doc.go",
        "github.com/gorilla/mux/mux.go",
        "github.com/gorilla/mux/regexp.go",
        "github.com/gorilla/mux/route.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/gorilla/context"],
)

go_library(
    name = "github.com/grpc-ecosystem/grpc-gateway/runtime",
    srcs = [
        "github.com/grpc-ecosystem/grpc-gateway/runtime/context.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/convert.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/doc.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/errors.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/handler.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/marshal_json.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/marshal_jsonpb.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/marshaler.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/marshaler_registry.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/mux.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/pattern.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/proto2_convert.go",
        "github.com/grpc-ecosystem/grpc-gateway/runtime/query.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/jsonpb",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/grpc-ecosystem/grpc-gateway/runtime/internal",
        "//vendor:github.com/grpc-ecosystem/grpc-gateway/utilities",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:google.golang.org/grpc/grpclog",
        "//vendor:google.golang.org/grpc/metadata",
    ],
)

go_library(
    name = "github.com/grpc-ecosystem/grpc-gateway/runtime/internal",
    srcs = ["github.com/grpc-ecosystem/grpc-gateway/runtime/internal/stream_chunk.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/grpc-ecosystem/grpc-gateway/utilities",
    srcs = [
        "github.com/grpc-ecosystem/grpc-gateway/utilities/doc.go",
        "github.com/grpc-ecosystem/grpc-gateway/utilities/pattern.go",
        "github.com/grpc-ecosystem/grpc-gateway/utilities/trie.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/hashicorp/golang-lru",
    srcs = [
        "github.com/hashicorp/golang-lru/2q.go",
        "github.com/hashicorp/golang-lru/arc.go",
        "github.com/hashicorp/golang-lru/lru.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/golang-lru/simplelru"],
)

go_library(
    name = "github.com/hashicorp/golang-lru/simplelru",
    srcs = ["github.com/hashicorp/golang-lru/simplelru/lru.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/hashicorp/hcl",
    srcs = [
        "github.com/hashicorp/hcl/decoder.go",
        "github.com/hashicorp/hcl/hcl.go",
        "github.com/hashicorp/hcl/lex.go",
        "github.com/hashicorp/hcl/parse.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/hashicorp/hcl/hcl/ast",
        "//vendor:github.com/hashicorp/hcl/hcl/parser",
        "//vendor:github.com/hashicorp/hcl/hcl/token",
        "//vendor:github.com/hashicorp/hcl/json/parser",
    ],
)

go_library(
    name = "github.com/hashicorp/hcl/hcl/ast",
    srcs = [
        "github.com/hashicorp/hcl/hcl/ast/ast.go",
        "github.com/hashicorp/hcl/hcl/ast/walk.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/hcl/hcl/token"],
)

go_library(
    name = "github.com/hashicorp/hcl/hcl/parser",
    srcs = [
        "github.com/hashicorp/hcl/hcl/parser/error.go",
        "github.com/hashicorp/hcl/hcl/parser/parser.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/hashicorp/hcl/hcl/ast",
        "//vendor:github.com/hashicorp/hcl/hcl/scanner",
        "//vendor:github.com/hashicorp/hcl/hcl/token",
    ],
)

go_library(
    name = "github.com/hashicorp/hcl/hcl/scanner",
    srcs = ["github.com/hashicorp/hcl/hcl/scanner/scanner.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/hcl/hcl/token"],
)

go_library(
    name = "github.com/hashicorp/hcl/hcl/strconv",
    srcs = ["github.com/hashicorp/hcl/hcl/strconv/quote.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/hashicorp/hcl/hcl/token",
    srcs = [
        "github.com/hashicorp/hcl/hcl/token/position.go",
        "github.com/hashicorp/hcl/hcl/token/token.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/hcl/hcl/strconv"],
)

go_library(
    name = "github.com/hashicorp/hcl/json/parser",
    srcs = [
        "github.com/hashicorp/hcl/json/parser/flatten.go",
        "github.com/hashicorp/hcl/json/parser/parser.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/hashicorp/hcl/hcl/ast",
        "//vendor:github.com/hashicorp/hcl/json/scanner",
        "//vendor:github.com/hashicorp/hcl/json/token",
    ],
)

go_library(
    name = "github.com/hashicorp/hcl/json/scanner",
    srcs = ["github.com/hashicorp/hcl/json/scanner/scanner.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/hcl/json/token"],
)

go_library(
    name = "github.com/hashicorp/hcl/json/token",
    srcs = [
        "github.com/hashicorp/hcl/json/token/position.go",
        "github.com/hashicorp/hcl/json/token/token.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/hashicorp/hcl/hcl/token"],
)

go_library(
    name = "github.com/hawkular/hawkular-client-go/metrics",
    srcs = [
        "github.com/hawkular/hawkular-client-go/metrics/client.go",
        "github.com/hawkular/hawkular-client-go/metrics/helpers.go",
        "github.com/hawkular/hawkular-client-go/metrics/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/heketi/heketi/client/api/go-client",
    srcs = [
        "github.com/heketi/heketi/client/api/go-client/backup.go",
        "github.com/heketi/heketi/client/api/go-client/client.go",
        "github.com/heketi/heketi/client/api/go-client/cluster.go",
        "github.com/heketi/heketi/client/api/go-client/device.go",
        "github.com/heketi/heketi/client/api/go-client/node.go",
        "github.com/heketi/heketi/client/api/go-client/topology.go",
        "github.com/heketi/heketi/client/api/go-client/volume.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/dgrijalva/jwt-go",
        "//vendor:github.com/heketi/heketi/pkg/glusterfs/api",
        "//vendor:github.com/heketi/heketi/pkg/utils",
    ],
)

go_library(
    name = "github.com/heketi/heketi/pkg/glusterfs/api",
    srcs = ["github.com/heketi/heketi/pkg/glusterfs/api/types.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/heketi/heketi/pkg/utils",
    srcs = [
        "github.com/heketi/heketi/pkg/utils/bodystring.go",
        "github.com/heketi/heketi/pkg/utils/jsonutils.go",
        "github.com/heketi/heketi/pkg/utils/log.go",
        "github.com/heketi/heketi/pkg/utils/sortedstrings.go",
        "github.com/heketi/heketi/pkg/utils/statusgroup.go",
        "github.com/heketi/heketi/pkg/utils/stringset.go",
        "github.com/heketi/heketi/pkg/utils/stringstack.go",
        "github.com/heketi/heketi/pkg/utils/uuid.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/lpabon/godbc"],
)

go_library(
    name = "github.com/howeyc/gopass",
    srcs = [
        "github.com/howeyc/gopass/pass.go",
        "github.com/howeyc/gopass/terminal.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/crypto/ssh/terminal"],
)

go_library(
    name = "github.com/imdario/mergo",
    srcs = [
        "github.com/imdario/mergo/doc.go",
        "github.com/imdario/mergo/map.go",
        "github.com/imdario/mergo/merge.go",
        "github.com/imdario/mergo/mergo.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/inconshreveable/mousetrap",
    srcs = ["github.com/inconshreveable/mousetrap/trap_others.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/influxdata/influxdb/client",
    srcs = ["github.com/influxdata/influxdb/client/influxdb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/influxdata/influxdb/models"],
)

go_library(
    name = "github.com/influxdata/influxdb/models",
    srcs = [
        "github.com/influxdata/influxdb/models/consistency.go",
        "github.com/influxdata/influxdb/models/points.go",
        "github.com/influxdata/influxdb/models/rows.go",
        "github.com/influxdata/influxdb/models/time.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/influxdata/influxdb/pkg/escape"],
)

go_library(
    name = "github.com/influxdata/influxdb/pkg/escape",
    srcs = [
        "github.com/influxdata/influxdb/pkg/escape/bytes.go",
        "github.com/influxdata/influxdb/pkg/escape/strings.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/jmespath/go-jmespath",
    srcs = [
        "github.com/jmespath/go-jmespath/api.go",
        "github.com/jmespath/go-jmespath/astnodetype_string.go",
        "github.com/jmespath/go-jmespath/functions.go",
        "github.com/jmespath/go-jmespath/interpreter.go",
        "github.com/jmespath/go-jmespath/lexer.go",
        "github.com/jmespath/go-jmespath/parser.go",
        "github.com/jmespath/go-jmespath/toktype_string.go",
        "github.com/jmespath/go-jmespath/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/jonboulle/clockwork",
    srcs = ["github.com/jonboulle/clockwork/clockwork.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/jteeuwen/go-bindata",
    srcs = [
        "github.com/jteeuwen/go-bindata/asset.go",
        "github.com/jteeuwen/go-bindata/bytewriter.go",
        "github.com/jteeuwen/go-bindata/config.go",
        "github.com/jteeuwen/go-bindata/convert.go",
        "github.com/jteeuwen/go-bindata/debug.go",
        "github.com/jteeuwen/go-bindata/doc.go",
        "github.com/jteeuwen/go-bindata/release.go",
        "github.com/jteeuwen/go-bindata/restore.go",
        "github.com/jteeuwen/go-bindata/stringwriter.go",
        "github.com/jteeuwen/go-bindata/toc.go",
    ],
    tags = ["automanaged"],
)

go_binary(
    name = "github.com/jteeuwen/go-bindata/go-bindata_bin",
    library = ":github.com/jteeuwen/go-bindata/go-bindata",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/jteeuwen/go-bindata/go-bindata",
    srcs = [
        "github.com/jteeuwen/go-bindata/go-bindata/AppendSliceValue.go",
        "github.com/jteeuwen/go-bindata/go-bindata/main.go",
        "github.com/jteeuwen/go-bindata/go-bindata/version.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/jteeuwen/go-bindata"],
)

go_library(
    name = "github.com/juju/ratelimit",
    srcs = [
        "github.com/juju/ratelimit/ratelimit.go",
        "github.com/juju/ratelimit/reader.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/kardianos/osext",
    srcs = [
        "github.com/kardianos/osext/osext.go",
        "github.com/kardianos/osext/osext_procfs.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/kr/fs",
    srcs = [
        "github.com/kr/fs/filesystem.go",
        "github.com/kr/fs/walk.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/kr/pty",
    srcs = [
        "github.com/kr/pty/doc.go",
        "github.com/kr/pty/ioctl.go",
        "github.com/kr/pty/pty_linux.go",
        "github.com/kr/pty/run.go",
        "github.com/kr/pty/util.go",
        "github.com/kr/pty/ztypes_amd64.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/lpabon/godbc",
    srcs = ["github.com/lpabon/godbc/godbc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/magiconair/properties",
    srcs = [
        "github.com/magiconair/properties/decode.go",
        "github.com/magiconair/properties/doc.go",
        "github.com/magiconair/properties/integrate.go",
        "github.com/magiconair/properties/lex.go",
        "github.com/magiconair/properties/load.go",
        "github.com/magiconair/properties/parser.go",
        "github.com/magiconair/properties/properties.go",
        "github.com/magiconair/properties/rangecheck.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mailru/easyjson/buffer",
    srcs = ["github.com/mailru/easyjson/buffer/pool.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mailru/easyjson/jlexer",
    srcs = [
        "github.com/mailru/easyjson/jlexer/error.go",
        "github.com/mailru/easyjson/jlexer/lexer.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mailru/easyjson/jwriter",
    srcs = ["github.com/mailru/easyjson/jwriter/writer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/mailru/easyjson/buffer"],
)

go_library(
    name = "github.com/matttproud/golang_protobuf_extensions/pbutil",
    srcs = [
        "github.com/matttproud/golang_protobuf_extensions/pbutil/decode.go",
        "github.com/matttproud/golang_protobuf_extensions/pbutil/doc.go",
        "github.com/matttproud/golang_protobuf_extensions/pbutil/encode.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/mesos/mesos-go/detector",
    srcs = [
        "github.com/mesos/mesos-go/detector/doc.go",
        "github.com/mesos/mesos-go/detector/factory.go",
        "github.com/mesos/mesos-go/detector/interface.go",
        "github.com/mesos/mesos-go/detector/standalone.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mesos/mesos-go/mesosproto",
        "//vendor:github.com/mesos/mesos-go/mesosutil",
        "//vendor:github.com/mesos/mesos-go/upid",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/mesos/mesos-go/detector/zoo",
    srcs = [
        "github.com/mesos/mesos-go/detector/zoo/client2.go",
        "github.com/mesos/mesos-go/detector/zoo/detect.go",
        "github.com/mesos/mesos-go/detector/zoo/doc.go",
        "github.com/mesos/mesos-go/detector/zoo/plugin.go",
        "github.com/mesos/mesos-go/detector/zoo/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mesos/mesos-go/detector",
        "//vendor:github.com/mesos/mesos-go/mesosproto",
        "//vendor:github.com/samuel/go-zookeeper/zk",
    ],
)

go_library(
    name = "github.com/mesos/mesos-go/mesosproto",
    srcs = [
        "github.com/mesos/mesos-go/mesosproto/authentication.pb.go",
        "github.com/mesos/mesos-go/mesosproto/internal.pb.go",
        "github.com/mesos/mesos-go/mesosproto/mesos.pb.go",
        "github.com/mesos/mesos-go/mesosproto/messages.pb.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/gogo/protobuf/proto"],
)

go_library(
    name = "github.com/mesos/mesos-go/mesosutil",
    srcs = [
        "github.com/mesos/mesos-go/mesosutil/constants.go",
        "github.com/mesos/mesos-go/mesosutil/mesosprotoutil.go",
        "github.com/mesos/mesos-go/mesosutil/node.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mesos/mesos-go/mesosproto",
    ],
)

go_library(
    name = "github.com/mesos/mesos-go/upid",
    srcs = [
        "github.com/mesos/mesos-go/upid/doc.go",
        "github.com/mesos/mesos-go/upid/upid.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/miekg/coredns/middleware/etcd/msg",
    srcs = [
        "github.com/miekg/coredns/middleware/etcd/msg/path.go",
        "github.com/miekg/coredns/middleware/etcd/msg/service.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/miekg/dns"],
)

go_library(
    name = "github.com/miekg/dns",
    srcs = [
        "github.com/miekg/dns/client.go",
        "github.com/miekg/dns/clientconfig.go",
        "github.com/miekg/dns/defaults.go",
        "github.com/miekg/dns/dns.go",
        "github.com/miekg/dns/dnssec.go",
        "github.com/miekg/dns/dnssec_keygen.go",
        "github.com/miekg/dns/dnssec_keyscan.go",
        "github.com/miekg/dns/dnssec_privkey.go",
        "github.com/miekg/dns/doc.go",
        "github.com/miekg/dns/edns.go",
        "github.com/miekg/dns/format.go",
        "github.com/miekg/dns/generate.go",
        "github.com/miekg/dns/labels.go",
        "github.com/miekg/dns/msg.go",
        "github.com/miekg/dns/msg_helpers.go",
        "github.com/miekg/dns/nsecx.go",
        "github.com/miekg/dns/privaterr.go",
        "github.com/miekg/dns/rawmsg.go",
        "github.com/miekg/dns/reverse.go",
        "github.com/miekg/dns/sanitize.go",
        "github.com/miekg/dns/scan.go",
        "github.com/miekg/dns/scan_rr.go",
        "github.com/miekg/dns/scanner.go",
        "github.com/miekg/dns/server.go",
        "github.com/miekg/dns/sig0.go",
        "github.com/miekg/dns/singleinflight.go",
        "github.com/miekg/dns/tlsa.go",
        "github.com/miekg/dns/tsig.go",
        "github.com/miekg/dns/types.go",
        "github.com/miekg/dns/udp.go",
        "github.com/miekg/dns/udp_linux.go",
        "github.com/miekg/dns/update.go",
        "github.com/miekg/dns/xfr.go",
        "github.com/miekg/dns/zmsg.go",
        "github.com/miekg/dns/ztypes.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mistifyio/go-zfs",
    srcs = [
        "github.com/mistifyio/go-zfs/error.go",
        "github.com/mistifyio/go-zfs/utils.go",
        "github.com/mistifyio/go-zfs/zfs.go",
        "github.com/mistifyio/go-zfs/zpool.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/pborman/uuid"],
)

go_library(
    name = "github.com/mitchellh/go-wordwrap",
    srcs = ["github.com/mitchellh/go-wordwrap/wordwrap.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mitchellh/mapstructure",
    srcs = [
        "github.com/mitchellh/mapstructure/decode_hooks.go",
        "github.com/mitchellh/mapstructure/error.go",
        "github.com/mitchellh/mapstructure/mapstructure.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mreiferson/go-httpclient",
    srcs = ["github.com/mreiferson/go-httpclient/httpclient.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mvdan/xurls",
    srcs = [
        "github.com/mvdan/xurls/regex.go",
        "github.com/mvdan/xurls/schemes.go",
        "github.com/mvdan/xurls/tlds.go",
        "github.com/mvdan/xurls/tlds_pseudo.go",
        "github.com/mvdan/xurls/xurls.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/mxk/go-flowrate/flowrate",
    srcs = [
        "github.com/mxk/go-flowrate/flowrate/flowrate.go",
        "github.com/mxk/go-flowrate/flowrate/io.go",
        "github.com/mxk/go-flowrate/flowrate/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo",
    srcs = ["github.com/onsi/ginkgo/ginkgo_dsl.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/internal/codelocation",
        "//vendor:github.com/onsi/ginkgo/internal/failer",
        "//vendor:github.com/onsi/ginkgo/internal/remote",
        "//vendor:github.com/onsi/ginkgo/internal/suite",
        "//vendor:github.com/onsi/ginkgo/internal/testingtproxy",
        "//vendor:github.com/onsi/ginkgo/internal/writer",
        "//vendor:github.com/onsi/ginkgo/reporters",
        "//vendor:github.com/onsi/ginkgo/reporters/stenographer",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/config",
    srcs = ["github.com/onsi/ginkgo/config/config.go"],
    tags = ["automanaged"],
)

go_binary(
    name = "github.com/onsi/ginkgo/ginkgo_bin",
    library = ":github.com/onsi/ginkgo/ginkgo",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/bootstrap_command.go",
        "github.com/onsi/ginkgo/ginkgo/build_command.go",
        "github.com/onsi/ginkgo/ginkgo/convert_command.go",
        "github.com/onsi/ginkgo/ginkgo/generate_command.go",
        "github.com/onsi/ginkgo/ginkgo/help_command.go",
        "github.com/onsi/ginkgo/ginkgo/main.go",
        "github.com/onsi/ginkgo/ginkgo/nodot_command.go",
        "github.com/onsi/ginkgo/ginkgo/notifications.go",
        "github.com/onsi/ginkgo/ginkgo/run_command.go",
        "github.com/onsi/ginkgo/ginkgo/run_watch_and_build_command_flags.go",
        "github.com/onsi/ginkgo/ginkgo/suite_runner.go",
        "github.com/onsi/ginkgo/ginkgo/unfocus_command.go",
        "github.com/onsi/ginkgo/ginkgo/version_command.go",
        "github.com/onsi/ginkgo/ginkgo/watch_command.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/ginkgo/convert",
        "//vendor:github.com/onsi/ginkgo/ginkgo/interrupthandler",
        "//vendor:github.com/onsi/ginkgo/ginkgo/nodot",
        "//vendor:github.com/onsi/ginkgo/ginkgo/testrunner",
        "//vendor:github.com/onsi/ginkgo/ginkgo/testsuite",
        "//vendor:github.com/onsi/ginkgo/ginkgo/watch",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/convert",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/convert/ginkgo_ast_nodes.go",
        "github.com/onsi/ginkgo/ginkgo/convert/import.go",
        "github.com/onsi/ginkgo/ginkgo/convert/package_rewriter.go",
        "github.com/onsi/ginkgo/ginkgo/convert/test_finder.go",
        "github.com/onsi/ginkgo/ginkgo/convert/testfile_rewriter.go",
        "github.com/onsi/ginkgo/ginkgo/convert/testing_t_rewriter.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/interrupthandler",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/interrupthandler/interrupt_handler.go",
        "github.com/onsi/ginkgo/ginkgo/interrupthandler/sigquit_swallower_unix.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/nodot",
    srcs = ["github.com/onsi/ginkgo/ginkgo/nodot/nodot.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/testrunner",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/testrunner/log_writer.go",
        "github.com/onsi/ginkgo/ginkgo/testrunner/run_result.go",
        "github.com/onsi/ginkgo/ginkgo/testrunner/test_runner.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/ginkgo/testsuite",
        "//vendor:github.com/onsi/ginkgo/internal/remote",
        "//vendor:github.com/onsi/ginkgo/reporters/stenographer",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/testsuite",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/testsuite/test_suite.go",
        "github.com/onsi/ginkgo/ginkgo/testsuite/vendor_check_go16.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/ginkgo/watch",
    srcs = [
        "github.com/onsi/ginkgo/ginkgo/watch/delta.go",
        "github.com/onsi/ginkgo/ginkgo/watch/delta_tracker.go",
        "github.com/onsi/ginkgo/ginkgo/watch/dependencies.go",
        "github.com/onsi/ginkgo/ginkgo/watch/package_hash.go",
        "github.com/onsi/ginkgo/ginkgo/watch/package_hashes.go",
        "github.com/onsi/ginkgo/ginkgo/watch/suite.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/ginkgo/ginkgo/testsuite"],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/codelocation",
    srcs = ["github.com/onsi/ginkgo/internal/codelocation/code_location.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/ginkgo/types"],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/containernode",
    srcs = ["github.com/onsi/ginkgo/internal/containernode/container_node.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/internal/leafnodes",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/failer",
    srcs = ["github.com/onsi/ginkgo/internal/failer/failer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/ginkgo/types"],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/leafnodes",
    srcs = [
        "github.com/onsi/ginkgo/internal/leafnodes/benchmarker.go",
        "github.com/onsi/ginkgo/internal/leafnodes/interfaces.go",
        "github.com/onsi/ginkgo/internal/leafnodes/it_node.go",
        "github.com/onsi/ginkgo/internal/leafnodes/measure_node.go",
        "github.com/onsi/ginkgo/internal/leafnodes/runner.go",
        "github.com/onsi/ginkgo/internal/leafnodes/setup_nodes.go",
        "github.com/onsi/ginkgo/internal/leafnodes/suite_nodes.go",
        "github.com/onsi/ginkgo/internal/leafnodes/synchronized_after_suite_node.go",
        "github.com/onsi/ginkgo/internal/leafnodes/synchronized_before_suite_node.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/internal/codelocation",
        "//vendor:github.com/onsi/ginkgo/internal/failer",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/remote",
    srcs = [
        "github.com/onsi/ginkgo/internal/remote/aggregator.go",
        "github.com/onsi/ginkgo/internal/remote/forwarding_reporter.go",
        "github.com/onsi/ginkgo/internal/remote/output_interceptor.go",
        "github.com/onsi/ginkgo/internal/remote/output_interceptor_unix.go",
        "github.com/onsi/ginkgo/internal/remote/server.go",
        "github.com/onsi/ginkgo/internal/remote/syscall_dup_unix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/reporters",
        "//vendor:github.com/onsi/ginkgo/reporters/stenographer",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/spec",
    srcs = [
        "github.com/onsi/ginkgo/internal/spec/index_computer.go",
        "github.com/onsi/ginkgo/internal/spec/spec.go",
        "github.com/onsi/ginkgo/internal/spec/specs.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/internal/containernode",
        "//vendor:github.com/onsi/ginkgo/internal/leafnodes",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/specrunner",
    srcs = [
        "github.com/onsi/ginkgo/internal/specrunner/random_id.go",
        "github.com/onsi/ginkgo/internal/specrunner/spec_runner.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/internal/leafnodes",
        "//vendor:github.com/onsi/ginkgo/internal/spec",
        "//vendor:github.com/onsi/ginkgo/internal/writer",
        "//vendor:github.com/onsi/ginkgo/reporters",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/suite",
    srcs = ["github.com/onsi/ginkgo/internal/suite/suite.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/internal/containernode",
        "//vendor:github.com/onsi/ginkgo/internal/failer",
        "//vendor:github.com/onsi/ginkgo/internal/leafnodes",
        "//vendor:github.com/onsi/ginkgo/internal/spec",
        "//vendor:github.com/onsi/ginkgo/internal/specrunner",
        "//vendor:github.com/onsi/ginkgo/internal/writer",
        "//vendor:github.com/onsi/ginkgo/reporters",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/testingtproxy",
    srcs = ["github.com/onsi/ginkgo/internal/testingtproxy/testing_t_proxy.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/internal/writer",
    srcs = [
        "github.com/onsi/ginkgo/internal/writer/fake_writer.go",
        "github.com/onsi/ginkgo/internal/writer/writer.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/ginkgo/reporters",
    srcs = [
        "github.com/onsi/ginkgo/reporters/default_reporter.go",
        "github.com/onsi/ginkgo/reporters/fake_reporter.go",
        "github.com/onsi/ginkgo/reporters/junit_reporter.go",
        "github.com/onsi/ginkgo/reporters/reporter.go",
        "github.com/onsi/ginkgo/reporters/teamcity_reporter.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/reporters/stenographer",
        "//vendor:github.com/onsi/ginkgo/types",
    ],
)

go_library(
    name = "github.com/onsi/ginkgo/reporters/stenographer",
    srcs = [
        "github.com/onsi/ginkgo/reporters/stenographer/console_logging.go",
        "github.com/onsi/ginkgo/reporters/stenographer/fake_stenographer.go",
        "github.com/onsi/ginkgo/reporters/stenographer/stenographer.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/ginkgo/types"],
)

go_library(
    name = "github.com/onsi/ginkgo/types",
    srcs = [
        "github.com/onsi/ginkgo/types/code_location.go",
        "github.com/onsi/ginkgo/types/synchronization.go",
        "github.com/onsi/ginkgo/types/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/gomega",
    srcs = [
        "github.com/onsi/gomega/gomega_dsl.go",
        "github.com/onsi/gomega/matchers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/gomega/internal/assertion",
        "//vendor:github.com/onsi/gomega/internal/asyncassertion",
        "//vendor:github.com/onsi/gomega/internal/testingtsupport",
        "//vendor:github.com/onsi/gomega/matchers",
        "//vendor:github.com/onsi/gomega/types",
    ],
)

go_library(
    name = "github.com/onsi/gomega/format",
    srcs = ["github.com/onsi/gomega/format/format.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/gomega/gstruct",
    srcs = [
        "github.com/onsi/gomega/gstruct/elements.go",
        "github.com/onsi/gomega/gstruct/fields.go",
        "github.com/onsi/gomega/gstruct/ignore.go",
        "github.com/onsi/gomega/gstruct/pointer.go",
        "github.com/onsi/gomega/gstruct/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/gomega/format",
        "//vendor:github.com/onsi/gomega/gstruct/errors",
        "//vendor:github.com/onsi/gomega/types",
    ],
)

go_library(
    name = "github.com/onsi/gomega/gstruct/errors",
    srcs = ["github.com/onsi/gomega/gstruct/errors/nested_types.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/gomega/types"],
)

go_library(
    name = "github.com/onsi/gomega/internal/assertion",
    srcs = ["github.com/onsi/gomega/internal/assertion/assertion.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/gomega/types"],
)

go_library(
    name = "github.com/onsi/gomega/internal/asyncassertion",
    srcs = ["github.com/onsi/gomega/internal/asyncassertion/async_assertion.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/gomega/internal/oraclematcher",
        "//vendor:github.com/onsi/gomega/types",
    ],
)

go_library(
    name = "github.com/onsi/gomega/internal/oraclematcher",
    srcs = ["github.com/onsi/gomega/internal/oraclematcher/oracle_matcher.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/gomega/types"],
)

go_library(
    name = "github.com/onsi/gomega/internal/testingtsupport",
    srcs = ["github.com/onsi/gomega/internal/testingtsupport/testing_t_support.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/gomega/types"],
)

go_library(
    name = "github.com/onsi/gomega/matchers",
    srcs = [
        "github.com/onsi/gomega/matchers/and.go",
        "github.com/onsi/gomega/matchers/assignable_to_type_of_matcher.go",
        "github.com/onsi/gomega/matchers/be_a_directory.go",
        "github.com/onsi/gomega/matchers/be_a_regular_file.go",
        "github.com/onsi/gomega/matchers/be_an_existing_file.go",
        "github.com/onsi/gomega/matchers/be_closed_matcher.go",
        "github.com/onsi/gomega/matchers/be_empty_matcher.go",
        "github.com/onsi/gomega/matchers/be_equivalent_to_matcher.go",
        "github.com/onsi/gomega/matchers/be_false_matcher.go",
        "github.com/onsi/gomega/matchers/be_identical_to.go",
        "github.com/onsi/gomega/matchers/be_nil_matcher.go",
        "github.com/onsi/gomega/matchers/be_numerically_matcher.go",
        "github.com/onsi/gomega/matchers/be_sent_matcher.go",
        "github.com/onsi/gomega/matchers/be_temporally_matcher.go",
        "github.com/onsi/gomega/matchers/be_true_matcher.go",
        "github.com/onsi/gomega/matchers/be_zero_matcher.go",
        "github.com/onsi/gomega/matchers/consist_of.go",
        "github.com/onsi/gomega/matchers/contain_element_matcher.go",
        "github.com/onsi/gomega/matchers/contain_substring_matcher.go",
        "github.com/onsi/gomega/matchers/equal_matcher.go",
        "github.com/onsi/gomega/matchers/have_cap_matcher.go",
        "github.com/onsi/gomega/matchers/have_key_matcher.go",
        "github.com/onsi/gomega/matchers/have_key_with_value_matcher.go",
        "github.com/onsi/gomega/matchers/have_len_matcher.go",
        "github.com/onsi/gomega/matchers/have_occurred_matcher.go",
        "github.com/onsi/gomega/matchers/have_prefix_matcher.go",
        "github.com/onsi/gomega/matchers/have_suffix_matcher.go",
        "github.com/onsi/gomega/matchers/match_error_matcher.go",
        "github.com/onsi/gomega/matchers/match_json_matcher.go",
        "github.com/onsi/gomega/matchers/match_regexp_matcher.go",
        "github.com/onsi/gomega/matchers/match_yaml_matcher.go",
        "github.com/onsi/gomega/matchers/not.go",
        "github.com/onsi/gomega/matchers/or.go",
        "github.com/onsi/gomega/matchers/panic_matcher.go",
        "github.com/onsi/gomega/matchers/receive_matcher.go",
        "github.com/onsi/gomega/matchers/succeed_matcher.go",
        "github.com/onsi/gomega/matchers/type_support.go",
        "github.com/onsi/gomega/matchers/with_transform.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/gomega/format",
        "//vendor:github.com/onsi/gomega/internal/oraclematcher",
        "//vendor:github.com/onsi/gomega/matchers/support/goraph/bipartitegraph",
        "//vendor:github.com/onsi/gomega/types",
        "//vendor:gopkg.in/yaml.v2",
    ],
)

go_library(
    name = "github.com/onsi/gomega/matchers/support/goraph/bipartitegraph",
    srcs = [
        "github.com/onsi/gomega/matchers/support/goraph/bipartitegraph/bipartitegraph.go",
        "github.com/onsi/gomega/matchers/support/goraph/bipartitegraph/bipartitegraphmatching.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/gomega/matchers/support/goraph/edge",
        "//vendor:github.com/onsi/gomega/matchers/support/goraph/node",
        "//vendor:github.com/onsi/gomega/matchers/support/goraph/util",
    ],
)

go_library(
    name = "github.com/onsi/gomega/matchers/support/goraph/edge",
    srcs = ["github.com/onsi/gomega/matchers/support/goraph/edge/edge.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/onsi/gomega/matchers/support/goraph/node"],
)

go_library(
    name = "github.com/onsi/gomega/matchers/support/goraph/node",
    srcs = ["github.com/onsi/gomega/matchers/support/goraph/node/node.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/gomega/matchers/support/goraph/util",
    srcs = ["github.com/onsi/gomega/matchers/support/goraph/util/util.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/onsi/gomega/types",
    srcs = ["github.com/onsi/gomega/types/types.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/capabilities_linux.go",
        "github.com/opencontainers/runc/libcontainer/console.go",
        "github.com/opencontainers/runc/libcontainer/console_linux.go",
        "github.com/opencontainers/runc/libcontainer/container.go",
        "github.com/opencontainers/runc/libcontainer/container_linux.go",
        "github.com/opencontainers/runc/libcontainer/criu_opts_unix.go",
        "github.com/opencontainers/runc/libcontainer/error.go",
        "github.com/opencontainers/runc/libcontainer/factory.go",
        "github.com/opencontainers/runc/libcontainer/factory_linux.go",
        "github.com/opencontainers/runc/libcontainer/generic_error.go",
        "github.com/opencontainers/runc/libcontainer/init_linux.go",
        "github.com/opencontainers/runc/libcontainer/message_linux.go",
        "github.com/opencontainers/runc/libcontainer/network_linux.go",
        "github.com/opencontainers/runc/libcontainer/notify_linux.go",
        "github.com/opencontainers/runc/libcontainer/process.go",
        "github.com/opencontainers/runc/libcontainer/process_linux.go",
        "github.com/opencontainers/runc/libcontainer/restored_process.go",
        "github.com/opencontainers/runc/libcontainer/rootfs_linux.go",
        "github.com/opencontainers/runc/libcontainer/setgroups_linux.go",
        "github.com/opencontainers/runc/libcontainer/setns_init_linux.go",
        "github.com/opencontainers/runc/libcontainer/standard_init_linux.go",
        "github.com/opencontainers/runc/libcontainer/state_linux.go",
        "github.com/opencontainers/runc/libcontainer/stats.go",
        "github.com/opencontainers/runc/libcontainer/stats_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Sirupsen/logrus",
        "//vendor:github.com/docker/docker/pkg/mount",
        "//vendor:github.com/docker/docker/pkg/symlink",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/opencontainers/runc/libcontainer/apparmor",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/systemd",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs/validate",
        "//vendor:github.com/opencontainers/runc/libcontainer/criurpc",
        "//vendor:github.com/opencontainers/runc/libcontainer/keys",
        "//vendor:github.com/opencontainers/runc/libcontainer/label",
        "//vendor:github.com/opencontainers/runc/libcontainer/seccomp",
        "//vendor:github.com/opencontainers/runc/libcontainer/stacktrace",
        "//vendor:github.com/opencontainers/runc/libcontainer/system",
        "//vendor:github.com/opencontainers/runc/libcontainer/user",
        "//vendor:github.com/opencontainers/runc/libcontainer/utils",
        "//vendor:github.com/syndtr/gocapability/capability",
        "//vendor:github.com/vishvananda/netlink",
        "//vendor:github.com/vishvananda/netlink/nl",
    ],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/apparmor",
    srcs = ["github.com/opencontainers/runc/libcontainer/apparmor/apparmor_disabled.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/cgroups",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/cgroups/cgroups.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/stats.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/go-units",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
    ],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/cgroups/fs",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/apply_raw.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/blkio.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/cpu.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/cpuacct.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/cpuset.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/devices.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/freezer.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/hugetlb.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/memory.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/name.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/net_cls.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/net_prio.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/perf_event.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/pids.go",
        "github.com/opencontainers/runc/libcontainer/cgroups/fs/utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:github.com/opencontainers/runc/libcontainer/system",
        "//vendor:github.com/opencontainers/runc/libcontainer/utils",
    ],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/cgroups/systemd",
    srcs = ["github.com/opencontainers/runc/libcontainer/cgroups/systemd/apply_systemd.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-systemd/dbus",
        "//vendor:github.com/coreos/go-systemd/util",
        "//vendor:github.com/godbus/dbus",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
    ],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/configs",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/configs/blkio_device.go",
        "github.com/opencontainers/runc/libcontainer/configs/cgroup_unix.go",
        "github.com/opencontainers/runc/libcontainer/configs/config.go",
        "github.com/opencontainers/runc/libcontainer/configs/config_unix.go",
        "github.com/opencontainers/runc/libcontainer/configs/device.go",
        "github.com/opencontainers/runc/libcontainer/configs/device_defaults.go",
        "github.com/opencontainers/runc/libcontainer/configs/hugepage_limit.go",
        "github.com/opencontainers/runc/libcontainer/configs/interface_priority_map.go",
        "github.com/opencontainers/runc/libcontainer/configs/mount.go",
        "github.com/opencontainers/runc/libcontainer/configs/namespaces.go",
        "github.com/opencontainers/runc/libcontainer/configs/namespaces_syscall.go",
        "github.com/opencontainers/runc/libcontainer/configs/namespaces_unix.go",
        "github.com/opencontainers/runc/libcontainer/configs/network.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/Sirupsen/logrus"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/configs/validate",
    srcs = ["github.com/opencontainers/runc/libcontainer/configs/validate/validator.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:github.com/opencontainers/runc/libcontainer/selinux",
    ],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/criurpc",
    srcs = ["github.com/opencontainers/runc/libcontainer/criurpc/criurpc.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/keys",
    srcs = ["github.com/opencontainers/runc/libcontainer/keys/keyctl.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/label",
    srcs = ["github.com/opencontainers/runc/libcontainer/label/label.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/seccomp",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/seccomp/config.go",
        "github.com/opencontainers/runc/libcontainer/seccomp/seccomp_unsupported.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/opencontainers/runc/libcontainer/configs"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/selinux",
    srcs = ["github.com/opencontainers/runc/libcontainer/selinux/selinux.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/opencontainers/runc/libcontainer/system"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/stacktrace",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/stacktrace/capture.go",
        "github.com/opencontainers/runc/libcontainer/stacktrace/frame.go",
        "github.com/opencontainers/runc/libcontainer/stacktrace/stacktrace.go",
    ],
    tags = ["automanaged"],
)

cgo_genrule(
    name = "github.com/opencontainers/runc/libcontainer/system_cgo",
    srcs = ["github.com/opencontainers/runc/libcontainer/system/sysconfig.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/system",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/system/linux.go",
        "github.com/opencontainers/runc/libcontainer/system/proc.go",
        "github.com/opencontainers/runc/libcontainer/system/setns_linux.go",
        "github.com/opencontainers/runc/libcontainer/system/syscall_linux_64.go",
        "github.com/opencontainers/runc/libcontainer/system/xattrs_linux.go",
    ],
    library = ":github.com/opencontainers/runc/libcontainer/system_cgo",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/user",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/user/lookup.go",
        "github.com/opencontainers/runc/libcontainer/user/lookup_unix.go",
        "github.com/opencontainers/runc/libcontainer/user/user.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/opencontainers/runc/libcontainer/utils",
    srcs = [
        "github.com/opencontainers/runc/libcontainer/utils/utils.go",
        "github.com/opencontainers/runc/libcontainer/utils/utils_unix.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/pborman/uuid",
    srcs = [
        "github.com/pborman/uuid/dce.go",
        "github.com/pborman/uuid/doc.go",
        "github.com/pborman/uuid/hash.go",
        "github.com/pborman/uuid/json.go",
        "github.com/pborman/uuid/node.go",
        "github.com/pborman/uuid/time.go",
        "github.com/pborman/uuid/util.go",
        "github.com/pborman/uuid/uuid.go",
        "github.com/pborman/uuid/version1.go",
        "github.com/pborman/uuid/version4.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/pelletier/go-buffruneio",
    srcs = ["github.com/pelletier/go-buffruneio/buffruneio.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/pelletier/go-toml",
    srcs = [
        "github.com/pelletier/go-toml/doc.go",
        "github.com/pelletier/go-toml/keysparsing.go",
        "github.com/pelletier/go-toml/lexer.go",
        "github.com/pelletier/go-toml/match.go",
        "github.com/pelletier/go-toml/parser.go",
        "github.com/pelletier/go-toml/position.go",
        "github.com/pelletier/go-toml/query.go",
        "github.com/pelletier/go-toml/querylexer.go",
        "github.com/pelletier/go-toml/queryparser.go",
        "github.com/pelletier/go-toml/token.go",
        "github.com/pelletier/go-toml/toml.go",
        "github.com/pelletier/go-toml/tomltree_conversions.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/pelletier/go-buffruneio"],
)

go_library(
    name = "github.com/pkg/errors",
    srcs = [
        "github.com/pkg/errors/errors.go",
        "github.com/pkg/errors/stack.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/pkg/sftp",
    srcs = [
        "github.com/pkg/sftp/attrs.go",
        "github.com/pkg/sftp/attrs_unix.go",
        "github.com/pkg/sftp/client.go",
        "github.com/pkg/sftp/conn.go",
        "github.com/pkg/sftp/packet.go",
        "github.com/pkg/sftp/release.go",
        "github.com/pkg/sftp/server.go",
        "github.com/pkg/sftp/server_statvfs_impl.go",
        "github.com/pkg/sftp/server_statvfs_linux.go",
        "github.com/pkg/sftp/server_unix.go",
        "github.com/pkg/sftp/sftp.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/kr/fs",
        "//vendor:github.com/pkg/errors",
        "//vendor:golang.org/x/crypto/ssh",
    ],
)

go_library(
    name = "github.com/pmezard/go-difflib/difflib",
    srcs = ["github.com/pmezard/go-difflib/difflib/difflib.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/prometheus/client_golang/prometheus",
    srcs = [
        "github.com/prometheus/client_golang/prometheus/collector.go",
        "github.com/prometheus/client_golang/prometheus/counter.go",
        "github.com/prometheus/client_golang/prometheus/desc.go",
        "github.com/prometheus/client_golang/prometheus/doc.go",
        "github.com/prometheus/client_golang/prometheus/expvar.go",
        "github.com/prometheus/client_golang/prometheus/gauge.go",
        "github.com/prometheus/client_golang/prometheus/go_collector.go",
        "github.com/prometheus/client_golang/prometheus/histogram.go",
        "github.com/prometheus/client_golang/prometheus/http.go",
        "github.com/prometheus/client_golang/prometheus/metric.go",
        "github.com/prometheus/client_golang/prometheus/process_collector.go",
        "github.com/prometheus/client_golang/prometheus/push.go",
        "github.com/prometheus/client_golang/prometheus/registry.go",
        "github.com/prometheus/client_golang/prometheus/summary.go",
        "github.com/prometheus/client_golang/prometheus/untyped.go",
        "github.com/prometheus/client_golang/prometheus/value.go",
        "github.com/prometheus/client_golang/prometheus/vec.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/beorn7/perks/quantile",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/prometheus/client_model/go",
        "//vendor:github.com/prometheus/common/expfmt",
        "//vendor:github.com/prometheus/procfs",
    ],
)

go_library(
    name = "github.com/prometheus/client_model/go",
    srcs = ["github.com/prometheus/client_model/go/metrics.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "github.com/prometheus/common/expfmt",
    srcs = [
        "github.com/prometheus/common/expfmt/decode.go",
        "github.com/prometheus/common/expfmt/encode.go",
        "github.com/prometheus/common/expfmt/expfmt.go",
        "github.com/prometheus/common/expfmt/json_decode.go",
        "github.com/prometheus/common/expfmt/text_create.go",
        "github.com/prometheus/common/expfmt/text_parse.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:bitbucket.org/ww/goautoneg",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/matttproud/golang_protobuf_extensions/pbutil",
        "//vendor:github.com/prometheus/client_model/go",
        "//vendor:github.com/prometheus/common/model",
    ],
)

go_library(
    name = "github.com/prometheus/common/model",
    srcs = [
        "github.com/prometheus/common/model/alert.go",
        "github.com/prometheus/common/model/fingerprinting.go",
        "github.com/prometheus/common/model/labels.go",
        "github.com/prometheus/common/model/labelset.go",
        "github.com/prometheus/common/model/metric.go",
        "github.com/prometheus/common/model/model.go",
        "github.com/prometheus/common/model/signature.go",
        "github.com/prometheus/common/model/silence.go",
        "github.com/prometheus/common/model/time.go",
        "github.com/prometheus/common/model/value.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/prometheus/procfs",
    srcs = [
        "github.com/prometheus/procfs/doc.go",
        "github.com/prometheus/procfs/fs.go",
        "github.com/prometheus/procfs/ipvs.go",
        "github.com/prometheus/procfs/mdstat.go",
        "github.com/prometheus/procfs/proc.go",
        "github.com/prometheus/procfs/proc_io.go",
        "github.com/prometheus/procfs/proc_limits.go",
        "github.com/prometheus/procfs/proc_stat.go",
        "github.com/prometheus/procfs/stat.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/quobyte/api",
    srcs = [
        "github.com/quobyte/api/quobyte.go",
        "github.com/quobyte/api/rpc_client.go",
        "github.com/quobyte/api/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/rackspace/gophercloud",
    srcs = [
        "github.com/rackspace/gophercloud/auth_options.go",
        "github.com/rackspace/gophercloud/auth_results.go",
        "github.com/rackspace/gophercloud/doc.go",
        "github.com/rackspace/gophercloud/endpoint_search.go",
        "github.com/rackspace/gophercloud/params.go",
        "github.com/rackspace/gophercloud/provider_client.go",
        "github.com/rackspace/gophercloud/results.go",
        "github.com/rackspace/gophercloud/service_client.go",
        "github.com/rackspace/gophercloud/util.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/mitchellh/mapstructure"],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/auth_env.go",
        "github.com/rackspace/gophercloud/openstack/client.go",
        "github.com/rackspace/gophercloud/openstack/endpoint_location.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v2/tokens",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v3/tokens",
        "//vendor:github.com/rackspace/gophercloud/openstack/utils",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/doc.go",
        "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/requests.go",
        "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/results.go",
        "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/urls.go",
        "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/common/extensions",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/common/extensions/doc.go",
        "github.com/rackspace/gophercloud/openstack/common/extensions/errors.go",
        "github.com/rackspace/gophercloud/openstack/common/extensions/requests.go",
        "github.com/rackspace/gophercloud/openstack/common/extensions/results.go",
        "github.com/rackspace/gophercloud/openstack/common/extensions/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume/results.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig/doc.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig/results.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach/doc.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach/results.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/flavors",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/flavors/doc.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/flavors/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/flavors/results.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/flavors/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/images",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/images/doc.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/images/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/images/results.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/images/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/compute/v2/servers",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/compute/v2/servers/doc.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/servers/requests.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/servers/results.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/servers/urls.go",
        "github.com/rackspace/gophercloud/openstack/compute/v2/servers/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/flavors",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/images",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/identity/v2/tenants",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/identity/v2/tenants/doc.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tenants/requests.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tenants/results.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tenants/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/identity/v2/tokens",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/identity/v2/tokens/doc.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tokens/errors.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tokens/requests.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tokens/results.go",
        "github.com/rackspace/gophercloud/openstack/identity/v2/tokens/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v2/tenants",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/identity/v3/extensions/trust",
    srcs = ["github.com/rackspace/gophercloud/openstack/identity/v3/extensions/trust/request.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v3/tokens",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/identity/v3/tokens",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/identity/v3/tokens/doc.go",
        "github.com/rackspace/gophercloud/openstack/identity/v3/tokens/errors.go",
        "github.com/rackspace/gophercloud/openstack/identity/v3/tokens/requests.go",
        "github.com/rackspace/gophercloud/openstack/identity/v3/tokens/results.go",
        "github.com/rackspace/gophercloud/openstack/identity/v3/tokens/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions",
    srcs = ["github.com/rackspace/gophercloud/openstack/networking/v2/extensions/delegate.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/common/extensions",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/networking/v2/ports",
    srcs = [
        "github.com/rackspace/gophercloud/openstack/networking/v2/ports/doc.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/ports/errors.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/ports/requests.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/ports/results.go",
        "github.com/rackspace/gophercloud/openstack/networking/v2/ports/urls.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/openstack/utils",
    srcs = ["github.com/rackspace/gophercloud/openstack/utils/choose_version.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/rackspace/gophercloud"],
)

go_library(
    name = "github.com/rackspace/gophercloud/pagination",
    srcs = [
        "github.com/rackspace/gophercloud/pagination/http.go",
        "github.com/rackspace/gophercloud/pagination/linked.go",
        "github.com/rackspace/gophercloud/pagination/marker.go",
        "github.com/rackspace/gophercloud/pagination/null.go",
        "github.com/rackspace/gophercloud/pagination/pager.go",
        "github.com/rackspace/gophercloud/pagination/pkg.go",
        "github.com/rackspace/gophercloud/pagination/single.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/rackspace/gophercloud"],
)

go_library(
    name = "github.com/rackspace/gophercloud/rackspace",
    srcs = [
        "github.com/rackspace/gophercloud/rackspace/auth_env.go",
        "github.com/rackspace/gophercloud/rackspace/client.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack",
        "//vendor:github.com/rackspace/gophercloud/openstack/utils",
        "//vendor:github.com/rackspace/gophercloud/rackspace/identity/v2/tokens",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes",
    srcs = [
        "github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes/delegate.go",
        "github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes/doc.go",
        "github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes/results.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/rackspace/compute/v2/servers",
    srcs = [
        "github.com/rackspace/gophercloud/rackspace/compute/v2/servers/delegate.go",
        "github.com/rackspace/gophercloud/rackspace/compute/v2/servers/doc.go",
        "github.com/rackspace/gophercloud/rackspace/compute/v2/servers/requests.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach",
    srcs = [
        "github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach/delegate.go",
        "github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach",
        "//vendor:github.com/rackspace/gophercloud/pagination",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/rackspace/identity/v2/tokens",
    srcs = [
        "github.com/rackspace/gophercloud/rackspace/identity/v2/tokens/delegate.go",
        "github.com/rackspace/gophercloud/rackspace/identity/v2/tokens/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/identity/v2/tokens",
    ],
)

go_library(
    name = "github.com/rackspace/gophercloud/testhelper",
    srcs = [
        "github.com/rackspace/gophercloud/testhelper/convenience.go",
        "github.com/rackspace/gophercloud/testhelper/doc.go",
        "github.com/rackspace/gophercloud/testhelper/http_responses.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/rackspace/gophercloud/testhelper/client",
    srcs = ["github.com/rackspace/gophercloud/testhelper/client/fake.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/testhelper",
    ],
)

go_library(
    name = "github.com/renstrom/dedent",
    srcs = ["github.com/renstrom/dedent/dedent.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/robfig/cron",
    srcs = [
        "github.com/robfig/cron/constantdelay.go",
        "github.com/robfig/cron/cron.go",
        "github.com/robfig/cron/doc.go",
        "github.com/robfig/cron/parser.go",
        "github.com/robfig/cron/spec.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/rubiojr/go-vhd/vhd",
    srcs = [
        "github.com/rubiojr/go-vhd/vhd/util.go",
        "github.com/rubiojr/go-vhd/vhd/vhd.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/russross/blackfriday",
    srcs = [
        "github.com/russross/blackfriday/block.go",
        "github.com/russross/blackfriday/html.go",
        "github.com/russross/blackfriday/inline.go",
        "github.com/russross/blackfriday/latex.go",
        "github.com/russross/blackfriday/markdown.go",
        "github.com/russross/blackfriday/smartypants.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/shurcooL/sanitized_anchor_name"],
)

go_library(
    name = "github.com/samuel/go-zookeeper/zk",
    srcs = [
        "github.com/samuel/go-zookeeper/zk/conn.go",
        "github.com/samuel/go-zookeeper/zk/constants.go",
        "github.com/samuel/go-zookeeper/zk/flw.go",
        "github.com/samuel/go-zookeeper/zk/lock.go",
        "github.com/samuel/go-zookeeper/zk/server_help.go",
        "github.com/samuel/go-zookeeper/zk/server_java.go",
        "github.com/samuel/go-zookeeper/zk/structs.go",
        "github.com/samuel/go-zookeeper/zk/tracer.go",
        "github.com/samuel/go-zookeeper/zk/util.go",
    ],
    tags = ["automanaged"],
)

cgo_genrule(
    name = "github.com/seccomp/libseccomp-golang_cgo",
    srcs = [
        "github.com/seccomp/libseccomp-golang/seccomp.go",
        "github.com/seccomp/libseccomp-golang/seccomp_internal.go",
    ],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/seccomp/libseccomp-golang",
    srcs = [],
    library = ":github.com/seccomp/libseccomp-golang_cgo",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/shurcooL/sanitized_anchor_name",
    srcs = ["github.com/shurcooL/sanitized_anchor_name/main.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/spf13/afero",
    srcs = [
        "github.com/spf13/afero/afero.go",
        "github.com/spf13/afero/basepath.go",
        "github.com/spf13/afero/cacheOnReadFs.go",
        "github.com/spf13/afero/const_win_unix.go",
        "github.com/spf13/afero/copyOnWriteFs.go",
        "github.com/spf13/afero/httpFs.go",
        "github.com/spf13/afero/ioutil.go",
        "github.com/spf13/afero/memmap.go",
        "github.com/spf13/afero/memradix.go",
        "github.com/spf13/afero/os.go",
        "github.com/spf13/afero/path.go",
        "github.com/spf13/afero/readonlyfs.go",
        "github.com/spf13/afero/regexpfs.go",
        "github.com/spf13/afero/sftp.go",
        "github.com/spf13/afero/unionFile.go",
        "github.com/spf13/afero/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/pkg/sftp",
        "//vendor:github.com/spf13/afero/mem",
        "//vendor:github.com/spf13/afero/sftp",
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/norm",
    ],
)

go_library(
    name = "github.com/spf13/afero/mem",
    srcs = [
        "github.com/spf13/afero/mem/dir.go",
        "github.com/spf13/afero/mem/dirmap.go",
        "github.com/spf13/afero/mem/file.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/spf13/afero/sftp",
    srcs = ["github.com/spf13/afero/sftp/file.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/pkg/sftp"],
)

go_library(
    name = "github.com/spf13/cast",
    srcs = [
        "github.com/spf13/cast/cast.go",
        "github.com/spf13/cast/caste.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/jwalterweatherman"],
)

go_library(
    name = "github.com/spf13/cobra",
    srcs = [
        "github.com/spf13/cobra/bash_completions.go",
        "github.com/spf13/cobra/cobra.go",
        "github.com/spf13/cobra/command.go",
        "github.com/spf13/cobra/command_notwin.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)

go_library(
    name = "github.com/spf13/cobra/doc",
    srcs = [
        "github.com/spf13/cobra/doc/man_docs.go",
        "github.com/spf13/cobra/doc/md_docs.go",
        "github.com/spf13/cobra/doc/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/cpuguy83/go-md2man/md2man",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_library(
    name = "github.com/spf13/jwalterweatherman",
    srcs = ["github.com/spf13/jwalterweatherman/thatswhyyoualwaysleaveanote.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/spf13/pflag",
    srcs = [
        "github.com/spf13/pflag/bool.go",
        "github.com/spf13/pflag/count.go",
        "github.com/spf13/pflag/duration.go",
        "github.com/spf13/pflag/flag.go",
        "github.com/spf13/pflag/float32.go",
        "github.com/spf13/pflag/float64.go",
        "github.com/spf13/pflag/golangflag.go",
        "github.com/spf13/pflag/int.go",
        "github.com/spf13/pflag/int32.go",
        "github.com/spf13/pflag/int64.go",
        "github.com/spf13/pflag/int8.go",
        "github.com/spf13/pflag/int_slice.go",
        "github.com/spf13/pflag/ip.go",
        "github.com/spf13/pflag/ipmask.go",
        "github.com/spf13/pflag/ipnet.go",
        "github.com/spf13/pflag/string.go",
        "github.com/spf13/pflag/string_array.go",
        "github.com/spf13/pflag/string_slice.go",
        "github.com/spf13/pflag/uint.go",
        "github.com/spf13/pflag/uint16.go",
        "github.com/spf13/pflag/uint32.go",
        "github.com/spf13/pflag/uint64.go",
        "github.com/spf13/pflag/uint8.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/spf13/viper",
    srcs = [
        "github.com/spf13/viper/flags.go",
        "github.com/spf13/viper/util.go",
        "github.com/spf13/viper/viper.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/fsnotify/fsnotify",
        "//vendor:github.com/hashicorp/hcl",
        "//vendor:github.com/magiconair/properties",
        "//vendor:github.com/mitchellh/mapstructure",
        "//vendor:github.com/pelletier/go-toml",
        "//vendor:github.com/spf13/afero",
        "//vendor:github.com/spf13/cast",
        "//vendor:github.com/spf13/jwalterweatherman",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/yaml.v2",
    ],
)

go_library(
    name = "github.com/square/go-jose",
    srcs = [
        "github.com/square/go-jose/asymmetric.go",
        "github.com/square/go-jose/crypter.go",
        "github.com/square/go-jose/doc.go",
        "github.com/square/go-jose/encoding.go",
        "github.com/square/go-jose/jwe.go",
        "github.com/square/go-jose/jwk.go",
        "github.com/square/go-jose/jws.go",
        "github.com/square/go-jose/shared.go",
        "github.com/square/go-jose/signing.go",
        "github.com/square/go-jose/symmetric.go",
        "github.com/square/go-jose/utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/square/go-jose/cipher",
        "//vendor:github.com/square/go-jose/json",
    ],
)

go_library(
    name = "github.com/square/go-jose/cipher",
    srcs = [
        "github.com/square/go-jose/cipher/cbc_hmac.go",
        "github.com/square/go-jose/cipher/concat_kdf.go",
        "github.com/square/go-jose/cipher/ecdh_es.go",
        "github.com/square/go-jose/cipher/key_wrap.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/square/go-jose/json",
    srcs = [
        "github.com/square/go-jose/json/decode.go",
        "github.com/square/go-jose/json/encode.go",
        "github.com/square/go-jose/json/indent.go",
        "github.com/square/go-jose/json/scanner.go",
        "github.com/square/go-jose/json/stream.go",
        "github.com/square/go-jose/json/tags.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/stretchr/objx",
    srcs = [
        "github.com/stretchr/objx/accessors.go",
        "github.com/stretchr/objx/constants.go",
        "github.com/stretchr/objx/conversions.go",
        "github.com/stretchr/objx/doc.go",
        "github.com/stretchr/objx/map.go",
        "github.com/stretchr/objx/mutations.go",
        "github.com/stretchr/objx/security.go",
        "github.com/stretchr/objx/tests.go",
        "github.com/stretchr/objx/type_specific_codegen.go",
        "github.com/stretchr/objx/value.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/stretchr/testify/assert",
    srcs = [
        "github.com/stretchr/testify/assert/assertions.go",
        "github.com/stretchr/testify/assert/doc.go",
        "github.com/stretchr/testify/assert/errors.go",
        "github.com/stretchr/testify/assert/forward_assertions.go",
        "github.com/stretchr/testify/assert/http_assertions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/pmezard/go-difflib/difflib",
    ],
)

go_library(
    name = "github.com/stretchr/testify/mock",
    srcs = [
        "github.com/stretchr/testify/mock/doc.go",
        "github.com/stretchr/testify/mock/mock.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/stretchr/objx",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)

go_library(
    name = "github.com/stretchr/testify/require",
    srcs = [
        "github.com/stretchr/testify/require/doc.go",
        "github.com/stretchr/testify/require/forward_requirements.go",
        "github.com/stretchr/testify/require/requirements.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)

go_library(
    name = "github.com/syndtr/gocapability/capability",
    srcs = [
        "github.com/syndtr/gocapability/capability/capability.go",
        "github.com/syndtr/gocapability/capability/capability_linux.go",
        "github.com/syndtr/gocapability/capability/enum.go",
        "github.com/syndtr/gocapability/capability/enum_gen.go",
        "github.com/syndtr/gocapability/capability/syscall_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/ugorji/go/codec",
    srcs = [
        "github.com/ugorji/go/codec/0doc.go",
        "github.com/ugorji/go/codec/binc.go",
        "github.com/ugorji/go/codec/cbor.go",
        "github.com/ugorji/go/codec/decode.go",
        "github.com/ugorji/go/codec/decode_go.go",
        "github.com/ugorji/go/codec/encode.go",
        "github.com/ugorji/go/codec/fast-path.generated.go",
        "github.com/ugorji/go/codec/gen.generated.go",
        "github.com/ugorji/go/codec/gen.go",
        "github.com/ugorji/go/codec/gen-helper.generated.go",
        "github.com/ugorji/go/codec/gen_16.go",
        "github.com/ugorji/go/codec/gen_17.go",
        "github.com/ugorji/go/codec/helper.go",
        "github.com/ugorji/go/codec/helper_internal.go",
        "github.com/ugorji/go/codec/helper_not_unsafe.go",
        "github.com/ugorji/go/codec/json.go",
        "github.com/ugorji/go/codec/msgpack.go",
        "github.com/ugorji/go/codec/noop.go",
        "github.com/ugorji/go/codec/prebuild.go",
        "github.com/ugorji/go/codec/rpc.go",
        "github.com/ugorji/go/codec/simple.go",
        "github.com/ugorji/go/codec/time.go",
    ],
    tags = ["automanaged"],
)

go_binary(
    name = "github.com/ugorji/go/codec/codecgen_bin",
    library = ":github.com/ugorji/go/codec/codecgen",
    tags = ["automanaged"],
)

go_library(
    name = "github.com/ugorji/go/codec/codecgen",
    srcs = [
        "github.com/ugorji/go/codec/codecgen/gen.go",
        "github.com/ugorji/go/codec/codecgen/z.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vishvananda/netlink",
    srcs = [
        "github.com/vishvananda/netlink/addr.go",
        "github.com/vishvananda/netlink/addr_linux.go",
        "github.com/vishvananda/netlink/filter.go",
        "github.com/vishvananda/netlink/filter_linux.go",
        "github.com/vishvananda/netlink/link.go",
        "github.com/vishvananda/netlink/link_linux.go",
        "github.com/vishvananda/netlink/neigh.go",
        "github.com/vishvananda/netlink/neigh_linux.go",
        "github.com/vishvananda/netlink/netlink.go",
        "github.com/vishvananda/netlink/protinfo.go",
        "github.com/vishvananda/netlink/protinfo_linux.go",
        "github.com/vishvananda/netlink/qdisc.go",
        "github.com/vishvananda/netlink/qdisc_linux.go",
        "github.com/vishvananda/netlink/route.go",
        "github.com/vishvananda/netlink/route_linux.go",
        "github.com/vishvananda/netlink/xfrm.go",
        "github.com/vishvananda/netlink/xfrm_policy.go",
        "github.com/vishvananda/netlink/xfrm_policy_linux.go",
        "github.com/vishvananda/netlink/xfrm_state.go",
        "github.com/vishvananda/netlink/xfrm_state_linux.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/vishvananda/netlink/nl"],
)

go_library(
    name = "github.com/vishvananda/netlink/nl",
    srcs = [
        "github.com/vishvananda/netlink/nl/addr_linux.go",
        "github.com/vishvananda/netlink/nl/link_linux.go",
        "github.com/vishvananda/netlink/nl/nl_linux.go",
        "github.com/vishvananda/netlink/nl/route_linux.go",
        "github.com/vishvananda/netlink/nl/tc_linux.go",
        "github.com/vishvananda/netlink/nl/xfrm_linux.go",
        "github.com/vishvananda/netlink/nl/xfrm_policy_linux.go",
        "github.com/vishvananda/netlink/nl/xfrm_state_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vmware/govmomi",
    srcs = ["github.com/vmware/govmomi/client.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/session",
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/find",
    srcs = [
        "github.com/vmware/govmomi/find/error.go",
        "github.com/vmware/govmomi/find/finder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/list",
        "//vendor:github.com/vmware/govmomi/object",
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/list",
    srcs = [
        "github.com/vmware/govmomi/list/lister.go",
        "github.com/vmware/govmomi/list/path.go",
        "github.com/vmware/govmomi/list/recurser.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/object",
    srcs = [
        "github.com/vmware/govmomi/object/authorization_manager.go",
        "github.com/vmware/govmomi/object/cluster_compute_resource.go",
        "github.com/vmware/govmomi/object/common.go",
        "github.com/vmware/govmomi/object/compute_resource.go",
        "github.com/vmware/govmomi/object/custom_fields_manager.go",
        "github.com/vmware/govmomi/object/customization_spec_manager.go",
        "github.com/vmware/govmomi/object/datacenter.go",
        "github.com/vmware/govmomi/object/datastore.go",
        "github.com/vmware/govmomi/object/diagnostic_manager.go",
        "github.com/vmware/govmomi/object/distributed_virtual_portgroup.go",
        "github.com/vmware/govmomi/object/distributed_virtual_switch.go",
        "github.com/vmware/govmomi/object/extension_manager.go",
        "github.com/vmware/govmomi/object/file_manager.go",
        "github.com/vmware/govmomi/object/folder.go",
        "github.com/vmware/govmomi/object/history_collector.go",
        "github.com/vmware/govmomi/object/host_account_manager.go",
        "github.com/vmware/govmomi/object/host_config_manager.go",
        "github.com/vmware/govmomi/object/host_datastore_browser.go",
        "github.com/vmware/govmomi/object/host_datastore_system.go",
        "github.com/vmware/govmomi/object/host_firewall_system.go",
        "github.com/vmware/govmomi/object/host_network_system.go",
        "github.com/vmware/govmomi/object/host_service_system.go",
        "github.com/vmware/govmomi/object/host_storage_system.go",
        "github.com/vmware/govmomi/object/host_system.go",
        "github.com/vmware/govmomi/object/host_virtual_nic_manager.go",
        "github.com/vmware/govmomi/object/host_vsan_system.go",
        "github.com/vmware/govmomi/object/http_nfc_lease.go",
        "github.com/vmware/govmomi/object/list_view.go",
        "github.com/vmware/govmomi/object/namespace_manager.go",
        "github.com/vmware/govmomi/object/network.go",
        "github.com/vmware/govmomi/object/network_reference.go",
        "github.com/vmware/govmomi/object/option_manager.go",
        "github.com/vmware/govmomi/object/ovf_manager.go",
        "github.com/vmware/govmomi/object/resource_pool.go",
        "github.com/vmware/govmomi/object/search_index.go",
        "github.com/vmware/govmomi/object/storage_pod.go",
        "github.com/vmware/govmomi/object/storage_resource_manager.go",
        "github.com/vmware/govmomi/object/task.go",
        "github.com/vmware/govmomi/object/types.go",
        "github.com/vmware/govmomi/object/virtual_app.go",
        "github.com/vmware/govmomi/object/virtual_device_list.go",
        "github.com/vmware/govmomi/object/virtual_disk_manager.go",
        "github.com/vmware/govmomi/object/virtual_machine.go",
        "github.com/vmware/govmomi/object/vmware_distributed_virtual_switch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/session",
        "//vendor:github.com/vmware/govmomi/task",
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/methods",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/progress",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/property",
    srcs = [
        "github.com/vmware/govmomi/property/collector.go",
        "github.com/vmware/govmomi/property/wait.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/methods",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/session",
    srcs = [
        "github.com/vmware/govmomi/session/keep_alive.go",
        "github.com/vmware/govmomi/session/manager.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/methods",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/task",
    srcs = [
        "github.com/vmware/govmomi/task/error.go",
        "github.com/vmware/govmomi/task/wait.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/vim25/progress",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/vim25",
    srcs = [
        "github.com/vmware/govmomi/vim25/client.go",
        "github.com/vmware/govmomi/vim25/doc.go",
        "github.com/vmware/govmomi/vim25/retry.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/vim25/methods",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/debug",
    srcs = ["github.com/vmware/govmomi/vim25/debug/debug.go"],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/methods",
    srcs = [
        "github.com/vmware/govmomi/vim25/methods/internal.go",
        "github.com/vmware/govmomi/vim25/methods/methods.go",
        "github.com/vmware/govmomi/vim25/methods/service_content.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/mo",
    srcs = [
        "github.com/vmware/govmomi/vim25/mo/ancestors.go",
        "github.com/vmware/govmomi/vim25/mo/entity.go",
        "github.com/vmware/govmomi/vim25/mo/extra.go",
        "github.com/vmware/govmomi/vim25/mo/mo.go",
        "github.com/vmware/govmomi/vim25/mo/reference.go",
        "github.com/vmware/govmomi/vim25/mo/registry.go",
        "github.com/vmware/govmomi/vim25/mo/retrieve.go",
        "github.com/vmware/govmomi/vim25/mo/type_info.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/vim25/methods",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/progress",
    srcs = [
        "github.com/vmware/govmomi/vim25/progress/aggregator.go",
        "github.com/vmware/govmomi/vim25/progress/doc.go",
        "github.com/vmware/govmomi/vim25/progress/prefix.go",
        "github.com/vmware/govmomi/vim25/progress/reader.go",
        "github.com/vmware/govmomi/vim25/progress/report.go",
        "github.com/vmware/govmomi/vim25/progress/scale.go",
        "github.com/vmware/govmomi/vim25/progress/sinker.go",
        "github.com/vmware/govmomi/vim25/progress/tee.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/soap",
    srcs = [
        "github.com/vmware/govmomi/vim25/soap/client.go",
        "github.com/vmware/govmomi/vim25/soap/debug.go",
        "github.com/vmware/govmomi/vim25/soap/error.go",
        "github.com/vmware/govmomi/vim25/soap/soap.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/vmware/govmomi/vim25/debug",
        "//vendor:github.com/vmware/govmomi/vim25/progress",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:github.com/vmware/govmomi/vim25/xml",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/types",
    srcs = [
        "github.com/vmware/govmomi/vim25/types/base.go",
        "github.com/vmware/govmomi/vim25/types/enum.go",
        "github.com/vmware/govmomi/vim25/types/fault.go",
        "github.com/vmware/govmomi/vim25/types/helpers.go",
        "github.com/vmware/govmomi/vim25/types/if.go",
        "github.com/vmware/govmomi/vim25/types/internal.go",
        "github.com/vmware/govmomi/vim25/types/registry.go",
        "github.com/vmware/govmomi/vim25/types/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vmware/govmomi/vim25/xml",
    srcs = [
        "github.com/vmware/govmomi/vim25/xml/extras.go",
        "github.com/vmware/govmomi/vim25/xml/marshal.go",
        "github.com/vmware/govmomi/vim25/xml/read.go",
        "github.com/vmware/govmomi/vim25/xml/typeinfo.go",
        "github.com/vmware/govmomi/vim25/xml/xml.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/vmware/photon-controller-go-sdk/photon",
    srcs = [
        "github.com/vmware/photon-controller-go-sdk/photon/apitypes.go",
        "github.com/vmware/photon-controller-go-sdk/photon/auth.go",
        "github.com/vmware/photon-controller-go-sdk/photon/availabilityzones.go",
        "github.com/vmware/photon-controller-go-sdk/photon/client.go",
        "github.com/vmware/photon-controller-go-sdk/photon/clusters.go",
        "github.com/vmware/photon-controller-go-sdk/photon/deployments.go",
        "github.com/vmware/photon-controller-go-sdk/photon/disks.go",
        "github.com/vmware/photon-controller-go-sdk/photon/flavors.go",
        "github.com/vmware/photon-controller-go-sdk/photon/hosts.go",
        "github.com/vmware/photon-controller-go-sdk/photon/images.go",
        "github.com/vmware/photon-controller-go-sdk/photon/info.go",
        "github.com/vmware/photon-controller-go-sdk/photon/networks.go",
        "github.com/vmware/photon-controller-go-sdk/photon/projects.go",
        "github.com/vmware/photon-controller-go-sdk/photon/resourcetickets.go",
        "github.com/vmware/photon-controller-go-sdk/photon/restclient.go",
        "github.com/vmware/photon-controller-go-sdk/photon/status.go",
        "github.com/vmware/photon-controller-go-sdk/photon/tasks.go",
        "github.com/vmware/photon-controller-go-sdk/photon/tenants.go",
        "github.com/vmware/photon-controller-go-sdk/photon/util.go",
        "github.com/vmware/photon-controller-go-sdk/photon/virtualnetworks.go",
        "github.com/vmware/photon-controller-go-sdk/photon/vms.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/vmware/photon-controller-go-sdk/photon/lightwave"],
)

go_library(
    name = "github.com/vmware/photon-controller-go-sdk/photon/lightwave",
    srcs = [
        "github.com/vmware/photon-controller-go-sdk/photon/lightwave/jwttoken.go",
        "github.com/vmware/photon-controller-go-sdk/photon/lightwave/oidcclient.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/xanzy/go-cloudstack/cloudstack",
    srcs = [
        "github.com/xanzy/go-cloudstack/cloudstack/APIDiscoveryService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AccountService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AddressService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AffinityGroupService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AlertService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AsyncjobService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AuthenticationService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/AutoScaleService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/BaremetalService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/CertificateService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/CloudIdentifierService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ClusterService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ConfigurationService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/DiskOfferingService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/DomainService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/EventService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/FirewallService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/GuestOSService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/HostService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/HypervisorService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ISOService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ImageStoreService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/InternalLBService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/LDAPService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/LimitService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/LoadBalancerService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NATService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NetworkACLService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NetworkDeviceService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NetworkOfferingService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NetworkService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NicService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/NiciraNVPService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/OvsElementService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/PodService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/PoolService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/PortableIPService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ProjectService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/QuotaService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/RegionService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ResourcemetadataService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ResourcetagsService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/RouterService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SSHService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SecurityGroupService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ServiceOfferingService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SnapshotService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/StoragePoolService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/StratosphereSSPService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SwiftService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SystemCapacityService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/SystemVMService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/TemplateService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/UCSService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/UsageService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/UserService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VLANService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VMGroupService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VPCService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VPNService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VirtualMachineService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/VolumeService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/ZoneService.go",
        "github.com/xanzy/go-cloudstack/cloudstack/cloudstack.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/xiang90/probing",
    srcs = [
        "github.com/xiang90/probing/prober.go",
        "github.com/xiang90/probing/server.go",
        "github.com/xiang90/probing/status.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "github.com/xyproto/simpleredis",
    srcs = ["github.com/xyproto/simpleredis/simpleredis.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/garyburd/redigo/redis"],
)

go_library(
    name = "go4.org/errorutil",
    srcs = ["go4.org/errorutil/highlight.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/crypto/bcrypt",
    srcs = [
        "golang.org/x/crypto/bcrypt/base64.go",
        "golang.org/x/crypto/bcrypt/bcrypt.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/crypto/blowfish"],
)

go_library(
    name = "golang.org/x/crypto/blowfish",
    srcs = [
        "golang.org/x/crypto/blowfish/block.go",
        "golang.org/x/crypto/blowfish/cipher.go",
        "golang.org/x/crypto/blowfish/const.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/crypto/curve25519",
    srcs = [
        "golang.org/x/crypto/curve25519/const_amd64.s",
        "golang.org/x/crypto/curve25519/cswap_amd64.s",
        "golang.org/x/crypto/curve25519/doc.go",
        "golang.org/x/crypto/curve25519/freeze_amd64.s",
        "golang.org/x/crypto/curve25519/ladderstep_amd64.s",
        "golang.org/x/crypto/curve25519/mont25519_amd64.go",
        "golang.org/x/crypto/curve25519/mul_amd64.s",
        "golang.org/x/crypto/curve25519/square_amd64.s",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/crypto/ed25519",
    srcs = ["golang.org/x/crypto/ed25519/ed25519.go"],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/crypto/ed25519/internal/edwards25519"],
)

go_library(
    name = "golang.org/x/crypto/ed25519/internal/edwards25519",
    srcs = [
        "golang.org/x/crypto/ed25519/internal/edwards25519/const.go",
        "golang.org/x/crypto/ed25519/internal/edwards25519/edwards25519.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/crypto/pkcs12",
    srcs = [
        "golang.org/x/crypto/pkcs12/bmp-string.go",
        "golang.org/x/crypto/pkcs12/crypto.go",
        "golang.org/x/crypto/pkcs12/errors.go",
        "golang.org/x/crypto/pkcs12/mac.go",
        "golang.org/x/crypto/pkcs12/pbkdf.go",
        "golang.org/x/crypto/pkcs12/pkcs12.go",
        "golang.org/x/crypto/pkcs12/safebags.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/crypto/pkcs12/internal/rc2"],
)

go_library(
    name = "golang.org/x/crypto/pkcs12/internal/rc2",
    srcs = ["golang.org/x/crypto/pkcs12/internal/rc2/rc2.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/crypto/ssh",
    srcs = [
        "golang.org/x/crypto/ssh/buffer.go",
        "golang.org/x/crypto/ssh/certs.go",
        "golang.org/x/crypto/ssh/channel.go",
        "golang.org/x/crypto/ssh/cipher.go",
        "golang.org/x/crypto/ssh/client.go",
        "golang.org/x/crypto/ssh/client_auth.go",
        "golang.org/x/crypto/ssh/common.go",
        "golang.org/x/crypto/ssh/connection.go",
        "golang.org/x/crypto/ssh/doc.go",
        "golang.org/x/crypto/ssh/handshake.go",
        "golang.org/x/crypto/ssh/kex.go",
        "golang.org/x/crypto/ssh/keys.go",
        "golang.org/x/crypto/ssh/mac.go",
        "golang.org/x/crypto/ssh/messages.go",
        "golang.org/x/crypto/ssh/mux.go",
        "golang.org/x/crypto/ssh/server.go",
        "golang.org/x/crypto/ssh/session.go",
        "golang.org/x/crypto/ssh/tcpip.go",
        "golang.org/x/crypto/ssh/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/crypto/curve25519",
        "//vendor:golang.org/x/crypto/ed25519",
    ],
)

go_library(
    name = "golang.org/x/crypto/ssh/terminal",
    srcs = [
        "golang.org/x/crypto/ssh/terminal/terminal.go",
        "golang.org/x/crypto/ssh/terminal/util.go",
        "golang.org/x/crypto/ssh/terminal/util_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/exp/inotify",
    srcs = ["golang.org/x/exp/inotify/inotify_linux.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/context",
    srcs = [
        "golang.org/x/net/context/context.go",
        "golang.org/x/net/context/go17.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/context/ctxhttp",
    srcs = ["golang.org/x/net/context/ctxhttp/ctxhttp.go"],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "golang.org/x/net/html",
    srcs = [
        "golang.org/x/net/html/const.go",
        "golang.org/x/net/html/doc.go",
        "golang.org/x/net/html/doctype.go",
        "golang.org/x/net/html/entity.go",
        "golang.org/x/net/html/escape.go",
        "golang.org/x/net/html/foreign.go",
        "golang.org/x/net/html/node.go",
        "golang.org/x/net/html/parse.go",
        "golang.org/x/net/html/render.go",
        "golang.org/x/net/html/token.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/html/atom"],
)

go_library(
    name = "golang.org/x/net/html/atom",
    srcs = [
        "golang.org/x/net/html/atom/atom.go",
        "golang.org/x/net/html/atom/table.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/http2",
    srcs = [
        "golang.org/x/net/http2/client_conn_pool.go",
        "golang.org/x/net/http2/configure_transport.go",
        "golang.org/x/net/http2/errors.go",
        "golang.org/x/net/http2/fixed_buffer.go",
        "golang.org/x/net/http2/flow.go",
        "golang.org/x/net/http2/frame.go",
        "golang.org/x/net/http2/go16.go",
        "golang.org/x/net/http2/go17.go",
        "golang.org/x/net/http2/gotrack.go",
        "golang.org/x/net/http2/headermap.go",
        "golang.org/x/net/http2/http2.go",
        "golang.org/x/net/http2/pipe.go",
        "golang.org/x/net/http2/server.go",
        "golang.org/x/net/http2/transport.go",
        "golang.org/x/net/http2/write.go",
        "golang.org/x/net/http2/writesched.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/http2/hpack",
        "//vendor:golang.org/x/net/lex/httplex",
    ],
)

go_library(
    name = "golang.org/x/net/http2/hpack",
    srcs = [
        "golang.org/x/net/http2/hpack/encode.go",
        "golang.org/x/net/http2/hpack/hpack.go",
        "golang.org/x/net/http2/hpack/huffman.go",
        "golang.org/x/net/http2/hpack/tables.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/idna",
    srcs = [
        "golang.org/x/net/idna/idna.go",
        "golang.org/x/net/idna/punycode.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/internal/timeseries",
    srcs = ["golang.org/x/net/internal/timeseries/timeseries.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/lex/httplex",
    srcs = ["golang.org/x/net/lex/httplex/httplex.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/proxy",
    srcs = [
        "golang.org/x/net/proxy/direct.go",
        "golang.org/x/net/proxy/per_host.go",
        "golang.org/x/net/proxy/proxy.go",
        "golang.org/x/net/proxy/socks5.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/net/trace",
    srcs = [
        "golang.org/x/net/trace/events.go",
        "golang.org/x/net/trace/histogram.go",
        "golang.org/x/net/trace/trace.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/internal/timeseries",
    ],
)

go_library(
    name = "golang.org/x/net/websocket",
    srcs = [
        "golang.org/x/net/websocket/client.go",
        "golang.org/x/net/websocket/hybi.go",
        "golang.org/x/net/websocket/server.go",
        "golang.org/x/net/websocket/websocket.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/oauth2",
    srcs = [
        "golang.org/x/oauth2/oauth2.go",
        "golang.org/x/oauth2/token.go",
        "golang.org/x/oauth2/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2/internal",
    ],
)

go_library(
    name = "golang.org/x/oauth2/google",
    srcs = [
        "golang.org/x/oauth2/google/appengine.go",
        "golang.org/x/oauth2/google/default.go",
        "golang.org/x/oauth2/google/google.go",
        "golang.org/x/oauth2/google/jwt.go",
        "golang.org/x/oauth2/google/sdk.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/internal",
        "//vendor:golang.org/x/oauth2/jws",
        "//vendor:golang.org/x/oauth2/jwt",
    ],
)

go_library(
    name = "golang.org/x/oauth2/internal",
    srcs = [
        "golang.org/x/oauth2/internal/oauth2.go",
        "golang.org/x/oauth2/internal/token.go",
        "golang.org/x/oauth2/internal/transport.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "golang.org/x/oauth2/jws",
    srcs = ["golang.org/x/oauth2/jws/jws.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/oauth2/jwt",
    srcs = ["golang.org/x/oauth2/jwt/jwt.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/internal",
        "//vendor:golang.org/x/oauth2/jws",
    ],
)

go_library(
    name = "golang.org/x/sys/unix",
    srcs = [
        "golang.org/x/sys/unix/asm.s",
        "golang.org/x/sys/unix/asm_linux_amd64.s",
        "golang.org/x/sys/unix/bluetooth_linux.go",
        "golang.org/x/sys/unix/constants.go",
        "golang.org/x/sys/unix/env_unix.go",
        "golang.org/x/sys/unix/env_unset.go",
        "golang.org/x/sys/unix/flock.go",
        "golang.org/x/sys/unix/race0.go",
        "golang.org/x/sys/unix/sockcmsg_linux.go",
        "golang.org/x/sys/unix/sockcmsg_unix.go",
        "golang.org/x/sys/unix/str.go",
        "golang.org/x/sys/unix/syscall.go",
        "golang.org/x/sys/unix/syscall_linux.go",
        "golang.org/x/sys/unix/syscall_linux_amd64.go",
        "golang.org/x/sys/unix/syscall_unix.go",
        "golang.org/x/sys/unix/zerrors_linux_amd64.go",
        "golang.org/x/sys/unix/zsyscall_linux_amd64.go",
        "golang.org/x/sys/unix/zsysnum_linux_amd64.go",
        "golang.org/x/sys/unix/ztypes_linux_amd64.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/cases",
    srcs = [
        "golang.org/x/text/cases/cases.go",
        "golang.org/x/text/cases/context.go",
        "golang.org/x/text/cases/fold.go",
        "golang.org/x/text/cases/info.go",
        "golang.org/x/text/cases/map.go",
        "golang.org/x/text/cases/tables.go",
        "golang.org/x/text/cases/trieval.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/language",
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/norm",
    ],
)

go_library(
    name = "golang.org/x/text/encoding",
    srcs = ["golang.org/x/text/encoding/encoding.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/encoding/internal/identifier",
        "//vendor:golang.org/x/text/transform",
    ],
)

go_library(
    name = "golang.org/x/text/encoding/internal",
    srcs = ["golang.org/x/text/encoding/internal/internal.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/encoding",
        "//vendor:golang.org/x/text/encoding/internal/identifier",
        "//vendor:golang.org/x/text/transform",
    ],
)

go_library(
    name = "golang.org/x/text/encoding/internal/identifier",
    srcs = [
        "golang.org/x/text/encoding/internal/identifier/identifier.go",
        "golang.org/x/text/encoding/internal/identifier/mib.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/encoding/unicode",
    srcs = [
        "golang.org/x/text/encoding/unicode/override.go",
        "golang.org/x/text/encoding/unicode/unicode.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/encoding",
        "//vendor:golang.org/x/text/encoding/internal",
        "//vendor:golang.org/x/text/encoding/internal/identifier",
        "//vendor:golang.org/x/text/internal/utf8internal",
        "//vendor:golang.org/x/text/runes",
        "//vendor:golang.org/x/text/transform",
    ],
)

go_library(
    name = "golang.org/x/text/internal/tag",
    srcs = ["golang.org/x/text/internal/tag/tag.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/internal/utf8internal",
    srcs = ["golang.org/x/text/internal/utf8internal/utf8internal.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/language",
    srcs = [
        "golang.org/x/text/language/common.go",
        "golang.org/x/text/language/coverage.go",
        "golang.org/x/text/language/go1_2.go",
        "golang.org/x/text/language/index.go",
        "golang.org/x/text/language/language.go",
        "golang.org/x/text/language/lookup.go",
        "golang.org/x/text/language/match.go",
        "golang.org/x/text/language/parse.go",
        "golang.org/x/text/language/tables.go",
        "golang.org/x/text/language/tags.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/internal/tag"],
)

go_library(
    name = "golang.org/x/text/runes",
    srcs = [
        "golang.org/x/text/runes/cond.go",
        "golang.org/x/text/runes/runes.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "golang.org/x/text/secure/bidirule",
    srcs = ["golang.org/x/text/secure/bidirule/bidirule.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/bidi",
    ],
)

go_library(
    name = "golang.org/x/text/secure/precis",
    srcs = [
        "golang.org/x/text/secure/precis/class.go",
        "golang.org/x/text/secure/precis/context.go",
        "golang.org/x/text/secure/precis/doc.go",
        "golang.org/x/text/secure/precis/nickname.go",
        "golang.org/x/text/secure/precis/options.go",
        "golang.org/x/text/secure/precis/profile.go",
        "golang.org/x/text/secure/precis/profiles.go",
        "golang.org/x/text/secure/precis/tables.go",
        "golang.org/x/text/secure/precis/transformer.go",
        "golang.org/x/text/secure/precis/trieval.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/cases",
        "//vendor:golang.org/x/text/runes",
        "//vendor:golang.org/x/text/secure/bidirule",
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/norm",
        "//vendor:golang.org/x/text/width",
    ],
)

go_library(
    name = "golang.org/x/text/transform",
    srcs = ["golang.org/x/text/transform/transform.go"],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/unicode/bidi",
    srcs = [
        "golang.org/x/text/unicode/bidi/bidi.go",
        "golang.org/x/text/unicode/bidi/bracket.go",
        "golang.org/x/text/unicode/bidi/core.go",
        "golang.org/x/text/unicode/bidi/prop.go",
        "golang.org/x/text/unicode/bidi/tables.go",
        "golang.org/x/text/unicode/bidi/trieval.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "golang.org/x/text/unicode/norm",
    srcs = [
        "golang.org/x/text/unicode/norm/composition.go",
        "golang.org/x/text/unicode/norm/forminfo.go",
        "golang.org/x/text/unicode/norm/input.go",
        "golang.org/x/text/unicode/norm/iter.go",
        "golang.org/x/text/unicode/norm/normalize.go",
        "golang.org/x/text/unicode/norm/readwriter.go",
        "golang.org/x/text/unicode/norm/tables.go",
        "golang.org/x/text/unicode/norm/transform.go",
        "golang.org/x/text/unicode/norm/trie.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "golang.org/x/text/width",
    srcs = [
        "golang.org/x/text/width/kind_string.go",
        "golang.org/x/text/width/tables.go",
        "golang.org/x/text/width/transform.go",
        "golang.org/x/text/width/trieval.go",
        "golang.org/x/text/width/width.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "google.golang.org/api/cloudmonitoring/v2beta2",
    srcs = ["google.golang.org/api/cloudmonitoring/v2beta2/cloudmonitoring-gen.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
        "//vendor:google.golang.org/api/gensupport",
        "//vendor:google.golang.org/api/googleapi",
    ],
)

go_library(
    name = "google.golang.org/api/compute/v1",
    srcs = ["google.golang.org/api/compute/v1/compute-gen.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
        "//vendor:google.golang.org/api/gensupport",
        "//vendor:google.golang.org/api/googleapi",
    ],
)

go_library(
    name = "google.golang.org/api/container/v1",
    srcs = ["google.golang.org/api/container/v1/container-gen.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
        "//vendor:google.golang.org/api/gensupport",
        "//vendor:google.golang.org/api/googleapi",
    ],
)

go_library(
    name = "google.golang.org/api/dns/v1",
    srcs = ["google.golang.org/api/dns/v1/dns-gen.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
        "//vendor:google.golang.org/api/gensupport",
        "//vendor:google.golang.org/api/googleapi",
    ],
)

go_library(
    name = "google.golang.org/api/gensupport",
    srcs = [
        "google.golang.org/api/gensupport/backoff.go",
        "google.golang.org/api/gensupport/buffer.go",
        "google.golang.org/api/gensupport/doc.go",
        "google.golang.org/api/gensupport/json.go",
        "google.golang.org/api/gensupport/media.go",
        "google.golang.org/api/gensupport/params.go",
        "google.golang.org/api/gensupport/resumable.go",
        "google.golang.org/api/gensupport/retry.go",
        "google.golang.org/api/gensupport/send.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
        "//vendor:google.golang.org/api/googleapi",
    ],
)

go_library(
    name = "google.golang.org/api/googleapi",
    srcs = [
        "google.golang.org/api/googleapi/googleapi.go",
        "google.golang.org/api/googleapi/types.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:google.golang.org/api/googleapi/internal/uritemplates"],
)

go_library(
    name = "google.golang.org/api/googleapi/internal/uritemplates",
    srcs = [
        "google.golang.org/api/googleapi/internal/uritemplates/uritemplates.go",
        "google.golang.org/api/googleapi/internal/uritemplates/utils.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "google.golang.org/appengine",
    srcs = [
        "google.golang.org/appengine/appengine.go",
        "google.golang.org/appengine/appengine_vm.go",
        "google.golang.org/appengine/errors.go",
        "google.golang.org/appengine/identity.go",
        "google.golang.org/appengine/namespace.go",
        "google.golang.org/appengine/timeout.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/appengine/internal",
        "//vendor:google.golang.org/appengine/internal/app_identity",
        "//vendor:google.golang.org/appengine/internal/modules",
    ],
)

go_library(
    name = "google.golang.org/appengine/internal",
    srcs = [
        "google.golang.org/appengine/internal/api.go",
        "google.golang.org/appengine/internal/api_common.go",
        "google.golang.org/appengine/internal/app_id.go",
        "google.golang.org/appengine/internal/identity.go",
        "google.golang.org/appengine/internal/identity_vm.go",
        "google.golang.org/appengine/internal/internal.go",
        "google.golang.org/appengine/internal/main_vm.go",
        "google.golang.org/appengine/internal/metadata.go",
        "google.golang.org/appengine/internal/net.go",
        "google.golang.org/appengine/internal/transaction.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/appengine/internal/base",
        "//vendor:google.golang.org/appengine/internal/datastore",
        "//vendor:google.golang.org/appengine/internal/log",
        "//vendor:google.golang.org/appengine/internal/remote_api",
    ],
)

go_library(
    name = "google.golang.org/appengine/internal/app_identity",
    srcs = ["google.golang.org/appengine/internal/app_identity/app_identity_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/appengine/internal/base",
    srcs = ["google.golang.org/appengine/internal/base/api_base.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/appengine/internal/datastore",
    srcs = ["google.golang.org/appengine/internal/datastore/datastore_v3.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/appengine/internal/log",
    srcs = ["google.golang.org/appengine/internal/log/log_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/appengine/internal/modules",
    srcs = ["google.golang.org/appengine/internal/modules/modules_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/appengine/internal/remote_api",
    srcs = ["google.golang.org/appengine/internal/remote_api/remote_api.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "google.golang.org/grpc",
    srcs = [
        "google.golang.org/grpc/backoff.go",
        "google.golang.org/grpc/balancer.go",
        "google.golang.org/grpc/call.go",
        "google.golang.org/grpc/clientconn.go",
        "google.golang.org/grpc/doc.go",
        "google.golang.org/grpc/interceptor.go",
        "google.golang.org/grpc/rpc_util.go",
        "google.golang.org/grpc/server.go",
        "google.golang.org/grpc/stream.go",
        "google.golang.org/grpc/trace.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/http2",
        "//vendor:golang.org/x/net/trace",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:google.golang.org/grpc/credentials",
        "//vendor:google.golang.org/grpc/grpclog",
        "//vendor:google.golang.org/grpc/internal",
        "//vendor:google.golang.org/grpc/metadata",
        "//vendor:google.golang.org/grpc/naming",
        "//vendor:google.golang.org/grpc/transport",
    ],
)

go_library(
    name = "google.golang.org/grpc/codes",
    srcs = [
        "google.golang.org/grpc/codes/code_string.go",
        "google.golang.org/grpc/codes/codes.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "google.golang.org/grpc/credentials",
    srcs = [
        "google.golang.org/grpc/credentials/credentials.go",
        "google.golang.org/grpc/credentials/credentials_util_go17.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "google.golang.org/grpc/grpclog",
    srcs = ["google.golang.org/grpc/grpclog/logger.go"],
    tags = ["automanaged"],
)

go_library(
    name = "google.golang.org/grpc/internal",
    srcs = ["google.golang.org/grpc/internal/internal.go"],
    tags = ["automanaged"],
)

go_library(
    name = "google.golang.org/grpc/metadata",
    srcs = ["google.golang.org/grpc/metadata/metadata.go"],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "google.golang.org/grpc/naming",
    srcs = ["google.golang.org/grpc/naming/naming.go"],
    tags = ["automanaged"],
)

go_library(
    name = "google.golang.org/grpc/peer",
    srcs = ["google.golang.org/grpc/peer/peer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc/credentials",
    ],
)

go_library(
    name = "google.golang.org/grpc/transport",
    srcs = [
        "google.golang.org/grpc/transport/control.go",
        "google.golang.org/grpc/transport/go17.go",
        "google.golang.org/grpc/transport/handler_server.go",
        "google.golang.org/grpc/transport/http2_client.go",
        "google.golang.org/grpc/transport/http2_server.go",
        "google.golang.org/grpc/transport/http_util.go",
        "google.golang.org/grpc/transport/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/http2",
        "//vendor:golang.org/x/net/http2/hpack",
        "//vendor:golang.org/x/net/trace",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:google.golang.org/grpc/credentials",
        "//vendor:google.golang.org/grpc/grpclog",
        "//vendor:google.golang.org/grpc/metadata",
        "//vendor:google.golang.org/grpc/peer",
    ],
)

go_library(
    name = "gopkg.in/gcfg.v1",
    srcs = [
        "gopkg.in/gcfg.v1/doc.go",
        "gopkg.in/gcfg.v1/go1_2.go",
        "gopkg.in/gcfg.v1/read.go",
        "gopkg.in/gcfg.v1/set.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:gopkg.in/gcfg.v1/scanner",
        "//vendor:gopkg.in/gcfg.v1/token",
        "//vendor:gopkg.in/gcfg.v1/types",
    ],
)

go_library(
    name = "gopkg.in/gcfg.v1/scanner",
    srcs = [
        "gopkg.in/gcfg.v1/scanner/errors.go",
        "gopkg.in/gcfg.v1/scanner/scanner.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:gopkg.in/gcfg.v1/token"],
)

go_library(
    name = "gopkg.in/gcfg.v1/token",
    srcs = [
        "gopkg.in/gcfg.v1/token/position.go",
        "gopkg.in/gcfg.v1/token/serialize.go",
        "gopkg.in/gcfg.v1/token/token.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "gopkg.in/gcfg.v1/types",
    srcs = [
        "gopkg.in/gcfg.v1/types/bool.go",
        "gopkg.in/gcfg.v1/types/doc.go",
        "gopkg.in/gcfg.v1/types/enum.go",
        "gopkg.in/gcfg.v1/types/int.go",
        "gopkg.in/gcfg.v1/types/scan.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "gopkg.in/inf.v0",
    srcs = [
        "gopkg.in/inf.v0/dec.go",
        "gopkg.in/inf.v0/rounder.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "gopkg.in/natefinch/lumberjack.v2",
    srcs = [
        "gopkg.in/natefinch/lumberjack.v2/chown_linux.go",
        "gopkg.in/natefinch/lumberjack.v2/lumberjack.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "gopkg.in/yaml.v2",
    srcs = [
        "gopkg.in/yaml.v2/apic.go",
        "gopkg.in/yaml.v2/decode.go",
        "gopkg.in/yaml.v2/emitterc.go",
        "gopkg.in/yaml.v2/encode.go",
        "gopkg.in/yaml.v2/parserc.go",
        "gopkg.in/yaml.v2/readerc.go",
        "gopkg.in/yaml.v2/resolve.go",
        "gopkg.in/yaml.v2/scannerc.go",
        "gopkg.in/yaml.v2/sorter.go",
        "gopkg.in/yaml.v2/writerc.go",
        "gopkg.in/yaml.v2/yaml.go",
        "gopkg.in/yaml.v2/yamlh.go",
        "gopkg.in/yaml.v2/yamlprivateh.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/errors_test",
    srcs = ["k8s.io/apimachinery/pkg/api/errors/errors_test.go"],
    library = ":k8s.io/apimachinery/pkg/api/errors",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/errors",
    srcs = [
        "k8s.io/apimachinery/pkg/api/errors/doc.go",
        "k8s.io/apimachinery/pkg/api/errors/errors.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/meta_test",
    srcs = [
        "k8s.io/apimachinery/pkg/api/meta/multirestmapper_test.go",
        "k8s.io/apimachinery/pkg/api/meta/priority_test.go",
        "k8s.io/apimachinery/pkg/api/meta/restmapper_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/api/meta",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/meta",
    srcs = [
        "k8s.io/apimachinery/pkg/api/meta/default.go",
        "k8s.io/apimachinery/pkg/api/meta/doc.go",
        "k8s.io/apimachinery/pkg/api/meta/errors.go",
        "k8s.io/apimachinery/pkg/api/meta/firsthit_restmapper.go",
        "k8s.io/apimachinery/pkg/api/meta/help.go",
        "k8s.io/apimachinery/pkg/api/meta/interfaces.go",
        "k8s.io/apimachinery/pkg/api/meta/meta.go",
        "k8s.io/apimachinery/pkg/api/meta/multirestmapper.go",
        "k8s.io/apimachinery/pkg/api/meta/priority.go",
        "k8s.io/apimachinery/pkg/api/meta/restmapper.go",
        "k8s.io/apimachinery/pkg/api/meta/unstructured.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/validation",
    srcs = [
        "k8s.io/apimachinery/pkg/api/validation/doc.go",
        "k8s.io/apimachinery/pkg/api/validation/generic.go",
        "k8s.io/apimachinery/pkg/api/validation/objectmeta.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/validation",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apimachinery_test",
    srcs = ["k8s.io/apimachinery/pkg/apimachinery/types_test.go"],
    library = ":k8s.io/apimachinery/pkg/apimachinery",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/runtime/schema"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apimachinery",
    srcs = [
        "k8s.io/apimachinery/pkg/apimachinery/doc.go",
        "k8s.io/apimachinery/pkg/apimachinery/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apimachinery/registered_test",
    srcs = ["k8s.io/apimachinery/pkg/apimachinery/registered/registered_test.go"],
    library = ":k8s.io/apimachinery/pkg/apimachinery/registered",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apimachinery/registered",
    srcs = ["k8s.io/apimachinery/pkg/apimachinery/registered/registered.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apis/meta/v1_test",
    srcs = [
        "k8s.io/apimachinery/pkg/apis/meta/v1/duration_test.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/group_version_test.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/helpers_test.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/labels_test.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/time_test.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/types_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/apis/meta/v1",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/labels",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apis/meta/v1",
    srcs = [
        "k8s.io/apimachinery/pkg/apis/meta/v1/doc.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/duration.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/generated.pb.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/group_version.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/helpers.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/labels.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/meta.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/register.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/time.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/time_proto.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/types.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/types_swagger_doc_generated.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/watch.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/well_known_labels.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/zz_generated.deepcopy.go",
        "k8s.io/apimachinery/pkg/apis/meta/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    srcs = ["k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructured.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apis/meta/v1/validation_test",
    srcs = ["k8s.io/apimachinery/pkg/apis/meta/v1/validation/validation_test.go"],
    library = ":k8s.io/apimachinery/pkg/apis/meta/v1/validation",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/validation/field"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apis/meta/v1/validation",
    srcs = ["k8s.io/apimachinery/pkg/apis/meta/v1/validation/validation.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/conversion_test",
    srcs = [
        "k8s.io/apimachinery/pkg/conversion/converter_test.go",
        "k8s.io/apimachinery/pkg/conversion/deep_copy_test.go",
        "k8s.io/apimachinery/pkg/conversion/helper_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/conversion",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/conversion",
    srcs = [
        "k8s.io/apimachinery/pkg/conversion/cloner.go",
        "k8s.io/apimachinery/pkg/conversion/converter.go",
        "k8s.io/apimachinery/pkg/conversion/deep_equal.go",
        "k8s.io/apimachinery/pkg/conversion/doc.go",
        "k8s.io/apimachinery/pkg/conversion/helper.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/third_party/forked/golang/reflect"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/conversion/queryparams",
    srcs = [
        "k8s.io/apimachinery/pkg/conversion/queryparams/convert.go",
        "k8s.io/apimachinery/pkg/conversion/queryparams/doc.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/conversion/queryparams_xtest",
    srcs = ["k8s.io/apimachinery/pkg/conversion/queryparams/convert_test.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion/queryparams",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/labels_test",
    srcs = [
        "k8s.io/apimachinery/pkg/labels/labels_test.go",
        "k8s.io/apimachinery/pkg/labels/selector_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/labels",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/labels",
    srcs = [
        "k8s.io/apimachinery/pkg/labels/doc.go",
        "k8s.io/apimachinery/pkg/labels/labels.go",
        "k8s.io/apimachinery/pkg/labels/selector.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/openapi",
    srcs = [
        "k8s.io/apimachinery/pkg/openapi/common.go",
        "k8s.io/apimachinery/pkg/openapi/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/swagger_doc_generator_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/codec.go",
        "k8s.io/apimachinery/pkg/runtime/codec_check.go",
        "k8s.io/apimachinery/pkg/runtime/conversion.go",
        "k8s.io/apimachinery/pkg/runtime/doc.go",
        "k8s.io/apimachinery/pkg/runtime/embedded.go",
        "k8s.io/apimachinery/pkg/runtime/error.go",
        "k8s.io/apimachinery/pkg/runtime/extension.go",
        "k8s.io/apimachinery/pkg/runtime/generated.pb.go",
        "k8s.io/apimachinery/pkg/runtime/helper.go",
        "k8s.io/apimachinery/pkg/runtime/interfaces.go",
        "k8s.io/apimachinery/pkg/runtime/register.go",
        "k8s.io/apimachinery/pkg/runtime/scheme.go",
        "k8s.io/apimachinery/pkg/runtime/scheme_builder.go",
        "k8s.io/apimachinery/pkg/runtime/swagger_doc_generator.go",
        "k8s.io/apimachinery/pkg/runtime/types.go",
        "k8s.io/apimachinery/pkg/runtime/types_proto.go",
        "k8s.io/apimachinery/pkg/runtime/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/conversion/queryparams",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime_xtest",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/conversion_test.go",
        "k8s.io/apimachinery/pkg/runtime/embedded_test.go",
        "k8s.io/apimachinery/pkg/runtime/extension_test.go",
        "k8s.io/apimachinery/pkg/runtime/scheme_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/schema_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/schema/group_version_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime/schema",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/schema",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/schema/generated.pb.go",
        "k8s.io/apimachinery/pkg/runtime/schema/group_version.go",
        "k8s.io/apimachinery/pkg/runtime/schema/interfaces.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/gogo/protobuf/proto"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/codec_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime/serializer",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/serializer/codec_factory.go",
        "k8s.io/apimachinery/pkg/runtime/serializer/negotiated_codec.go",
        "k8s.io/apimachinery/pkg/runtime/serializer/protobuf_extension.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/json",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/protobuf",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/versioning",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/json_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/json/meta_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime/serializer/json",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/json",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/serializer/json/json.go",
        "k8s.io/apimachinery/pkg/runtime/serializer/json/meta.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/util/framer",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/json_xtest",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/json/json_test.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/json",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/protobuf",
    srcs = [
        "k8s.io/apimachinery/pkg/runtime/serializer/protobuf/doc.go",
        "k8s.io/apimachinery/pkg/runtime/serializer/protobuf/protobuf.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/util/framer",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/recognizer/recognizer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/recognizer/testing_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/recognizer/testing/recognizer_test.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/json",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/streaming_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/streaming/streaming_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime/serializer/streaming",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/framer",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/streaming",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/streaming/streaming.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/versioning_test",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/versioning/versioning_test.go"],
    library = ":k8s.io/apimachinery/pkg/runtime/serializer/versioning",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/versioning",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/versioning/versioning.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/runtime/serializer/yaml",
    srcs = ["k8s.io/apimachinery/pkg/runtime/serializer/yaml/yaml.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/selection",
    srcs = ["k8s.io/apimachinery/pkg/selection/operator.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/types",
    srcs = [
        "k8s.io/apimachinery/pkg/types/doc.go",
        "k8s.io/apimachinery/pkg/types/namespacedname.go",
        "k8s.io/apimachinery/pkg/types/nodename.go",
        "k8s.io/apimachinery/pkg/types/patch.go",
        "k8s.io/apimachinery/pkg/types/uid.go",
        "k8s.io/apimachinery/pkg/types/unix_user_id.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/diff_test",
    srcs = ["k8s.io/apimachinery/pkg/util/diff/diff_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/diff",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/diff",
    srcs = ["k8s.io/apimachinery/pkg/util/diff/diff.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/errors_test",
    srcs = ["k8s.io/apimachinery/pkg/util/errors/errors_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/errors",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/errors",
    srcs = [
        "k8s.io/apimachinery/pkg/util/errors/doc.go",
        "k8s.io/apimachinery/pkg/util/errors/errors.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/framer_test",
    srcs = ["k8s.io/apimachinery/pkg/util/framer/framer_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/framer",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/framer",
    srcs = ["k8s.io/apimachinery/pkg/util/framer/framer.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/json_test",
    srcs = ["k8s.io/apimachinery/pkg/util/json/json_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/json",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/json",
    srcs = ["k8s.io/apimachinery/pkg/util/json/json.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/net_test",
    srcs = [
        "k8s.io/apimachinery/pkg/util/net/http_test.go",
        "k8s.io/apimachinery/pkg/util/net/interface_test.go",
        "k8s.io/apimachinery/pkg/util/net/port_range_test.go",
        "k8s.io/apimachinery/pkg/util/net/port_split_test.go",
        "k8s.io/apimachinery/pkg/util/net/util_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/util/net",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/net",
    srcs = [
        "k8s.io/apimachinery/pkg/util/net/http.go",
        "k8s.io/apimachinery/pkg/util/net/interface.go",
        "k8s.io/apimachinery/pkg/util/net/port_range.go",
        "k8s.io/apimachinery/pkg/util/net/port_split.go",
        "k8s.io/apimachinery/pkg/util/net/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/http2",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/rand_test",
    srcs = ["k8s.io/apimachinery/pkg/util/rand/rand_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/rand",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/rand",
    srcs = ["k8s.io/apimachinery/pkg/util/rand/rand.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/runtime_test",
    srcs = ["k8s.io/apimachinery/pkg/util/runtime/runtime_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/runtime",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/runtime",
    srcs = ["k8s.io/apimachinery/pkg/util/runtime/runtime.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/sets_test",
    srcs = ["k8s.io/apimachinery/pkg/util/sets/set_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/sets",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/sets",
    srcs = [
        "k8s.io/apimachinery/pkg/util/sets/byte.go",
        "k8s.io/apimachinery/pkg/util/sets/doc.go",
        "k8s.io/apimachinery/pkg/util/sets/empty.go",
        "k8s.io/apimachinery/pkg/util/sets/int.go",
        "k8s.io/apimachinery/pkg/util/sets/int64.go",
        "k8s.io/apimachinery/pkg/util/sets/string.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/validation_test",
    srcs = ["k8s.io/apimachinery/pkg/util/validation/validation_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/validation",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/validation",
    srcs = ["k8s.io/apimachinery/pkg/util/validation/validation.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/validation/field_test",
    srcs = [
        "k8s.io/apimachinery/pkg/util/validation/field/errors_test.go",
        "k8s.io/apimachinery/pkg/util/validation/field/path_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/util/validation/field",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/validation/field",
    srcs = [
        "k8s.io/apimachinery/pkg/util/validation/field/errors.go",
        "k8s.io/apimachinery/pkg/util/validation/field/path.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/wait_test",
    srcs = ["k8s.io/apimachinery/pkg/util/wait/wait_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/wait",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/runtime"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/wait",
    srcs = [
        "k8s.io/apimachinery/pkg/util/wait/doc.go",
        "k8s.io/apimachinery/pkg/util/wait/wait.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/runtime"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/yaml_test",
    srcs = ["k8s.io/apimachinery/pkg/util/yaml/decoder_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/yaml",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/yaml",
    srcs = ["k8s.io/apimachinery/pkg/util/yaml/decoder.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/golang/glog",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/version",
    srcs = [
        "k8s.io/apimachinery/pkg/version/doc.go",
        "k8s.io/apimachinery/pkg/version/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/watch",
    srcs = [
        "k8s.io/apimachinery/pkg/watch/doc.go",
        "k8s.io/apimachinery/pkg/watch/filter.go",
        "k8s.io/apimachinery/pkg/watch/mux.go",
        "k8s.io/apimachinery/pkg/watch/streamwatcher.go",
        "k8s.io/apimachinery/pkg/watch/until.go",
        "k8s.io/apimachinery/pkg/watch/watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/watch_xtest",
    srcs = [
        "k8s.io/apimachinery/pkg/watch/filter_test.go",
        "k8s.io/apimachinery/pkg/watch/mux_test.go",
        "k8s.io/apimachinery/pkg/watch/streamwatcher_test.go",
        "k8s.io/apimachinery/pkg/watch/watch_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_test(
    name = "k8s.io/apimachinery/third_party/forked/golang/reflect_test",
    srcs = ["k8s.io/apimachinery/third_party/forked/golang/reflect/deep_equal_test.go"],
    library = ":k8s.io/apimachinery/third_party/forked/golang/reflect",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/third_party/forked/golang/reflect",
    srcs = [
        "k8s.io/apimachinery/third_party/forked/golang/reflect/deep_equal.go",
        "k8s.io/apimachinery/third_party/forked/golang/reflect/type.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/pkg/admission_test",
    srcs = ["k8s.io/apiserver/pkg/admission/chain_test.go"],
    library = ":k8s.io/apiserver/pkg/admission",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/runtime/schema"],
)

go_library(
    name = "k8s.io/apiserver/pkg/admission",
    srcs = [
        "k8s.io/apiserver/pkg/admission/attributes.go",
        "k8s.io/apiserver/pkg/admission/chain.go",
        "k8s.io/apiserver/pkg/admission/errors.go",
        "k8s.io/apiserver/pkg/admission/handler.go",
        "k8s.io/apiserver/pkg/admission/interfaces.go",
        "k8s.io/apiserver/pkg/admission/plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/authenticator",
    srcs = ["k8s.io/apiserver/pkg/authentication/authenticator/interfaces.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/group_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/group/group_adder_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/group",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/group",
    srcs = ["k8s.io/apiserver/pkg/authentication/group/group_adder.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/request/anonymous_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/anonymous/anonymous_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/request/anonymous",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/request/anonymous",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/anonymous/anonymous.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/request/bearertoken_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/bearertoken/bearertoken_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/request/bearertoken",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/request/bearertoken",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/bearertoken/bearertoken.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/request/headerrequest_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/headerrequest/requestheader_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/request/headerrequest",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/request/headerrequest",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/headerrequest/requestheader.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/x509",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/request/union_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/union/unionauth_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/request/union",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/request/union",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/union/union.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/request/x509_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/request/x509/x509_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/request/x509",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/request/x509",
    srcs = [
        "k8s.io/apiserver/pkg/authentication/request/x509/doc.go",
        "k8s.io/apiserver/pkg/authentication/request/x509/x509.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/serviceaccount_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/serviceaccount/util_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/serviceaccount",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/serviceaccount",
    srcs = ["k8s.io/apiserver/pkg/authentication/serviceaccount/util.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/api/validation"],
)

go_test(
    name = "k8s.io/apiserver/pkg/authentication/token/tokenfile_test",
    srcs = ["k8s.io/apiserver/pkg/authentication/token/tokenfile/tokenfile_test.go"],
    library = ":k8s.io/apiserver/pkg/authentication/token/tokenfile",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/token/tokenfile",
    srcs = ["k8s.io/apiserver/pkg/authentication/token/tokenfile/tokenfile.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/user",
    srcs = [
        "k8s.io/apiserver/pkg/authentication/user/doc.go",
        "k8s.io/apiserver/pkg/authentication/user/user.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authorization/authorizer",
    srcs = ["k8s.io/apiserver/pkg/authorization/authorizer/interfaces.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_test(
    name = "k8s.io/apiserver/pkg/authorization/union_test",
    srcs = ["k8s.io/apiserver/pkg/authorization/union/union_test.go"],
    library = ":k8s.io/apiserver/pkg/authorization/union",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authorization/authorizer"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authorization/union",
    srcs = ["k8s.io/apiserver/pkg/authorization/union/union.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/handlers/negotiation_test",
    srcs = ["k8s.io/apiserver/pkg/endpoints/handlers/negotiation/negotiate_test.go"],
    library = ":k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/handlers/negotiation/doc.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/negotiation/errors.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/negotiation/negotiate.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:bitbucket.org/ww/goautoneg",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/metrics",
    srcs = ["k8s.io/apiserver/pkg/endpoints/metrics/metrics.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/request_test",
    srcs = ["k8s.io/apiserver/pkg/endpoints/request/requestinfo_test.go"],
    library = ":k8s.io/apiserver/pkg/endpoints/request",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/sets"],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/request",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/request/context.go",
        "k8s.io/apiserver/pkg/endpoints/request/doc.go",
        "k8s.io/apiserver/pkg/endpoints/request/requestcontext.go",
        "k8s.io/apiserver/pkg/endpoints/request/requestinfo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/request_xtest",
    srcs = ["k8s.io/apiserver/pkg/endpoints/request/context_test.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/server",
    srcs = [
        "k8s.io/apiserver/pkg/server/config.go",
        "k8s.io/apiserver/pkg/server/config_selfclient.go",
        "k8s.io/apiserver/pkg/server/discovery.go",
        "k8s.io/apiserver/pkg/server/doc.go",
        "k8s.io/apiserver/pkg/server/genericapiserver.go",
        "k8s.io/apiserver/pkg/server/healthz.go",
        "k8s.io/apiserver/pkg/server/hooks.go",
        "k8s.io/apiserver/pkg/server/resource_config.go",
        "k8s.io/apiserver/pkg/server/resource_encoding_config.go",
        "k8s.io/apiserver/pkg/server/serve.go",
        "k8s.io/apiserver/pkg/server/storage_factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-systemd/daemon",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/pkg/errors",
        "//vendor:gopkg.in/natefinch/lumberjack.v2",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticatorfactory",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/union",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizerfactory",
        "//vendor:k8s.io/apiserver/pkg/authorization/union",
        "//vendor:k8s.io/apiserver/pkg/endpoints",
        "//vendor:k8s.io/apiserver/pkg/endpoints/filters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/openapi",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server/filters",
        "//vendor:k8s.io/apiserver/pkg/server/healthz",
        "//vendor:k8s.io/apiserver/pkg/server/mux",
        "//vendor:k8s.io/apiserver/pkg/server/options",
        "//vendor:k8s.io/apiserver/pkg/server/routes",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/server/healthz_test",
    srcs = ["k8s.io/apiserver/pkg/server/healthz/healthz_test.go"],
    library = ":k8s.io/apiserver/pkg/server/healthz",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/healthz",
    srcs = [
        "k8s.io/apiserver/pkg/server/healthz/doc.go",
        "k8s.io/apiserver/pkg/server/healthz/healthz.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/pkg/server/httplog_test",
    srcs = ["k8s.io/apiserver/pkg/server/httplog/log_test.go"],
    library = ":k8s.io/apiserver/pkg/server/httplog",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/httplog",
    srcs = [
        "k8s.io/apiserver/pkg/server/httplog/doc.go",
        "k8s.io/apiserver/pkg/server/httplog/log.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd/metrics",
    srcs = ["k8s.io/apiserver/pkg/storage/etcd/metrics/metrics.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/prometheus/client_golang/prometheus"],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/names_test",
    srcs = ["k8s.io/apiserver/pkg/storage/names/generate_test.go"],
    library = ":k8s.io/apiserver/pkg/storage/names",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/names",
    srcs = ["k8s.io/apiserver/pkg/storage/names/generate.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/rand"],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/cache_test",
    srcs = [
        "k8s.io/apiserver/pkg/util/cache/cache_test.go",
        "k8s.io/apiserver/pkg/util/cache/lruexpirecache_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/util/cache",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/groupcache/lru",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/cache",
    srcs = [
        "k8s.io/apiserver/pkg/util/cache/cache.go",
        "k8s.io/apiserver/pkg/util/cache/lruexpirecache.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/groupcache/lru"],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/flushwriter_test",
    srcs = ["k8s.io/apiserver/pkg/util/flushwriter/writer_test.go"],
    library = ":k8s.io/apiserver/pkg/util/flushwriter",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/flushwriter",
    srcs = [
        "k8s.io/apiserver/pkg/util/flushwriter/doc.go",
        "k8s.io/apiserver/pkg/util/flushwriter/writer.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/webhook",
    srcs = ["k8s.io/apiserver/pkg/util/webhook/webhook.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/wsstream_test",
    srcs = [
        "k8s.io/apiserver/pkg/util/wsstream/conn_test.go",
        "k8s.io/apiserver/pkg/util/wsstream/stream_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/util/wsstream",
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/websocket"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/wsstream",
    srcs = [
        "k8s.io/apiserver/pkg/util/wsstream/conn.go",
        "k8s.io/apiserver/pkg/util/wsstream/doc.go",
        "k8s.io/apiserver/pkg/util/wsstream/stream.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/cloud.google.com/go/compute/metadata",
    srcs = ["k8s.io/client-go/_vendor/cloud.google.com/go/compute/metadata/metadata.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:cloud.google.com/go/internal",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/net/context/ctxhttp",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/cloud.google.com/go/internal",
    srcs = ["k8s.io/client-go/_vendor/cloud.google.com/go/internal/cloud.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/PuerkitoBio/purell",
    srcs = ["k8s.io/client-go/_vendor/github.com/PuerkitoBio/purell/purell.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/PuerkitoBio/urlesc",
        "//vendor:golang.org/x/net/idna",
        "//vendor:golang.org/x/text/secure/precis",
        "//vendor:golang.org/x/text/unicode/norm",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/PuerkitoBio/urlesc",
    srcs = ["k8s.io/client-go/_vendor/github.com/PuerkitoBio/urlesc/urlesc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/http",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/http/client.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/http/http.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/http/url.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/claims.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/jose.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/jwk.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/jws.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/jwt.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/sig.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/sig_hmac.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/jose/sig_rsa.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key/key.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key/manager.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key/repo.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key/rotate.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/key/sync.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/pkg/health",
        "//vendor:github.com/coreos/pkg/timeutil",
        "//vendor:github.com/jonboulle/clockwork",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oauth2",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oauth2/error.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oauth2/oauth2.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/go-oidc/http"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/client.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/identity.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/interface.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/key.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/provider.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/transport.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/util.go",
        "k8s.io/client-go/_vendor/github.com/coreos/go-oidc/oidc/verification.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/http",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oauth2",
        "//vendor:github.com/coreos/pkg/timeutil",
        "//vendor:github.com/jonboulle/clockwork",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/pkg/health",
    srcs = ["k8s.io/client-go/_vendor/github.com/coreos/pkg/health/health.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/pkg/httputil"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/pkg/httputil",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/coreos/pkg/httputil/cookie.go",
        "k8s.io/client-go/_vendor/github.com/coreos/pkg/httputil/json.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/coreos/pkg/timeutil",
    srcs = ["k8s.io/client-go/_vendor/github.com/coreos/pkg/timeutil/backoff.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/bypass.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/common.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/config.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/doc.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/dump.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/format.go",
        "k8s.io/client-go/_vendor/github.com/davecgh/go-spew/spew/spew.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/docker/distribution/digest",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/docker/distribution/digest/digest.go",
        "k8s.io/client-go/_vendor/github.com/docker/distribution/digest/digester.go",
        "k8s.io/client-go/_vendor/github.com/docker/distribution/digest/doc.go",
        "k8s.io/client-go/_vendor/github.com/docker/distribution/digest/set.go",
        "k8s.io/client-go/_vendor/github.com/docker/distribution/digest/verifiers.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/docker/distribution/reference",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/docker/distribution/reference/reference.go",
        "k8s.io/client-go/_vendor/github.com/docker/distribution/reference/regexp.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/distribution/digest"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/emicklei/go-restful",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/compress.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/compressor_cache.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/compressor_pools.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/compressors.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/constants.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/container.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/cors_filter.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/curly.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/curly_route.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/doc.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/entity_accessors.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/filter.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/jsr311.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/logger.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/mime.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/options_filter.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/parameter.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/path_expression.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/request.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/response.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/route.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/route_builder.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/router.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/service_error.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/web_service.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/web_service_container.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/emicklei/go-restful/log"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/log",
    srcs = ["k8s.io/client-go/_vendor/github.com/emicklei/go-restful/log/log.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/api_declaration_list.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/config.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/model_builder.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/model_list.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/model_property_ext.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/model_property_list.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/ordered_route_map.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/swagger.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/swagger_builder.go",
        "k8s.io/client-go/_vendor/github.com/emicklei/go-restful/swagger/swagger_webservice.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/log",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/ghodss/yaml",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/ghodss/yaml/fields.go",
        "k8s.io/client-go/_vendor/github.com/ghodss/yaml/yaml.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:gopkg.in/yaml.v2"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/go-openapi/jsonpointer",
    srcs = ["k8s.io/client-go/_vendor/github.com/go-openapi/jsonpointer/pointer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/go-openapi/swag"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/go-openapi/jsonreference",
    srcs = ["k8s.io/client-go/_vendor/github.com/go-openapi/jsonreference/reference.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/PuerkitoBio/purell",
        "//vendor:github.com/go-openapi/jsonpointer",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/go-openapi/spec",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/bindata.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/contact_info.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/expander.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/external_docs.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/header.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/info.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/items.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/license.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/operation.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/parameter.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/path_item.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/paths.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/ref.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/response.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/responses.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/schema.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/security_scheme.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/spec.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/swagger.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/tag.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/spec/xml_object.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/jsonpointer",
        "//vendor:github.com/go-openapi/jsonreference",
        "//vendor:github.com/go-openapi/swag",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/go-openapi/swag",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/convert.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/convert_types.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/json.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/loading.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/net.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/path.go",
        "k8s.io/client-go/_vendor/github.com/go-openapi/swag/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mailru/easyjson/jlexer",
        "//vendor:github.com/mailru/easyjson/jwriter",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/clone.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/decode.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/decode_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/encode.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/encode_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/equal.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/extensions.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/extensions_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/lib.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/lib_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/message_set.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/pointer_unsafe.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/pointer_unsafe_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/properties.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/properties_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/skip_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/text.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/text_gogo.go",
        "k8s.io/client-go/_vendor/github.com/gogo/protobuf/proto/text_parser.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/gogo/protobuf/sortkeys",
    srcs = ["k8s.io/client-go/_vendor/github.com/gogo/protobuf/sortkeys/sortkeys.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/golang/glog",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/golang/glog/glog.go",
        "k8s.io/client-go/_vendor/github.com/golang/glog/glog_file.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/golang/groupcache/lru",
    srcs = ["k8s.io/client-go/_vendor/github.com/golang/groupcache/lru/lru.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/clone.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/decode.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/encode.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/equal.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/extensions.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/lib.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/message_set.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/pointer_unsafe.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/properties.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/text.go",
        "k8s.io/client-go/_vendor/github.com/golang/protobuf/proto/text_parser.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/google/gofuzz",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/google/gofuzz/doc.go",
        "k8s.io/client-go/_vendor/github.com/google/gofuzz/fuzz.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/howeyc/gopass",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/howeyc/gopass/pass.go",
        "k8s.io/client-go/_vendor/github.com/howeyc/gopass/terminal.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/crypto/ssh/terminal"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/imdario/mergo",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/imdario/mergo/doc.go",
        "k8s.io/client-go/_vendor/github.com/imdario/mergo/map.go",
        "k8s.io/client-go/_vendor/github.com/imdario/mergo/merge.go",
        "k8s.io/client-go/_vendor/github.com/imdario/mergo/mergo.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/jonboulle/clockwork",
    srcs = ["k8s.io/client-go/_vendor/github.com/jonboulle/clockwork/clockwork.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/juju/ratelimit",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/juju/ratelimit/ratelimit.go",
        "k8s.io/client-go/_vendor/github.com/juju/ratelimit/reader.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/mailru/easyjson/buffer",
    srcs = ["k8s.io/client-go/_vendor/github.com/mailru/easyjson/buffer/pool.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/mailru/easyjson/jlexer",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/mailru/easyjson/jlexer/error.go",
        "k8s.io/client-go/_vendor/github.com/mailru/easyjson/jlexer/lexer.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/mailru/easyjson/jwriter",
    srcs = ["k8s.io/client-go/_vendor/github.com/mailru/easyjson/jwriter/writer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/mailru/easyjson/buffer"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/pmezard/go-difflib/difflib",
    srcs = ["k8s.io/client-go/_vendor/github.com/pmezard/go-difflib/difflib/difflib.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/spf13/pflag",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/bool.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/count.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/duration.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/flag.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/float32.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/float64.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/golangflag.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/int.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/int32.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/int64.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/int8.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/int_slice.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/ip.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/ipmask.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/ipnet.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/string.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/string_array.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/string_slice.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/uint.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/uint16.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/uint32.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/uint64.go",
        "k8s.io/client-go/_vendor/github.com/spf13/pflag/uint8.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert/assertions.go",
        "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert/doc.go",
        "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert/errors.go",
        "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert/forward_assertions.go",
        "k8s.io/client-go/_vendor/github.com/stretchr/testify/assert/http_assertions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/pmezard/go-difflib/difflib",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/github.com/ugorji/go/codec",
    srcs = [
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/0doc.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/binc.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/cbor.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/decode.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/decode_go.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/encode.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/fast-path.generated.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/gen.generated.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/gen.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/gen-helper.generated.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/gen_16.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/gen_17.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/helper.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/helper_internal.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/helper_not_unsafe.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/json.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/msgpack.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/noop.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/prebuild.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/rpc.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/simple.go",
        "k8s.io/client-go/_vendor/github.com/ugorji/go/codec/time.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/crypto/ssh/terminal",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/crypto/ssh/terminal/terminal.go",
        "k8s.io/client-go/_vendor/golang.org/x/crypto/ssh/terminal/util.go",
        "k8s.io/client-go/_vendor/golang.org/x/crypto/ssh/terminal/util_linux.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/context",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/net/context/context.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/context/go17.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/context/ctxhttp",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/net/context/ctxhttp/ctxhttp.go"],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/http2",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/client_conn_pool.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/configure_transport.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/errors.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/fixed_buffer.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/flow.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/frame.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/go16.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/go17.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/gotrack.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/headermap.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/http2.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/pipe.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/server.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/transport.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/write.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/writesched.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/http2/hpack",
        "//vendor:golang.org/x/net/lex/httplex",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/http2/hpack",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/hpack/encode.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/hpack/hpack.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/hpack/huffman.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/http2/hpack/tables.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/idna",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/net/idna/idna.go",
        "k8s.io/client-go/_vendor/golang.org/x/net/idna/punycode.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/net/lex/httplex",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/net/lex/httplex/httplex.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/oauth2",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/oauth2.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/token.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2/internal",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/oauth2/google",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/google/appengine.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/google/default.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/google/google.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/google/jwt.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/google/sdk.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/internal",
        "//vendor:golang.org/x/oauth2/jws",
        "//vendor:golang.org/x/oauth2/jwt",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/oauth2/internal",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/internal/oauth2.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/internal/token.go",
        "k8s.io/client-go/_vendor/golang.org/x/oauth2/internal/transport.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/net/context"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/oauth2/jws",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/oauth2/jws/jws.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/oauth2/jwt",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/oauth2/jwt/jwt.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/internal",
        "//vendor:golang.org/x/oauth2/jws",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/sys/unix",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/asm.s",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/asm_linux_amd64.s",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/bluetooth_linux.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/constants.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/env_unix.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/env_unset.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/flock.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/race0.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/sockcmsg_linux.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/sockcmsg_unix.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/str.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/syscall.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/syscall_linux.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/syscall_linux_amd64.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/syscall_unix.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/zerrors_linux_amd64.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/zsyscall_linux_amd64.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/zsysnum_linux_amd64.go",
        "k8s.io/client-go/_vendor/golang.org/x/sys/unix/ztypes_linux_amd64.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/cases",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/cases.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/context.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/fold.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/info.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/map.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/cases/trieval.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/language",
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/norm",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/internal/tag",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/text/internal/tag/tag.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/language",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/language/common.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/coverage.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/go1_2.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/index.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/language.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/lookup.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/match.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/parse.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/language/tags.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/internal/tag"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/runes",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/runes/cond.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/runes/runes.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/secure/bidirule",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/text/secure/bidirule/bidirule.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/bidi",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/class.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/context.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/doc.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/nickname.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/options.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/profile.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/profiles.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/transformer.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/secure/precis/trieval.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/text/cases",
        "//vendor:golang.org/x/text/runes",
        "//vendor:golang.org/x/text/secure/bidirule",
        "//vendor:golang.org/x/text/transform",
        "//vendor:golang.org/x/text/unicode/norm",
        "//vendor:golang.org/x/text/width",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/transform",
    srcs = ["k8s.io/client-go/_vendor/golang.org/x/text/transform/transform.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/bidi.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/bracket.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/core.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/prop.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/bidi/trieval.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/composition.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/forminfo.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/input.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/iter.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/normalize.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/readwriter.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/transform.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/unicode/norm/trie.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "k8s.io/client-go/_vendor/golang.org/x/text/width",
    srcs = [
        "k8s.io/client-go/_vendor/golang.org/x/text/width/kind_string.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/width/tables.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/width/transform.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/width/trieval.go",
        "k8s.io/client-go/_vendor/golang.org/x/text/width/width.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/text/transform"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine",
    srcs = [
        "k8s.io/client-go/_vendor/google.golang.org/appengine/appengine.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/appengine_vm.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/errors.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/identity.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/namespace.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/timeout.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/appengine/internal",
        "//vendor:google.golang.org/appengine/internal/app_identity",
        "//vendor:google.golang.org/appengine/internal/modules",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal",
    srcs = [
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/api.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/api_common.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/app_id.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/identity.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/identity_vm.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/internal.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/main_vm.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/metadata.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/net.go",
        "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/transaction.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/appengine/internal/base",
        "//vendor:google.golang.org/appengine/internal/datastore",
        "//vendor:google.golang.org/appengine/internal/log",
        "//vendor:google.golang.org/appengine/internal/remote_api",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/app_identity",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/app_identity/app_identity_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/base",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/base/api_base.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/datastore",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/datastore/datastore_v3.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/log",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/log/log_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/modules",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/modules/modules_service.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/google.golang.org/appengine/internal/remote_api",
    srcs = ["k8s.io/client-go/_vendor/google.golang.org/appengine/internal/remote_api/remote_api.pb.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/gopkg.in/inf.v0",
    srcs = [
        "k8s.io/client-go/_vendor/gopkg.in/inf.v0/dec.go",
        "k8s.io/client-go/_vendor/gopkg.in/inf.v0/rounder.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/gopkg.in/yaml.v2",
    srcs = [
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/apic.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/decode.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/emitterc.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/encode.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/parserc.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/readerc.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/resolve.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/scannerc.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/sorter.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/writerc.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/yaml.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/yamlh.go",
        "k8s.io/client-go/_vendor/gopkg.in/yaml.v2/yamlprivateh.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/errors",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/errors/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/errors/errors.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/default.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/errors.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/firsthit_restmapper.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/help.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/interfaces.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/meta.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/multirestmapper.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/priority.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/restmapper.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/meta/unstructured.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/registered",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/registered/registered.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/duration.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/generated.pb.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/group_version.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/helpers.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/labels.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/meta.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/register.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/time.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/time_proto.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/types.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/types_swagger_doc_generated.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/watch.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/well_known_labels.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/zz_generated.deepcopy.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructured.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/cloner.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/converter.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/deep_equal.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/helper.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/third_party/forked/golang/reflect"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/queryparams",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/queryparams/convert.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/conversion/queryparams/doc.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/labels",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/labels/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/labels/labels.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/labels/selector.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/openapi",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/openapi/common.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/openapi/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/codec.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/codec_check.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/conversion.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/embedded.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/error.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/extension.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/generated.pb.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/helper.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/interfaces.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/register.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/scheme.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/scheme_builder.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/swagger_doc_generator.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/types.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/types_proto.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/conversion/queryparams",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/schema",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/schema/generated.pb.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/schema/group_version.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/schema/interfaces.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/gogo/protobuf/proto"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/codec_factory.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/negotiated_codec.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/protobuf_extension.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/json",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/protobuf",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/versioning",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/json",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/json/json.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/json/meta.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/util/framer",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/protobuf",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/protobuf/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/protobuf/protobuf.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/util/framer",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/recognizer/recognizer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/streaming",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/streaming/streaming.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/versioning",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/runtime/serializer/versioning/versioning.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/selection",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/selection/operator.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/namespacedname.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/nodename.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/patch.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/uid.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/types/unix_user_id.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/diff",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/diff/diff.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/errors",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/errors/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/errors/errors.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/framer",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/framer/framer.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/json",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/json/json.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net/http.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net/interface.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net/port_range.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net/port_split.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/net/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/http2",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/rand",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/rand/rand.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/runtime",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/runtime/runtime.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/byte.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/empty.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/int.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/int64.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/sets/string.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/validation",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/validation/validation.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/validation/field",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/validation/field/errors.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/validation/field/path.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/wait",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/wait/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/wait/wait.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/runtime"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/yaml",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/yaml/decoder.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/golang/glog",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/version",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/version/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/version/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/filter.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/mux.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/streamwatcher.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/until.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/watch/watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/third_party/forked/golang/reflect",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/third_party/forked/golang/reflect/deep_equal.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/third_party/forked/golang/reflect/type.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/discovery",
    srcs = [
        "k8s.io/client-go/discovery/discovery_client.go",
        "k8s.io/client-go/discovery/helper.go",
        "k8s.io/client-go/discovery/restmapper.go",
        "k8s.io/client-go/discovery/unstructured.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_test(
    name = "k8s.io/client-go/discovery_xtest",
    srcs = [
        "k8s.io/client-go/discovery/discovery_client_test.go",
        "k8s.io/client-go/discovery/helper_blackbox_test.go",
        "k8s.io/client-go/discovery/restmapper_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/rest/fake",
    ],
)

go_library(
    name = "k8s.io/client-go/discovery/fake",
    srcs = ["k8s.io/client-go/discovery/fake/discovery.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/version",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_test(
    name = "k8s.io/client-go/dynamic_test",
    srcs = [
        "k8s.io/client-go/dynamic/client_test.go",
        "k8s.io/client-go/dynamic/dynamic_util_test.go",
    ],
    library = ":k8s.io/client-go/dynamic",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/rest/watch",
    ],
)

go_library(
    name = "k8s.io/client-go/dynamic",
    srcs = [
        "k8s.io/client-go/dynamic/client.go",
        "k8s.io/client-go/dynamic/client_pool.go",
        "k8s.io/client-go/dynamic/dynamic_util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/conversion/queryparams",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes",
    srcs = [
        "k8s.io/client-go/kubernetes/clientset.go",
        "k8s.io/client-go/kubernetes/doc.go",
        "k8s.io/client-go/kubernetes/import_known_versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/kubernetes/typed/apps/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/autoscaling/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v2alpha1",
        "//vendor:k8s.io/client-go/kubernetes/typed/certificates/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/extensions/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/policy/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/storage/v1beta1",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/apis/apps/install",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/install",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/install",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling/install",
        "//vendor:k8s.io/client-go/pkg/apis/batch/install",
        "//vendor:k8s.io/client-go/pkg/apis/certificates/install",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/install",
        "//vendor:k8s.io/client-go/pkg/apis/policy/install",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/install",
        "//vendor:k8s.io/client-go/pkg/apis/storage/install",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/fake/clientset_generated.go",
        "k8s.io/client-go/kubernetes/fake/doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/discovery/fake",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/kubernetes/typed/apps/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/autoscaling/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v2alpha1",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/certificates/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/extensions/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/policy/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake",
        "//vendor:k8s.io/client-go/kubernetes/typed/storage/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/apps/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/apps_client.go",
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/statefulset.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/apps/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake/fake_apps_client.go",
        "k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake/fake_statefulset.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/apps/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/apps/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/authentication_client.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/tokenreview.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/tokenreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake/fake_authentication_client.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake/fake_tokenreview.go",
        "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake/fake_tokenreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/authorization_client.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/localsubjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/localsubjectaccessreview_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/selfsubjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/selfsubjectaccessreview_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/subjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/subjectaccessreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_authorization_client.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_localsubjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_localsubjectaccessreview_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_selfsubjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_selfsubjectaccessreview_expansion.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_subjectaccessreview.go",
        "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake/fake_subjectaccessreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/autoscaling/v1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/autoscaling_client.go",
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/doc.go",
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/horizontalpodautoscaler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling/v1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake/fake_autoscaling_client.go",
        "k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake/fake_horizontalpodautoscaler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/autoscaling/v1",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/batch/v1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/batch/v1/batch_client.go",
        "k8s.io/client-go/kubernetes/typed/batch/v1/doc.go",
        "k8s.io/client-go/kubernetes/typed/batch/v1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/batch/v1/job.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/batch/v1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/batch/v1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/batch/v1/fake/fake_batch_client.go",
        "k8s.io/client-go/kubernetes/typed/batch/v1/fake/fake_job.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v1",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/batch/v2alpha1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/batch_client.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/cronjob.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/doc.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/job.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v2alpha1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake/fake_batch_client.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake/fake_cronjob.go",
        "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake/fake_job.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/batch/v2alpha1",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v2alpha1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/core/v1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/core/v1/componentstatus.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/configmap.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/core_client.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/doc.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/endpoints.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/event.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/event_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/limitrange.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/namespace.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/namespace_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/node.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/node_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/persistentvolume.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/persistentvolumeclaim.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/pod.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/pod_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/podtemplate.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/replicationcontroller.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/resourcequota.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/secret.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/service.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/service_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/serviceaccount.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/policy/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/core/v1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_componentstatus.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_configmap.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_core_client.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_endpoints.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_event.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_event_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_limitrange.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_namespace.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_namespace_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_node.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_node_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_persistentvolume.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_persistentvolumeclaim.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_pod.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_pod_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_podtemplate.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_replicationcontroller.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_resourcequota.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_secret.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_service.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_service_expansion.go",
        "k8s.io/client-go/kubernetes/typed/core/v1/fake/fake_serviceaccount.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/policy/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/extensions/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/daemonset.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/deployment.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/deployment_expansion.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/extensions_client.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/ingress.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/podsecuritypolicy.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/replicaset.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/scale.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/scale_expansion.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_daemonset.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_deployment.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_deployment_expansion.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_extensions_client.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_ingress.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_podsecuritypolicy.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_replicaset.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_scale.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_scale_expansion.go",
        "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake/fake_thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/extensions/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/policy/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/eviction.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/eviction_expansion.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/poddisruptionbudget.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/policy_client.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/policy/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake/fake_eviction.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake/fake_eviction_expansion.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake/fake_poddisruptionbudget.go",
        "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake/fake_policy_client.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/policy/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/policy/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/clusterrole.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/clusterrolebinding.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/doc.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/rbac_client.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/role.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/fake_clusterrole.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/fake_clusterrolebinding.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/fake_rbac_client.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/fake_role.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake/fake_rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/storage/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/storage_client.go",
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/storageclass.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/storage/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake/fake_storage_client.go",
        "k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake/fake_storageclass.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/storage/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/storage/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/api",
    srcs = [
        "k8s.io/client-go/pkg/api/conversion.go",
        "k8s.io/client-go/pkg/api/defaults.go",
        "k8s.io/client-go/pkg/api/doc.go",
        "k8s.io/client-go/pkg/api/field_constants.go",
        "k8s.io/client-go/pkg/api/helpers.go",
        "k8s.io/client-go/pkg/api/json.go",
        "k8s.io/client-go/pkg/api/meta.go",
        "k8s.io/client-go/pkg/api/ref.go",
        "k8s.io/client-go/pkg/api/register.go",
        "k8s.io/client-go/pkg/api/resource_helpers.go",
        "k8s.io/client-go/pkg/api/types.go",
        "k8s.io/client-go/pkg/api/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/util/labels",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/api/install",
    srcs = ["k8s.io/client-go/pkg/api/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/api/v1",
    srcs = [
        "k8s.io/client-go/pkg/api/v1/conversion.go",
        "k8s.io/client-go/pkg/api/v1/defaults.go",
        "k8s.io/client-go/pkg/api/v1/doc.go",
        "k8s.io/client-go/pkg/api/v1/generate.go",
        "k8s.io/client-go/pkg/api/v1/generated.pb.go",
        "k8s.io/client-go/pkg/api/v1/helpers.go",
        "k8s.io/client-go/pkg/api/v1/meta.go",
        "k8s.io/client-go/pkg/api/v1/ref.go",
        "k8s.io/client-go/pkg/api/v1/register.go",
        "k8s.io/client-go/pkg/api/v1/resource_helpers.go",
        "k8s.io/client-go/pkg/api/v1/types.generated.go",
        "k8s.io/client-go/pkg/api/v1/types.go",
        "k8s.io/client-go/pkg/api/v1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/api/v1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/api/v1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/api/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/util/rand",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/extensions",
        "//vendor:k8s.io/client-go/pkg/util",
        "//vendor:k8s.io/client-go/pkg/util/parsers",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/apps",
    srcs = [
        "k8s.io/client-go/pkg/apis/apps/doc.go",
        "k8s.io/client-go/pkg/apis/apps/register.go",
        "k8s.io/client-go/pkg/apis/apps/types.go",
        "k8s.io/client-go/pkg/apis/apps/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/pkg/api",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/apps/install",
    srcs = ["k8s.io/client-go/pkg/apis/apps/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/apps",
        "//vendor:k8s.io/client-go/pkg/apis/apps/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/apps/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/apps/v1beta1/conversion.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/apps/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/apps",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authentication",
    srcs = [
        "k8s.io/client-go/pkg/apis/authentication/doc.go",
        "k8s.io/client-go/pkg/apis/authentication/register.go",
        "k8s.io/client-go/pkg/apis/authentication/types.go",
        "k8s.io/client-go/pkg/apis/authentication/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authentication/install",
    srcs = ["k8s.io/client-go/pkg/apis/authentication/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authentication",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authentication/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/conversion.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/authentication/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/authentication",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authorization",
    srcs = [
        "k8s.io/client-go/pkg/apis/authorization/doc.go",
        "k8s.io/client-go/pkg/apis/authorization/register.go",
        "k8s.io/client-go/pkg/apis/authorization/types.go",
        "k8s.io/client-go/pkg/apis/authorization/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authorization/install",
    srcs = ["k8s.io/client-go/pkg/apis/authorization/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authorization",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/authorization/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/conversion.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/authorization/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/authorization",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/autoscaling",
    srcs = [
        "k8s.io/client-go/pkg/apis/autoscaling/doc.go",
        "k8s.io/client-go/pkg/apis/autoscaling/register.go",
        "k8s.io/client-go/pkg/apis/autoscaling/types.go",
        "k8s.io/client-go/pkg/apis/autoscaling/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/autoscaling/install",
    srcs = ["k8s.io/client-go/pkg/apis/autoscaling/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling/v1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/autoscaling/v1",
    srcs = [
        "k8s.io/client-go/pkg/apis/autoscaling/v1/defaults.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/doc.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/register.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/types.generated.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/types.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/autoscaling/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/batch",
    srcs = [
        "k8s.io/client-go/pkg/apis/batch/doc.go",
        "k8s.io/client-go/pkg/apis/batch/register.go",
        "k8s.io/client-go/pkg/apis/batch/types.go",
        "k8s.io/client-go/pkg/apis/batch/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/pkg/api",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/batch/install",
    srcs = ["k8s.io/client-go/pkg/apis/batch/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/batch",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v1",
        "//vendor:k8s.io/client-go/pkg/apis/batch/v2alpha1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/batch/v1",
    srcs = [
        "k8s.io/client-go/pkg/apis/batch/v1/conversion.go",
        "k8s.io/client-go/pkg/apis/batch/v1/defaults.go",
        "k8s.io/client-go/pkg/apis/batch/v1/doc.go",
        "k8s.io/client-go/pkg/apis/batch/v1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/batch/v1/register.go",
        "k8s.io/client-go/pkg/apis/batch/v1/types.generated.go",
        "k8s.io/client-go/pkg/apis/batch/v1/types.go",
        "k8s.io/client-go/pkg/apis/batch/v1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/batch/v1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/batch/v1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/batch/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/batch",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/batch/v2alpha1",
    srcs = [
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/conversion.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/defaults.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/doc.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/register.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/types.generated.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/types.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/batch/v2alpha1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/batch",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/certificates",
    srcs = [
        "k8s.io/client-go/pkg/apis/certificates/doc.go",
        "k8s.io/client-go/pkg/apis/certificates/helpers.go",
        "k8s.io/client-go/pkg/apis/certificates/register.go",
        "k8s.io/client-go/pkg/apis/certificates/types.go",
        "k8s.io/client-go/pkg/apis/certificates/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/certificates/install",
    srcs = ["k8s.io/client-go/pkg/apis/certificates/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/certificates",
        "//vendor:k8s.io/client-go/pkg/apis/certificates/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/extensions",
    srcs = [
        "k8s.io/client-go/pkg/apis/extensions/doc.go",
        "k8s.io/client-go/pkg/apis/extensions/helpers.go",
        "k8s.io/client-go/pkg/apis/extensions/register.go",
        "k8s.io/client-go/pkg/apis/extensions/types.go",
        "k8s.io/client-go/pkg/apis/extensions/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/extensions/install",
    srcs = ["k8s.io/client-go/pkg/apis/extensions/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/extensions",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/extensions/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/conversion.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/extensions/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/autoscaling",
        "//vendor:k8s.io/client-go/pkg/apis/extensions",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/policy",
    srcs = [
        "k8s.io/client-go/pkg/apis/policy/doc.go",
        "k8s.io/client-go/pkg/apis/policy/register.go",
        "k8s.io/client-go/pkg/apis/policy/types.go",
        "k8s.io/client-go/pkg/apis/policy/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/policy/install",
    srcs = ["k8s.io/client-go/pkg/apis/policy/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/policy",
        "//vendor:k8s.io/client-go/pkg/apis/policy/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/policy/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/policy/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/policy/v1beta1/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/client-go/pkg/apis/policy",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/rbac",
    srcs = [
        "k8s.io/client-go/pkg/apis/rbac/doc.go",
        "k8s.io/client-go/pkg/apis/rbac/helpers.go",
        "k8s.io/client-go/pkg/apis/rbac/register.go",
        "k8s.io/client-go/pkg/apis/rbac/types.go",
        "k8s.io/client-go/pkg/apis/rbac/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/rbac/install",
    srcs = ["k8s.io/client-go/pkg/apis/rbac/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/rbac",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1alpha1",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/rbac/v1alpha1",
    srcs = [
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/conversion.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/defaults.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/doc.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/helpers.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/register.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/types.generated.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/types.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/rbac/v1alpha1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/rbac",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/storage",
    srcs = [
        "k8s.io/client-go/pkg/apis/storage/doc.go",
        "k8s.io/client-go/pkg/apis/storage/register.go",
        "k8s.io/client-go/pkg/apis/storage/types.go",
        "k8s.io/client-go/pkg/apis/storage/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/storage/install",
    srcs = ["k8s.io/client-go/pkg/apis/storage/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/storage",
        "//vendor:k8s.io/client-go/pkg/apis/storage/v1beta1",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/storage/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/storage/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/storage/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/storage",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/util",
    srcs = [
        "k8s.io/client-go/pkg/util/doc.go",
        "k8s.io/client-go/pkg/util/template.go",
        "k8s.io/client-go/pkg/util/umask.go",
        "k8s.io/client-go/pkg/util/util.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/pkg/util/labels",
    srcs = [
        "k8s.io/client-go/pkg/util/labels/doc.go",
        "k8s.io/client-go/pkg/util/labels/labels.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/apis/meta/v1"],
)

go_library(
    name = "k8s.io/client-go/pkg/util/parsers",
    srcs = ["k8s.io/client-go/pkg/util/parsers/parsers.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/docker/distribution/reference"],
)

go_library(
    name = "k8s.io/client-go/pkg/version",
    srcs = [
        "k8s.io/client-go/pkg/version/base.go",
        "k8s.io/client-go/pkg/version/doc.go",
        "k8s.io/client-go/pkg/version/version.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/version"],
)

go_library(
    name = "k8s.io/client-go/plugin/pkg/auth/authenticator/token/oidc/testing",
    srcs = ["k8s.io/client-go/plugin/pkg/auth/authenticator/token/oidc/testing/provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oidc",
    ],
)

go_library(
    name = "k8s.io/client-go/plugin/pkg/client/auth",
    srcs = ["k8s.io/client-go/plugin/pkg/client/auth/plugins.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/gcp",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/oidc",
    ],
)

go_test(
    name = "k8s.io/client-go/plugin/pkg/client/auth/gcp_test",
    srcs = ["k8s.io/client-go/plugin/pkg/client/auth/gcp/gcp_test.go"],
    library = ":k8s.io/client-go/plugin/pkg/client/auth/gcp",
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/oauth2"],
)

go_library(
    name = "k8s.io/client-go/plugin/pkg/client/auth/gcp",
    srcs = ["k8s.io/client-go/plugin/pkg/client/auth/gcp/gcp.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/jsonpath",
    ],
)

go_test(
    name = "k8s.io/client-go/plugin/pkg/client/auth/oidc_test",
    srcs = ["k8s.io/client-go/plugin/pkg/client/auth/oidc/oidc_test.go"],
    library = ":k8s.io/client-go/plugin/pkg/client/auth/oidc",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oauth2",
        "//vendor:k8s.io/client-go/plugin/pkg/auth/authenticator/token/oidc/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/plugin/pkg/client/auth/oidc",
    srcs = ["k8s.io/client-go/plugin/pkg/client/auth/oidc/oidc.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oauth2",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_test(
    name = "k8s.io/client-go/rest_test",
    srcs = [
        "k8s.io/client-go/rest/client_test.go",
        "k8s.io/client-go/rest/config_test.go",
        "k8s.io/client-go/rest/plugin_test.go",
        "k8s.io/client-go/rest/request_test.go",
        "k8s.io/client-go/rest/url_utils_test.go",
        "k8s.io/client-go/rest/urlbackoff_test.go",
    ],
    library = ":k8s.io/client-go/rest",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest/watch",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/util/clock",
        "//vendor:k8s.io/client-go/util/flowcontrol",
        "//vendor:k8s.io/client-go/util/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/rest",
    srcs = [
        "k8s.io/client-go/rest/client.go",
        "k8s.io/client-go/rest/config.go",
        "k8s.io/client-go/rest/plugin.go",
        "k8s.io/client-go/rest/request.go",
        "k8s.io/client-go/rest/transport.go",
        "k8s.io/client-go/rest/url_utils.go",
        "k8s.io/client-go/rest/urlbackoff.go",
        "k8s.io/client-go/rest/versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/version",
        "//vendor:k8s.io/client-go/rest/watch",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/metrics",
        "//vendor:k8s.io/client-go/transport",
        "//vendor:k8s.io/client-go/util/cert",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_library(
    name = "k8s.io/client-go/rest/fake",
    srcs = ["k8s.io/client-go/rest/fake/fake.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_library(
    name = "k8s.io/client-go/rest/watch",
    srcs = [
        "k8s.io/client-go/rest/watch/decoder.go",
        "k8s.io/client-go/rest/watch/encoder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_test(
    name = "k8s.io/client-go/rest/watch_xtest",
    srcs = [
        "k8s.io/client-go/rest/watch/decoder_test.go",
        "k8s.io/client-go/rest/watch/encoder_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest/watch",
    ],
)

go_library(
    name = "k8s.io/client-go/testing",
    srcs = [
        "k8s.io/client-go/testing/actions.go",
        "k8s.io/client-go/testing/fake.go",
        "k8s.io/client-go/testing/fixture.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/version",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/auth",
    srcs = ["k8s.io/client-go/tools/auth/clientauth.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/rest"],
)

go_test(
    name = "k8s.io/client-go/tools/auth_xtest",
    srcs = ["k8s.io/client-go/tools/auth/clientauth_test.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/tools/auth"],
)

go_test(
    name = "k8s.io/client-go/tools/cache_test",
    srcs = [
        "k8s.io/client-go/tools/cache/controller_test.go",
        "k8s.io/client-go/tools/cache/delta_fifo_test.go",
        "k8s.io/client-go/tools/cache/expiration_cache_test.go",
        "k8s.io/client-go/tools/cache/fifo_test.go",
        "k8s.io/client-go/tools/cache/index_test.go",
        "k8s.io/client-go/tools/cache/mutation_detector_test.go",
        "k8s.io/client-go/tools/cache/processor_listener_test.go",
        "k8s.io/client-go/tools/cache/reflector_test.go",
        "k8s.io/client-go/tools/cache/store_test.go",
        "k8s.io/client-go/tools/cache/undelta_store_test.go",
    ],
    library = ":k8s.io/client-go/tools/cache",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/cache/testing",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/cache",
    srcs = [
        "k8s.io/client-go/tools/cache/controller.go",
        "k8s.io/client-go/tools/cache/delta_fifo.go",
        "k8s.io/client-go/tools/cache/doc.go",
        "k8s.io/client-go/tools/cache/expiration_cache.go",
        "k8s.io/client-go/tools/cache/expiration_cache_fakes.go",
        "k8s.io/client-go/tools/cache/fake_custom_store.go",
        "k8s.io/client-go/tools/cache/fifo.go",
        "k8s.io/client-go/tools/cache/index.go",
        "k8s.io/client-go/tools/cache/listers.go",
        "k8s.io/client-go/tools/cache/listwatch.go",
        "k8s.io/client-go/tools/cache/mutation_detector.go",
        "k8s.io/client-go/tools/cache/reflector.go",
        "k8s.io/client-go/tools/cache/shared_informer.go",
        "k8s.io/client-go/tools/cache/store.go",
        "k8s.io/client-go/tools/cache/thread_safe_store.go",
        "k8s.io/client-go/tools/cache/undelta_store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_test(
    name = "k8s.io/client-go/tools/cache/testing_test",
    srcs = ["k8s.io/client-go/tools/cache/testing/fake_controller_source_test.go"],
    library = ":k8s.io/client-go/tools/cache/testing",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/cache/testing",
    srcs = ["k8s.io/client-go/tools/cache/testing/fake_controller_source.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
    ],
)

go_test(
    name = "k8s.io/client-go/tools/clientcmd_test",
    srcs = [
        "k8s.io/client-go/tools/clientcmd/client_config_test.go",
        "k8s.io/client-go/tools/clientcmd/loader_test.go",
        "k8s.io/client-go/tools/clientcmd/merged_client_builder_test.go",
        "k8s.io/client-go/tools/clientcmd/validation_test.go",
    ],
    library = ":k8s.io/client-go/tools/clientcmd",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/imdario/mergo",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/latest",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/clientcmd",
    srcs = [
        "k8s.io/client-go/tools/clientcmd/auth_loaders.go",
        "k8s.io/client-go/tools/clientcmd/client_config.go",
        "k8s.io/client-go/tools/clientcmd/config.go",
        "k8s.io/client-go/tools/clientcmd/doc.go",
        "k8s.io/client-go/tools/clientcmd/helpers.go",
        "k8s.io/client-go/tools/clientcmd/loader.go",
        "k8s.io/client-go/tools/clientcmd/merged_client_builder.go",
        "k8s.io/client-go/tools/clientcmd/overrides.go",
        "k8s.io/client-go/tools/clientcmd/validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/howeyc/gopass",
        "//vendor:github.com/imdario/mergo",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/auth",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/latest",
        "//vendor:k8s.io/client-go/util/homedir",
    ],
)

go_test(
    name = "k8s.io/client-go/tools/clientcmd/api_test",
    srcs = [
        "k8s.io/client-go/tools/clientcmd/api/helpers_test.go",
        "k8s.io/client-go/tools/clientcmd/api/types_test.go",
    ],
    library = ":k8s.io/client-go/tools/clientcmd/api",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/ghodss/yaml"],
)

go_library(
    name = "k8s.io/client-go/tools/clientcmd/api",
    srcs = [
        "k8s.io/client-go/tools/clientcmd/api/helpers.go",
        "k8s.io/client-go/tools/clientcmd/api/register.go",
        "k8s.io/client-go/tools/clientcmd/api/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/clientcmd/api/latest",
    srcs = ["k8s.io/client-go/tools/clientcmd/api/latest/latest.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/json",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/versioning",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/v1",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/clientcmd/api/v1",
    srcs = [
        "k8s.io/client-go/tools/clientcmd/api/v1/conversion.go",
        "k8s.io/client-go/tools/clientcmd/api/v1/register.go",
        "k8s.io/client-go/tools/clientcmd/api/v1/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/metrics",
    srcs = ["k8s.io/client-go/tools/metrics/metrics.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/tools/portforward",
    srcs = [
        "k8s.io/client-go/tools/portforward/doc.go",
        "k8s.io/client-go/tools/portforward/portforward.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/client-go/pkg/api",
    ],
)

go_test(
    name = "k8s.io/client-go/tools/record_test",
    srcs = [
        "k8s.io/client-go/tools/record/event_test.go",
        "k8s.io/client-go/tools/record/events_cache_test.go",
    ],
    library = ":k8s.io/client-go/tools/record",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/strategicpatch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/client-go/tools/record",
    srcs = [
        "k8s.io/client-go/tools/record/doc.go",
        "k8s.io/client-go/tools/record/event.go",
        "k8s.io/client-go/tools/record/events_cache.go",
        "k8s.io/client-go/tools/record/fake.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/golang/groupcache/lru",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/strategicpatch",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_test(
    name = "k8s.io/client-go/transport_test",
    srcs = [
        "k8s.io/client-go/transport/cache_test.go",
        "k8s.io/client-go/transport/round_trippers_test.go",
        "k8s.io/client-go/transport/transport_test.go",
    ],
    library = ":k8s.io/client-go/transport",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/transport",
    srcs = [
        "k8s.io/client-go/transport/cache.go",
        "k8s.io/client-go/transport/config.go",
        "k8s.io/client-go/transport/round_trippers.go",
        "k8s.io/client-go/transport/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
    ],
)

go_library(
    name = "k8s.io/gengo/args",
    srcs = ["k8s.io/gengo/args/args.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/parser",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/examples/deepcopy-gen/generators",
    srcs = ["k8s.io/gengo/examples/deepcopy-gen/generators/deepcopy.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/examples/set-gen/sets",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/examples/defaulter-gen/generators",
    srcs = ["k8s.io/gengo/examples/defaulter-gen/generators/defaulter.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/examples/import-boss/generators",
    srcs = ["k8s.io/gengo/examples/import-boss/generators/import_restrict.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/examples/set-gen/generators",
    srcs = [
        "k8s.io/gengo/examples/set-gen/generators/sets.go",
        "k8s.io/gengo/examples/set-gen/generators/tags.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/examples/set-gen/sets",
    srcs = [
        "k8s.io/gengo/examples/set-gen/sets/byte.go",
        "k8s.io/gengo/examples/set-gen/sets/doc.go",
        "k8s.io/gengo/examples/set-gen/sets/empty.go",
        "k8s.io/gengo/examples/set-gen/sets/int.go",
        "k8s.io/gengo/examples/set-gen/sets/int64.go",
        "k8s.io/gengo/examples/set-gen/sets/string.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/gengo/generator",
    srcs = [
        "k8s.io/gengo/generator/default_generator.go",
        "k8s.io/gengo/generator/default_package.go",
        "k8s.io/gengo/generator/doc.go",
        "k8s.io/gengo/generator/error_tracker.go",
        "k8s.io/gengo/generator/execute.go",
        "k8s.io/gengo/generator/generator.go",
        "k8s.io/gengo/generator/import_tracker.go",
        "k8s.io/gengo/generator/snippet_writer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/parser",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/namer",
    srcs = [
        "k8s.io/gengo/namer/doc.go",
        "k8s.io/gengo/namer/import_tracker.go",
        "k8s.io/gengo/namer/namer.go",
        "k8s.io/gengo/namer/order.go",
        "k8s.io/gengo/namer/plural_namer.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/gengo/types"],
)

go_library(
    name = "k8s.io/gengo/parser",
    srcs = [
        "k8s.io/gengo/parser/doc.go",
        "k8s.io/gengo/parser/parse.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_library(
    name = "k8s.io/gengo/types",
    srcs = [
        "k8s.io/gengo/types/comments.go",
        "k8s.io/gengo/types/doc.go",
        "k8s.io/gengo/types/flatten.go",
        "k8s.io/gengo/types/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/heapster/metrics/api/v1/types",
    srcs = [
        "k8s.io/heapster/metrics/api/v1/types/historical_types.go",
        "k8s.io/heapster/metrics/api/v1/types/model_types.go",
        "k8s.io/heapster/metrics/api/v1/types/types.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/heapster/metrics/apis/metrics/v1alpha1",
    srcs = ["k8s.io/heapster/metrics/apis/metrics/v1alpha1/types.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/rbac/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/clusterrole.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/clusterrolebinding.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/generated_expansion.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/rbac_client.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/role.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/fake_clusterrole.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/fake_clusterrolebinding.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/fake_rbac_client.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/fake_role.go",
        "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake/fake_rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/rbac/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/rbac/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/rbac/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/helpers.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/rbac/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/apis/rbac",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/fields_test",
    srcs = [
        "k8s.io/apimachinery/pkg/fields/fields_test.go",
        "k8s.io/apimachinery/pkg/fields/selector_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/fields",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/fields",
    srcs = [
        "k8s.io/apimachinery/pkg/fields/doc.go",
        "k8s.io/apimachinery/pkg/fields/fields.go",
        "k8s.io/apimachinery/pkg/fields/requirements.go",
        "k8s.io/apimachinery/pkg/fields/selector.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/selection"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/fields",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/fields/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/fields/fields.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/fields/requirements.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/fields/selector.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/selection"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apimachinery/announced_test",
    srcs = ["k8s.io/apimachinery/pkg/apimachinery/announced/announced_test.go"],
    library = ":k8s.io/apimachinery/pkg/apimachinery/announced",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/sets"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apimachinery/announced",
    srcs = [
        "k8s.io/apimachinery/pkg/apimachinery/announced/announced.go",
        "k8s.io/apimachinery/pkg/apimachinery/announced/group_factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/announced",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/announced/announced.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/apimachinery/announced/group_factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/certificates/v1beta1",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/certificates_client.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/certificatesigningrequest.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/certificatesigningrequest_expansion.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/doc.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/generated_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/certificates/v1beta1",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_library(
    name = "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake",
    srcs = [
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake/doc.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake/fake_certificates_client.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake/fake_certificatesigningrequest.go",
        "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake/fake_certificatesigningrequest_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/typed/certificates/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/certificates/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "k8s.io/client-go/pkg/apis/certificates/v1beta1",
    srcs = [
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/conversion.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/defaults.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/doc.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/generated.pb.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/helpers.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/register.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/types.generated.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/types.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/types_swagger_doc_generated.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/zz_generated.conversion.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/zz_generated.deepcopy.go",
        "k8s.io/client-go/pkg/apis/certificates/v1beta1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/certificates",
    ],
)

go_library(
    name = "github.com/armon/circbuf",
    srcs = ["github.com/armon/circbuf/circbuf.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/apis/meta/internalversion_test",
    srcs = ["k8s.io/apimachinery/pkg/apis/meta/internalversion/register_test.go"],
    library = ":k8s.io/apimachinery/pkg/apis/meta/internalversion",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/apis/meta/internalversion",
    srcs = [
        "k8s.io/apimachinery/pkg/apis/meta/internalversion/conversion.go",
        "k8s.io/apimachinery/pkg/apis/meta/internalversion/register.go",
        "k8s.io/apimachinery/pkg/apis/meta/internalversion/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_test(
    name = "k8s.io/client-go/util/cert_test",
    srcs = ["k8s.io/client-go/util/cert/csr_test.go"],
    library = ":k8s.io/client-go/util/cert",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/cert",
    srcs = [
        "k8s.io/client-go/util/cert/cert.go",
        "k8s.io/client-go/util/cert/csr.go",
        "k8s.io/client-go/util/cert/io.go",
        "k8s.io/client-go/util/cert/pem.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/cert/triple",
    srcs = ["k8s.io/client-go/util/cert/triple/triple.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/util/cert"],
)

go_test(
    name = "k8s.io/client-go/util/clock_test",
    srcs = ["k8s.io/client-go/util/clock/clock_test.go"],
    library = ":k8s.io/client-go/util/clock",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/clock",
    srcs = ["k8s.io/client-go/util/clock/clock.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/client-go/util/flowcontrol_test",
    srcs = [
        "k8s.io/client-go/util/flowcontrol/backoff_test.go",
        "k8s.io/client-go/util/flowcontrol/throttle_test.go",
    ],
    library = ":k8s.io/client-go/util/flowcontrol",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/util/clock"],
)

go_library(
    name = "k8s.io/client-go/util/flowcontrol",
    srcs = [
        "k8s.io/client-go/util/flowcontrol/backoff.go",
        "k8s.io/client-go/util/flowcontrol/throttle.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/juju/ratelimit",
        "//vendor:k8s.io/client-go/util/clock",
        "//vendor:k8s.io/client-go/util/integer",
    ],
)

go_library(
    name = "k8s.io/client-go/util/homedir",
    srcs = ["k8s.io/client-go/util/homedir/homedir.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/client-go/util/integer_test",
    srcs = ["k8s.io/client-go/util/integer/integer_test.go"],
    library = ":k8s.io/client-go/util/integer",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/integer",
    srcs = ["k8s.io/client-go/util/integer/integer.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/client-go/util/jsonpath_test",
    srcs = [
        "k8s.io/client-go/util/jsonpath/jsonpath_test.go",
        "k8s.io/client-go/util/jsonpath/parser_test.go",
    ],
    library = ":k8s.io/client-go/util/jsonpath",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/jsonpath",
    srcs = [
        "k8s.io/client-go/util/jsonpath/doc.go",
        "k8s.io/client-go/util/jsonpath/jsonpath.go",
        "k8s.io/client-go/util/jsonpath/node.go",
        "k8s.io/client-go/util/jsonpath/parser.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/third_party/forked/golang/template"],
)

go_test(
    name = "k8s.io/client-go/util/testing_test",
    srcs = ["k8s.io/client-go/util/testing/fake_handler_test.go"],
    library = ":k8s.io/client-go/util/testing",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/util/testing",
    srcs = [
        "k8s.io/client-go/util/testing/fake_handler.go",
        "k8s.io/client-go/util/testing/tmpdir.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/features",
    srcs = ["k8s.io/apiserver/pkg/features/kube_features.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/util/feature"],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/feature_test",
    srcs = ["k8s.io/apiserver/pkg/util/feature/feature_gate_test.go"],
    library = ":k8s.io/apiserver/pkg/util/feature",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/feature",
    srcs = ["k8s.io/apiserver/pkg/util/feature/feature_gate.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/flag_test",
    srcs = ["k8s.io/apiserver/pkg/util/flag/namedcertkey_flag_test.go"],
    library = ":k8s.io/apiserver/pkg/util/flag",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/flag",
    srcs = [
        "k8s.io/apiserver/pkg/util/flag/configuration_map.go",
        "k8s.io/apiserver/pkg/util/flag/flags.go",
        "k8s.io/apiserver/pkg/util/flag/namedcertkey_flag.go",
        "k8s.io/apiserver/pkg/util/flag/string_flag.go",
        "k8s.io/apiserver/pkg/util/flag/tristate.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/doc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/password/doc.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password/allow_test",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/password/allow/allow_test.go"],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/password/allow",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password/allow",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/password/allow/allow.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password/keystone",
    srcs = [
        "k8s.io/apiserver/plugin/pkg/authenticator/password/keystone/doc.go",
        "k8s.io/apiserver/plugin/pkg/authenticator/password/keystone/keystone.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile_test",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile/passwordfile_test.go"],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile/passwordfile.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth_test",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth/basicauth_test.go"],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth/basicauth.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken_test",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken/anytoken_test.go"],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken/anytoken.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/oidc_test",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/oidc_test.go"],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/token/oidc",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/testing",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/oidc",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/oidc.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/testing",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/oidc/testing/provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oidc",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest/tokentest.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/webhook_test",
    srcs = [
        "k8s.io/apiserver/plugin/pkg/authenticator/token/webhook/certs_test.go",
        "k8s.io/apiserver/plugin/pkg/authenticator/token/webhook/webhook_test.go",
    ],
    library = ":k8s.io/apiserver/plugin/pkg/authenticator/token/webhook",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/v1beta1",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/v1",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authenticator/token/webhook",
    srcs = ["k8s.io/apiserver/plugin/pkg/authenticator/token/webhook/webhook.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/util/cache",
        "//vendor:k8s.io/apiserver/pkg/util/webhook",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/install",
        "//vendor:k8s.io/client-go/pkg/apis/authentication/v1beta1",
    ],
)

go_test(
    name = "k8s.io/apiserver/plugin/pkg/authorizer/webhook_test",
    srcs = [
        "k8s.io/apiserver/plugin/pkg/authorizer/webhook/certs_test.go",
        "k8s.io/apiserver/plugin/pkg/authorizer/webhook/webhook_test.go",
    ],
    library = ":k8s.io/apiserver/plugin/pkg/authorizer/webhook",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/v1",
    ],
)

go_library(
    name = "k8s.io/apiserver/plugin/pkg/authorizer/webhook",
    srcs = ["k8s.io/apiserver/plugin/pkg/authorizer/webhook/webhook.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/util/cache",
        "//vendor:k8s.io/apiserver/pkg/util/webhook",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/install",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/strategicpatch_test",
    srcs = ["k8s.io/apimachinery/pkg/util/strategicpatch/patch_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/strategicpatch",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/ghodss/yaml",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/strategicpatch",
    srcs = ["k8s.io/apimachinery/pkg/util/strategicpatch/patch.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
        "//vendor:k8s.io/apimachinery/third_party/forked/golang/json",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/uuid",
    srcs = ["k8s.io/apimachinery/pkg/util/uuid/uuid.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/pborman/uuid",
        "//vendor:k8s.io/apimachinery/pkg/types",
    ],
)

go_library(
    name = "k8s.io/apimachinery/third_party/forked/golang/json",
    srcs = ["k8s.io/apimachinery/third_party/forked/golang/json/fields.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/third_party/forked/golang/netutil",
    srcs = ["k8s.io/apimachinery/third_party/forked/golang/netutil/addr.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/pkg/util/proxy_test",
    srcs = [
        "k8s.io/apiserver/pkg/util/proxy/dial_test.go",
        "k8s.io/apiserver/pkg/util/proxy/transport_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/util/proxy",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/net"],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/proxy",
    srcs = [
        "k8s.io/apiserver/pkg/util/proxy/dial.go",
        "k8s.io/apiserver/pkg/util/proxy/doc.go",
        "k8s.io/apiserver/pkg/util/proxy/transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/html",
        "//vendor:golang.org/x/net/html/atom",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/third_party/forked/golang/netutil",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/trie",
    srcs = ["k8s.io/apiserver/pkg/util/trie/trie.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/authentication/authenticatorfactory",
    srcs = [
        "k8s.io/apiserver/pkg/authentication/authenticatorfactory/delegating.go",
        "k8s.io/apiserver/pkg/authentication/authenticatorfactory/loopback.go",
        "k8s.io/apiserver/pkg/authentication/authenticatorfactory/requestheader.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/group",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/anonymous",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/bearertoken",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/headerrequest",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/union",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/x509",
        "//vendor:k8s.io/apiserver/pkg/authentication/token/tokenfile",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/token/webhook",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/authorization/authorizerfactory_test",
    srcs = ["k8s.io/apiserver/pkg/authorization/authorizerfactory/authz_test.go"],
    library = ":k8s.io/apiserver/pkg/authorization/authorizerfactory",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/authorization/authorizerfactory",
    srcs = [
        "k8s.io/apiserver/pkg/authorization/authorizerfactory/builtin.go",
        "k8s.io/apiserver/pkg/authorization/authorizerfactory/delegating.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/plugin/pkg/authorizer/webhook",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/validation/path_test",
    srcs = ["k8s.io/apimachinery/pkg/api/validation/path/name_test.go"],
    library = ":k8s.io/apimachinery/pkg/api/validation/path",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/validation/path",
    srcs = ["k8s.io/apimachinery/pkg/api/validation/path/name.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/pkg/server/filters_test",
    srcs = [
        "k8s.io/apiserver/pkg/server/filters/cors_test.go",
        "k8s.io/apiserver/pkg/server/filters/maxinflight_test.go",
        "k8s.io/apiserver/pkg/server/filters/timeout_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/server/filters",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/endpoints/filters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/filters",
    srcs = [
        "k8s.io/apiserver/pkg/server/filters/cors.go",
        "k8s.io/apiserver/pkg/server/filters/doc.go",
        "k8s.io/apiserver/pkg/server/filters/longrunning.go",
        "k8s.io/apiserver/pkg/server/filters/maxinflight.go",
        "k8s.io/apiserver/pkg/server/filters/timeout.go",
        "k8s.io/apiserver/pkg/server/filters/wrap.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/server/httplog",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/storagebackend",
    srcs = ["k8s.io/apiserver/pkg/storage/storagebackend/config.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/runtime"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/httpstream_test",
    srcs = ["k8s.io/apimachinery/pkg/util/httpstream/httpstream_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/httpstream",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/httpstream",
    srcs = [
        "k8s.io/apimachinery/pkg/util/httpstream/doc.go",
        "k8s.io/apimachinery/pkg/util/httpstream/httpstream.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/httpstream/spdy_test",
    srcs = [
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/connection_test.go",
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/roundtripper_test.go",
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/upgrade_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/util/httpstream/spdy",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/elazarl/goproxy",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/httpstream/spdy",
    srcs = [
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/connection.go",
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/roundtripper.go",
        "k8s.io/apimachinery/pkg/util/httpstream/spdy/upgrade.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/spdystream",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/third_party/forked/golang/netutil",
    ],
)

go_library(
    name = "github.com/appc/spec/schema/types/resource",
    srcs = [
        "github.com/appc/spec/schema/types/resource/amount.go",
        "github.com/appc/spec/schema/types/resource/math.go",
        "github.com/appc/spec/schema/types/resource/quantity.go",
        "github.com/appc/spec/schema/types/resource/scale_int.go",
        "github.com/appc/spec/schema/types/resource/suffix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/equality",
    srcs = ["k8s.io/apimachinery/pkg/api/equality/semantic.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/resource_test",
    srcs = [
        "k8s.io/apimachinery/pkg/api/resource/amount_test.go",
        "k8s.io/apimachinery/pkg/api/resource/math_test.go",
        "k8s.io/apimachinery/pkg/api/resource/quantity_test.go",
        "k8s.io/apimachinery/pkg/api/resource/scale_int_test.go",
    ],
    library = ":k8s.io/apimachinery/pkg/api/resource",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/resource",
    srcs = [
        "k8s.io/apimachinery/pkg/api/resource/amount.go",
        "k8s.io/apimachinery/pkg/api/resource/generated.pb.go",
        "k8s.io/apimachinery/pkg/api/resource/math.go",
        "k8s.io/apimachinery/pkg/api/resource/quantity.go",
        "k8s.io/apimachinery/pkg/api/resource/quantity_proto.go",
        "k8s.io/apimachinery/pkg/api/resource/scale_int.go",
        "k8s.io/apimachinery/pkg/api/resource/suffix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/resource_xtest",
    srcs = ["k8s.io/apimachinery/pkg/api/resource/quantity_example_test.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/api/resource"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/api/testing",
    srcs = [
        "k8s.io/apimachinery/pkg/api/testing/codec.go",
        "k8s.io/apimachinery/pkg/api/testing/fuzzer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apimachinery/pkg/types",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/util/trace",
    srcs = ["k8s.io/apiserver/pkg/util/trace/trace.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/strategicpatch",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/strategicpatch/patch.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
        "//vendor:k8s.io/apimachinery/third_party/forked/golang/json",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/third_party/forked/golang/json",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/third_party/forked/golang/json/fields.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/third_party/forked/golang/template",
    srcs = [
        "k8s.io/client-go/third_party/forked/golang/template/exec.go",
        "k8s.io/client-go/third_party/forked/golang/template/funcs.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/client-go/util/workqueue_test",
    srcs = [
        "k8s.io/client-go/util/workqueue/default_rate_limiters_test.go",
        "k8s.io/client-go/util/workqueue/delaying_queue_test.go",
        "k8s.io/client-go/util/workqueue/rate_limitting_queue_test.go",
        "k8s.io/client-go/util/workqueue/timed_queue_test.go",
    ],
    library = ":k8s.io/client-go/util/workqueue",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/client-go/util/workqueue",
    srcs = [
        "k8s.io/client-go/util/workqueue/default_rate_limiters.go",
        "k8s.io/client-go/util/workqueue/delaying_queue.go",
        "k8s.io/client-go/util/workqueue/doc.go",
        "k8s.io/client-go/util/workqueue/metrics.go",
        "k8s.io/client-go/util/workqueue/parallelizer.go",
        "k8s.io/client-go/util/workqueue/queue.go",
        "k8s.io/client-go/util/workqueue/rate_limitting_queue.go",
        "k8s.io/client-go/util/workqueue/timed_queue.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/juju/ratelimit",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_test(
    name = "k8s.io/client-go/util/workqueue_xtest",
    srcs = ["k8s.io/client-go/util/workqueue/queue_test.go"],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/client-go/util/workqueue"],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/intstr_test",
    srcs = ["k8s.io/apimachinery/pkg/util/intstr/intstr_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/intstr",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/ghodss/yaml"],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/intstr",
    srcs = [
        "k8s.io/apimachinery/pkg/util/intstr/generated.pb.go",
        "k8s.io/apimachinery/pkg/util/intstr/intstr.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/equality",
    srcs = ["k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/equality/semantic.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/amount.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/generated.pb.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/math.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/quantity.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/quantity_proto.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/scale_int.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/api/resource/suffix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
    ],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/httpstream",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/httpstream/doc.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/httpstream/httpstream.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/intstr",
    srcs = [
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/intstr/generated.pb.go",
        "k8s.io/client-go/_vendor/k8s.io/apimachinery/pkg/util/intstr/intstr.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
    ],
)

go_test(
    name = "k8s.io/client-go/tools/portforward_test",
    srcs = ["k8s.io/client-go/tools/portforward/portforward_test.go"],
    library = ":k8s.io/client-go/tools/portforward",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/util/httpstream"],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage_test",
    srcs = [
        "k8s.io/apiserver/pkg/storage/cacher_whitebox_test.go",
        "k8s.io/apiserver/pkg/storage/selection_predicate_test.go",
        "k8s.io/apiserver/pkg/storage/time_budget_test.go",
        "k8s.io/apiserver/pkg/storage/util_test.go",
        "k8s.io/apiserver/pkg/storage/watch_cache_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/storage",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage",
    srcs = [
        "k8s.io/apiserver/pkg/storage/cacher.go",
        "k8s.io/apiserver/pkg/storage/doc.go",
        "k8s.io/apiserver/pkg/storage/errors.go",
        "k8s.io/apiserver/pkg/storage/interfaces.go",
        "k8s.io/apiserver/pkg/storage/selection_predicate.go",
        "k8s.io/apiserver/pkg/storage/time_budget.go",
        "k8s.io/apiserver/pkg/storage/util.go",
        "k8s.io/apiserver/pkg/storage/watch_cache.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/validation/path",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/routes/data/swagger",
    srcs = ["k8s.io/apiserver/pkg/server/routes/data/swagger/datafile.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/options",
    srcs = [
        "k8s.io/apiserver/pkg/server/options/authentication.go",
        "k8s.io/apiserver/pkg/server/options/authorization.go",
        "k8s.io/apiserver/pkg/server/options/doc.go",
        "k8s.io/apiserver/pkg/server/options/etcd.go",
        "k8s.io/apiserver/pkg/server/options/server_run_options.go",
        "k8s.io/apiserver/pkg/server/options/serving.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticatorfactory",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizerfactory",
        "//vendor:k8s.io/apiserver/pkg/features",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/apis/example",
    srcs = [
        "k8s.io/apiserver/pkg/apis/example/doc.go",
        "k8s.io/apiserver/pkg/apis/example/register.go",
        "k8s.io/apiserver/pkg/apis/example/types.go",
        "k8s.io/apiserver/pkg/apis/example/zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/apis/example/fuzzer",
    srcs = ["k8s.io/apiserver/pkg/apis/example/fuzzer/fuzzer.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/apis/example/install",
    srcs = ["k8s.io/apiserver/pkg/apis/example/install/install.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/apis/example/v1",
    srcs = [
        "k8s.io/apiserver/pkg/apis/example/v1/conversion.go",
        "k8s.io/apiserver/pkg/apis/example/v1/defaults.go",
        "k8s.io/apiserver/pkg/apis/example/v1/doc.go",
        "k8s.io/apiserver/pkg/apis/example/v1/generated.pb.go",
        "k8s.io/apiserver/pkg/apis/example/v1/register.go",
        "k8s.io/apiserver/pkg/apis/example/v1/types.generated.go",
        "k8s.io/apiserver/pkg/apis/example/v1/types.go",
        "k8s.io/apiserver/pkg/apis/example/v1/types_swagger_doc_generated.go",
        "k8s.io/apiserver/pkg/apis/example/v1/zz_generated.conversion.go",
        "k8s.io/apiserver/pkg/apis/example/v1/zz_generated.deepcopy.go",
        "k8s.io/apiserver/pkg/apis/example/v1/zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/api/validation_test",
    srcs = ["k8s.io/apimachinery/pkg/api/validation/objectmeta_test.go"],
    library = ":k8s.io/apimachinery/pkg/api/validation",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_binary(
    name = "k8s.io/client-go/examples/in-cluster_bin",
    library = ":k8s.io/client-go/examples/in-cluster",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/examples/in-cluster",
    srcs = ["k8s.io/client-go/examples/in-cluster/main.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_binary(
    name = "k8s.io/client-go/examples/out-of-cluster_bin",
    library = ":k8s.io/client-go/examples/out-of-cluster",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/examples/out-of-cluster",
    srcs = ["k8s.io/client-go/examples/out-of-cluster/main.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/tools/clientcmd",
    ],
)

go_binary(
    name = "k8s.io/client-go/examples/third-party-resources_bin",
    library = ":k8s.io/client-go/examples/third-party-resources",
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/client-go/examples/third-party-resources",
    srcs = [
        "k8s.io/client-go/examples/third-party-resources/main.go",
        "k8s.io/client-go/examples/third-party-resources/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/gcp",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
    ],
)

go_test(
    name = "k8s.io/apimachinery/pkg/util/jsonmergepatch_test",
    srcs = ["k8s.io/apimachinery/pkg/util/jsonmergepatch/patch_test.go"],
    library = ":k8s.io/apimachinery/pkg/util/jsonmergepatch",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
    ],
)

go_library(
    name = "k8s.io/apimachinery/pkg/util/jsonmergepatch",
    srcs = ["k8s.io/apimachinery/pkg/util/jsonmergepatch/patch.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
        "//vendor:k8s.io/apimachinery/pkg/util/strategicpatch",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints_test",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/apiserver_test.go",
        "k8s.io/apiserver/pkg/endpoints/installer_test.go",
        "k8s.io/apiserver/pkg/endpoints/proxy_test.go",
        "k8s.io/apiserver/pkg/endpoints/watch_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/endpoints",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/fuzzer",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/endpoints/filters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/endpoints/testing",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/apiserver.go",
        "k8s.io/apiserver/pkg/endpoints/discovery.go",
        "k8s.io/apiserver/pkg/endpoints/doc.go",
        "k8s.io/apiserver/pkg/endpoints/groupversion.go",
        "k8s.io/apiserver/pkg/endpoints/installer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/filters_test",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/filters/audit_test.go",
        "k8s.io/apiserver/pkg/endpoints/filters/authentication_test.go",
        "k8s.io/apiserver/pkg/endpoints/filters/authorization_test.go",
        "k8s.io/apiserver/pkg/endpoints/filters/impersonation_test.go",
        "k8s.io/apiserver/pkg/endpoints/filters/requestinfo_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/endpoints/filters",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/batch:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/client-go/pkg/apis/authentication",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/filters",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/filters/audit.go",
        "k8s.io/apiserver/pkg/endpoints/filters/authentication.go",
        "k8s.io/apiserver/pkg/endpoints/filters/authorization.go",
        "k8s.io/apiserver/pkg/endpoints/filters/doc.go",
        "k8s.io/apiserver/pkg/endpoints/filters/impersonation.go",
        "k8s.io/apiserver/pkg/endpoints/filters/requestinfo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/serviceaccount",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/server/httplog",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/apis/authentication",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/handlers_test",
    srcs = ["k8s.io/apiserver/pkg/endpoints/handlers/rest_test.go"],
    library = ":k8s.io/apiserver/pkg/endpoints/handlers",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/strategicpatch",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/handlers",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/handlers/discovery.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/doc.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/patch.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/proxy.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/rest.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/watch.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/evanphx/json-patch",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/strategicpatch",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server/httplog",
        "//vendor:k8s.io/apiserver/pkg/util/proxy",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
        "//vendor:k8s.io/apiserver/pkg/util/wsstream",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters_test",
    srcs = ["k8s.io/apiserver/pkg/endpoints/handlers/responsewriters/status_test.go"],
    library = ":k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters/doc.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters/errors.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters/status.go",
        "k8s.io/apiserver/pkg/endpoints/handlers/responsewriters/writers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/negotiation",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/util/flushwriter",
        "//vendor:k8s.io/apiserver/pkg/util/wsstream",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/endpoints/openapi_test",
    srcs = ["k8s.io/apiserver/pkg/endpoints/openapi/openapi_test.go"],
    library = ":k8s.io/apiserver/pkg/endpoints/openapi",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/openapi",
    srcs = ["k8s.io/apiserver/pkg/endpoints/openapi/openapi.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/util/trie",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/endpoints/testing",
    srcs = [
        "k8s.io/apiserver/pkg/endpoints/testing/types.generated.go",
        "k8s.io/apiserver/pkg/endpoints/testing/types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry",
    srcs = ["k8s.io/apiserver/pkg/registry/doc.go"],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry/generic",
    srcs = [
        "k8s.io/apiserver/pkg/registry/generic/doc.go",
        "k8s.io/apiserver/pkg/registry/generic/matcher.go",
        "k8s.io/apiserver/pkg/registry/generic/options.go",
        "k8s.io/apiserver/pkg/registry/generic/storage_decorator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend/factory",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/registry/generic/registry_test",
    srcs = [
        "k8s.io/apiserver/pkg/registry/generic/registry/decorated_watcher_test.go",
        "k8s.io/apiserver/pkg/registry/generic/registry/store_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/registry/generic/registry",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/selection",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing",
        "//vendor:k8s.io/apiserver/pkg/storage/names",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend/factory",
        "//vendor:k8s.io/apiserver/pkg/storage/testing",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry/generic/registry",
    srcs = [
        "k8s.io/apiserver/pkg/registry/generic/registry/decorated_watcher.go",
        "k8s.io/apiserver/pkg/registry/generic/registry/doc.go",
        "k8s.io/apiserver/pkg/registry/generic/registry/storage_factory.go",
        "k8s.io/apiserver/pkg/registry/generic/registry/store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/validation/path",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/errors",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend/factory",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/registry/generic/rest_test",
    srcs = [
        "k8s.io/apiserver/pkg/registry/generic/rest/proxy_test.go",
        "k8s.io/apiserver/pkg/registry/generic/rest/response_checker_test.go",
        "k8s.io/apiserver/pkg/registry/generic/rest/streamer_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/registry/generic/rest",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:golang.org/x/net/websocket",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/features",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
        "//vendor:k8s.io/apiserver/pkg/util/proxy",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry/generic/rest",
    srcs = [
        "k8s.io/apiserver/pkg/registry/generic/rest/doc.go",
        "k8s.io/apiserver/pkg/registry/generic/rest/proxy.go",
        "k8s.io/apiserver/pkg/registry/generic/rest/response_checker.go",
        "k8s.io/apiserver/pkg/registry/generic/rest/streamer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mxk/go-flowrate/flowrate",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apiserver/pkg/features",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
        "//vendor:k8s.io/apiserver/pkg/util/proxy",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/registry/rest_test",
    srcs = ["k8s.io/apiserver/pkg/registry/rest/meta_test.go"],
    library = ":k8s.io/apiserver/pkg/registry/rest",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/uuid",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry/rest",
    srcs = [
        "k8s.io/apiserver/pkg/registry/rest/create.go",
        "k8s.io/apiserver/pkg/registry/rest/delete.go",
        "k8s.io/apiserver/pkg/registry/rest/doc.go",
        "k8s.io/apiserver/pkg/registry/rest/export.go",
        "k8s.io/apiserver/pkg/registry/rest/meta.go",
        "k8s.io/apiserver/pkg/registry/rest/rest.go",
        "k8s.io/apiserver/pkg/registry/rest/update.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/validation",
        "//vendor:k8s.io/apimachinery/pkg/api/validation/path",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/uuid",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/storage/names",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/registry/rest/resttest",
    srcs = ["k8s.io/apiserver/pkg/registry/rest/resttest/resttest.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/validation/path",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/server_test",
    srcs = [
        "k8s.io/apiserver/pkg/server/genericapiserver_test.go",
        "k8s.io/apiserver/pkg/server/resource_config_test.go",
        "k8s.io/apiserver/pkg/server/serve_test.go",
        "k8s.io/apiserver/pkg/server/storage_factory_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/server",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/generated/openapi:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/install",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server/options",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/server/mux_test",
    srcs = ["k8s.io/apiserver/pkg/server/mux/container_test.go"],
    library = ":k8s.io/apiserver/pkg/server/mux",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/mux",
    srcs = [
        "k8s.io/apiserver/pkg/server/mux/container.go",
        "k8s.io/apiserver/pkg/server/mux/doc.go",
        "k8s.io/apiserver/pkg/server/mux/pathrecorder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/server/openapi_test",
    srcs = ["k8s.io/apiserver/pkg/server/openapi/openapi_test.go"],
    library = ":k8s.io/apiserver/pkg/server/openapi",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/openapi",
    srcs = [
        "k8s.io/apiserver/pkg/server/openapi/doc.go",
        "k8s.io/apiserver/pkg/server/openapi/openapi.go",
        "k8s.io/apiserver/pkg/server/openapi/util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
        "//vendor:k8s.io/apiserver/pkg/server/mux",
        "//vendor:k8s.io/apiserver/pkg/util/trie",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/server/routes",
    srcs = [
        "k8s.io/apiserver/pkg/server/routes/doc.go",
        "k8s.io/apiserver/pkg/server/routes/index.go",
        "k8s.io/apiserver/pkg/server/routes/metrics.go",
        "k8s.io/apiserver/pkg/server/routes/openapi.go",
        "k8s.io/apiserver/pkg/server/routes/profiling.go",
        "k8s.io/apiserver/pkg/server/routes/swagger.go",
        "k8s.io/apiserver/pkg/server/routes/swaggerui.go",
        "k8s.io/apiserver/pkg/server/routes/version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/elazarl/go-bindata-assetfs",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/server/mux",
        "//vendor:k8s.io/apiserver/pkg/server/openapi",
        "//vendor:k8s.io/apiserver/pkg/server/routes/data/swagger",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/metrics",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/errors",
    srcs = [
        "k8s.io/apiserver/pkg/storage/errors/doc.go",
        "k8s.io/apiserver/pkg/storage/errors/storage.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/storage",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/etcd_test",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd/api_object_versioner_test.go",
        "k8s.io/apiserver/pkg/storage/etcd/etcd_helper_test.go",
        "k8s.io/apiserver/pkg/storage/etcd/etcd_watcher_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/storage/etcd",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/etcdtest",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing",
        "//vendor:k8s.io/apiserver/pkg/storage/testing",
        "//vendor:k8s.io/apiserver/pkg/storage/tests",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd/api_object_versioner.go",
        "k8s.io/apiserver/pkg/storage/etcd/doc.go",
        "k8s.io/apiserver/pkg/storage/etcd/etcd_helper.go",
        "k8s.io/apiserver/pkg/storage/etcd/etcd_watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/metrics",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/util",
        "//vendor:k8s.io/apiserver/pkg/util/cache",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd/etcdtest",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd/etcdtest/doc.go",
        "k8s.io/apiserver/pkg/storage/etcd/etcdtest/etcdtest.go",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd/testing",
    srcs = ["k8s.io/apiserver/pkg/storage/etcd/testing/utils.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http",
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:github.com/coreos/etcd/pkg/testutil",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/etcdtest",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing/testingcert",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd/testing/testingcert",
    srcs = ["k8s.io/apiserver/pkg/storage/etcd/testing/testingcert/certificates.go"],
    tags = ["automanaged"],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/etcd/util_test",
    srcs = ["k8s.io/apiserver/pkg/storage/etcd/util/etcd_util_test.go"],
    library = ":k8s.io/apiserver/pkg/storage/etcd/util",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd/util",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd/util/doc.go",
        "k8s.io/apiserver/pkg/storage/etcd/util/etcd_util.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/coreos/etcd/client"],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/etcd3_test",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd3/compact_test.go",
        "k8s.io/apiserver/pkg/storage/etcd3/store_test.go",
        "k8s.io/apiserver/pkg/storage/etcd3/watcher_test.go",
    ],
    library = ":k8s.io/apiserver/pkg/storage/etcd3",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/storage",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/etcd3",
    srcs = [
        "k8s.io/apiserver/pkg/storage/etcd3/compact.go",
        "k8s.io/apiserver/pkg/storage/etcd3/event.go",
        "k8s.io/apiserver/pkg/storage/etcd3/store.go",
        "k8s.io/apiserver/pkg/storage/etcd3/watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/storagebackend/factory_test",
    srcs = ["k8s.io/apiserver/pkg/storage/storagebackend/factory/tls_test.go"],
    library = ":k8s.io/apiserver/pkg/storage/storagebackend/factory",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing/testingcert",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/storagebackend/factory",
    srcs = [
        "k8s.io/apiserver/pkg/storage/storagebackend/factory/etcd2.go",
        "k8s.io/apiserver/pkg/storage/storagebackend/factory/etcd3.go",
        "k8s.io/apiserver/pkg/storage/storagebackend/factory/factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd3",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/testing",
    srcs = [
        "k8s.io/apiserver/pkg/storage/testing/types.generated.go",
        "k8s.io/apiserver/pkg/storage/testing/types.go",
        "k8s.io/apiserver/pkg/storage/testing/utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apiserver/pkg/storage",
    ],
)

go_test(
    name = "k8s.io/apiserver/pkg/storage/tests_test",
    srcs = ["k8s.io/apiserver/pkg/storage/tests/cacher_test.go"],
    library = ":k8s.io/apiserver/pkg/storage/tests",
    tags = ["automanaged"],
    deps = [
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/etcdtest",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/testing",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd3",
        "//vendor:k8s.io/client-go/pkg/api/install",
    ],
)

go_library(
    name = "k8s.io/apiserver/pkg/storage/tests",
    srcs = ["k8s.io/apiserver/pkg/storage/tests/utils.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/client-go/pkg/api/install",
    ],
)
