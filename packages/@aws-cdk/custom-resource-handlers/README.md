# Custom Resource Handlers

This package contains the following custom resource handlers:

## Stable

- aws-certificatemanager/dns-validated-certificate-handler
- aws-cloudfront/edge-function
- aws-dynamodb/replica-handler
- aws-ec2/restrict-default-security-group-handler
- aws-ecr/auto-delete-images-handler
- aws-ecs/lambda-source
- aws-eks/custom-resource-handler
- aws-eks/kubectl-handler
- aws-eks-v2/kubectl-handler
- aws-events-targets/aws-api-handler
- aws-iam/oidc-handler
- aws-logs/log-retention-handler
- aws-route53/cross-account-zone-delegation-handler
- aws-route53/delete-existing-record-set-handler
- aws-s3/auto-delete-objects-handler
- aws-s3/notifications-resource-handler
- aws-s3-deployment/bucket-deployment-handler
- aws-ses/drop-spam-handler
- aws-stepfunctions-tasks/cross-region-aws-sdk-handler
- aws-stepfunctions-tasks/eval-nodejs-handler
- aws-stepfunctions-tasks/role-policy-handler
- aws-synthetics/auto-delete-underlying-resources-handler
- custom-resources/aws-custom-resource-handler
- pipelines/approve-lambda
- triggers/lambda

These handlers are copied into `aws-cdk-lib/custom-resource-handlers` at build time
and included as part of the `aws-cdk-lib` package.

## Experimental

- aws-amplify-alpha/asset-deployment-handler
- aws-glue-alpha/partition-index-handler
- aws-redshift-alpha/asset-deployment-handler

These handlers are excluded from `aws-cdk-lib/custom-resource-handlers` and are individually
copied into their respective `-alpha` packages at build time. When an `-alpha` package is
stabilized, part of the stabilization process **must** be to remove `-alpha` from the folder
name, so that it is included in `aws-cdk-lib`.

`*/generated.ts` files are not supported for alpha modules due to import paths that only work for stable modules in `aws-cdk-lib`. These files must be added to `custom-resources-framework/config.ts` as `ComponentType.NO_OP`.

## Nodejs Entrypoint

This package also includes `nodejs-entrypoint.ts`, which is a wrapper that talks to
CloudFormation for you. At build time, `nodejs-entrypoint.js` is bundled into the
`.js` file of the custom resource handler, creating one `index.js` file. This file
is then either copied into a CDK asset or copied as inline code directly in the
CloudFormation template.

## Future Work

In the future this package will contain all custom resources and their tests.
