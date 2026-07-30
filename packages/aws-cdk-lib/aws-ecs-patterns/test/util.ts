import type { IConstruct } from 'constructs';
import * as cdk from '../../core';

/**
 * Acknowledge validation rules that fire on aws-ecs-patterns test infrastructure patterns.
 * Call on the stack (or any ancestor scope) to suppress known false positives.
 */
export function acknowledgeTestValidationRules(scope: IConstruct) {
  cdk.Validations.of(scope).acknowledge(
    { id: 'CloudFormation-Validate::W3697', reason: 'LaunchConfiguration used intentionally in tests' },
    { id: 'CloudFormation-Validate::E3047', reason: 'Intentionally testing invalid Fargate CPU/Memory combinations' },
    { id: 'CloudFormation-Validate::W9013', reason: 'Duplicate subnets are intentional in tests' },
    { id: 'CloudFormation-Validate::W9002', reason: 'Deprecated properties used intentionally in tests' },
  );
}
