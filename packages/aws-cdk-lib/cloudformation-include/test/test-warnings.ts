import type { IConstruct } from 'constructs';
import { Validations } from '../../core/lib/validation';

export function acknowledgeTestWarnings(construct: IConstruct) {
  Validations.of(construct).acknowledge(...[
    'E1041',
    'E2001',
    'E3045',
    'F0005',
    'F0013',
    // This fixture intentionally uses a CDK intrinsic object as an operand where Fn::Equals expects a scalar.
    // Validator reports this invalid Fn::Equals operand as E8003 instead of the broader F0014.
    'E8003',
    'F1029',
    'F2002',
    'F2012',
    'F2015',
    'F3002',
    'F3002',
    'F3003',
    'F3004',
    'F3012',
    'F3014',
    'W1020',
    // This fixture intentionally contains an unused Fn::Sub variable to test that
    // cloudformation-include preserves the original expression unchanged.
    'W1019',
    // These fixtures intentionally contain literal AWS::Region and AWS::NoValue strings.
    // The tests exercise template ingestion, not W1054's recommendation to replace them with Ref.
    'W1054',
    'W1102',
    'W2531',
    'W3011',
    'W3045',
    'W9003',
    'E3639',
    'F0018',
    'F3016',
    'E3639',
    'E3001',
    'E8001',
    'E3055',
    'E3016',
  ].map(code => ({
    id: `CloudFormation-Validate::${code}`,
    reason: 'These tests validate the ingestion of templates into CDK. Whether the properties are valid or not is irrelevant',
  })));
}
