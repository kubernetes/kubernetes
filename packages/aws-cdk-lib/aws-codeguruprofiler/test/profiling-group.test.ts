import { Template } from '../../assertions';
import { AccountRootPrincipal, Role } from '../../aws-iam';
import { Stack } from '../../core';
import { ProfilingGroup, ComputePlatform } from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('profiling group', () => {
  test('attach read permission to Profiling group via fromProfilingGroupArn', () => {
    const stack = new Stack();
    // dummy role to test out read permissions on ProfilingGroup
    const readAppRole = new Role(stack, 'ReadAppRole', {
      assumedBy: new AccountRootPrincipal(),
    });

    const profilingGroup = ProfilingGroup.fromProfilingGroupArn(stack, 'MyProfilingGroup', 'arn:aws:codeguru-profiler:us-east-1:123456789012:profilingGroup/MyAwesomeProfilingGroup');
    expect(profilingGroup.profilingGroupName).toBe('MyAwesomeProfilingGroup');
    expect(profilingGroup.profilingGroupArn).toBe('arn:aws:codeguru-profiler:us-east-1:123456789012:profilingGroup/MyAwesomeProfilingGroup');
    expect(profilingGroup.env.region).toBe('us-east-1');

    profilingGroup.grantRead(readAppRole);

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'ReadAppRole52FE6317': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'ReadAppRoleDefaultPolicy4BB8955C': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    'codeguru-profiler:GetProfile',
                    'codeguru-profiler:DescribeProfilingGroup',
                  ],
                  'Effect': 'Allow',
                  'Resource': 'arn:aws:codeguru-profiler:us-east-1:123456789012:profilingGroup/MyAwesomeProfilingGroup',
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'ReadAppRoleDefaultPolicy4BB8955C',
            'Roles': [
              {
                'Ref': 'ReadAppRole52FE6317',
              },
            ],
          },
        },
      },
    });
  });

  test('attach publish permission to Profiling group via fromProfilingGroupName', () => {
    const stack = new Stack();
    // dummy role to test out publish permissions on ProfilingGroup
    const publishAppRole = new Role(stack, 'PublishAppRole', {
      assumedBy: new AccountRootPrincipal(),
    });

    const profilingGroup = ProfilingGroup.fromProfilingGroupName(stack, 'MyProfilingGroup', 'MyAwesomeProfilingGroup');
    profilingGroup.grantPublish(publishAppRole);

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'PublishAppRole9FEBD682': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'PublishAppRoleDefaultPolicyCA1E15C3': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    'codeguru-profiler:ConfigureAgent',
                    'codeguru-profiler:PostAgentProfile',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        {
                          'Ref': 'AWS::Partition',
                        },
                        ':codeguru-profiler:',
                        {
                          'Ref': 'AWS::Region',
                        },
                        ':',
                        {
                          'Ref': 'AWS::AccountId',
                        },
                        ':profilingGroup/MyAwesomeProfilingGroup',
                      ],
                    ],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'PublishAppRoleDefaultPolicyCA1E15C3',
            'Roles': [
              {
                'Ref': 'PublishAppRole9FEBD682',
              },
            ],
          },
        },
      },
    });
  });

  test('use name specified via fromProfilingGroupName', () => {
    const stack = new Stack();

    const profilingGroup = ProfilingGroup.fromProfilingGroupName(stack, 'MyProfilingGroup', 'MyAwesomeProfilingGroup');
    expect(profilingGroup.profilingGroupName).toEqual('MyAwesomeProfilingGroup');
  });

  test('default profiling group', () => {
    const stack = new Stack();
    new ProfilingGroup(stack, 'MyProfilingGroup', {
      profilingGroupName: 'MyAwesomeProfilingGroup',
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProfilingGroup829F0507': {
          'Type': 'AWS::CodeGuruProfiler::ProfilingGroup',
          'Properties': {
            'ProfilingGroupName': 'MyAwesomeProfilingGroup',
          },
        },
      },
    });
  });

  test('allows setting its ComputePlatform', () => {
    const stack = new Stack();
    new ProfilingGroup(stack, 'MyProfilingGroup', {
      profilingGroupName: 'MyAwesomeProfilingGroup',
      computePlatform: ComputePlatform.AWS_LAMBDA,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CodeGuruProfiler::ProfilingGroup', {
      'ComputePlatform': 'AWSLambda',
    });
  });

  test('default profiling group without name', () => {
    const stack = new Stack();
    new ProfilingGroup(stack, 'MyProfilingGroup', {
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProfilingGroup829F0507': {
          'Type': 'AWS::CodeGuruProfiler::ProfilingGroup',
          'Properties': {
            'ProfilingGroupName': 'MyProfilingGroup',
          },
        },
      },
    });
  });

  test('default profiling group without name when name exceeding limit is generated', () => {
    const stack = new Stack();
    new ProfilingGroup(stack, 'MyProfilingGroupWithAReallyLongProfilingGroupNameThatExceedsTheLimitOfProfilingGroupNameSize_InOrderToDoSoTheNameMustBeGreaterThanTwoHundredAndFiftyFiveCharacters_InSuchCasesWePickUpTheFirstOneTwentyCharactersFromTheBeginningAndTheEndAndConcatenateThemToGetTheIdentifier', {
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProfilingGroupWithAReallyLongProfilingGroupNameThatExceedsTheLimitOfProfilingGroupNameSizeInOrderToDoSoTheNameMustBeGreaterThanTwoHundredAndFiftyFiveCharactersInSuchCasesWePickUpTheFirstOneTwentyCharactersFromTheBeginningAndTheEndAndConca4B39908C': {
          'Type': 'AWS::CodeGuruProfiler::ProfilingGroup',
          'Properties': {
            'ProfilingGroupName': 'MyProfilingGroupWithAReallyLongProfilingGroupNameThatExceedsTheLimitOfProfilingGroupNameSizeInOrderToDoSoTheNameMustBeGrnTwoHundredAndFiftyFiveCharactersInSuchCasesWePickUpTheFirstOneTwentyCharactersFromTheBeginningAndTheEndAndConca2FE009B0',
          },
        },
      },
    });
  });

  test('grant publish permissions profiling group', () => {
    const stack = new Stack();
    const profilingGroup = new ProfilingGroup(stack, 'MyProfilingGroup', {
      profilingGroupName: 'MyAwesomeProfilingGroup',
    });
    const publishAppRole = new Role(stack, 'PublishAppRole', {
      assumedBy: new AccountRootPrincipal(),
    });

    profilingGroup.grantPublish(publishAppRole);

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProfilingGroup829F0507': {
          'Type': 'AWS::CodeGuruProfiler::ProfilingGroup',
          'Properties': {
            'ProfilingGroupName': 'MyAwesomeProfilingGroup',
          },
        },
        'PublishAppRole9FEBD682': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'PublishAppRoleDefaultPolicyCA1E15C3': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    'codeguru-profiler:ConfigureAgent',
                    'codeguru-profiler:PostAgentProfile',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::GetAtt': [
                      'MyProfilingGroup829F0507',
                      'Arn',
                    ],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'PublishAppRoleDefaultPolicyCA1E15C3',
            'Roles': [
              {
                'Ref': 'PublishAppRole9FEBD682',
              },
            ],
          },
        },
      },
    });
  });

  test('grant read permissions profiling group', () => {
    const stack = new Stack();
    const profilingGroup = new ProfilingGroup(stack, 'MyProfilingGroup', {
      profilingGroupName: 'MyAwesomeProfilingGroup',
    });
    const readAppRole = new Role(stack, 'ReadAppRole', {
      assumedBy: new AccountRootPrincipal(),
    });

    profilingGroup.grantRead(readAppRole);

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyProfilingGroup829F0507': {
          'Type': 'AWS::CodeGuruProfiler::ProfilingGroup',
          'Properties': {
            'ProfilingGroupName': 'MyAwesomeProfilingGroup',
          },
        },
        'ReadAppRole52FE6317': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
        'ReadAppRoleDefaultPolicy4BB8955C': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': [
                    'codeguru-profiler:GetProfile',
                    'codeguru-profiler:DescribeProfilingGroup',
                  ],
                  'Effect': 'Allow',
                  'Resource': {
                    'Fn::GetAtt': [
                      'MyProfilingGroup829F0507',
                      'Arn',
                    ],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'PolicyName': 'ReadAppRoleDefaultPolicy4BB8955C',
            'Roles': [
              {
                'Ref': 'ReadAppRole52FE6317',
              },
            ],
          },
        },
      },
    });
  });
});
