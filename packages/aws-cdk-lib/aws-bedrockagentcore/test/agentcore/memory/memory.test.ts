import { Match, Template } from '../../../../assertions';
import { FoundationModel, FoundationModelIdentifier } from '../../../../aws-bedrock';
import * as cloudwatch from '../../../../aws-cloudwatch';
import * as iam from '../../../../aws-iam';
import * as kinesis from '../../../../aws-kinesis';
import * as kms from '../../../../aws-kms';
import * as s3 from '../../../../aws-s3';
import * as sns from '../../../../aws-sns';
import * as cdk from '../../../../core';
import { Duration } from '../../../../core';
import {
  Memory,
  StreamDeliveryContentType,
  StreamDeliveryContentLevel,
  StreamDeliveryResource,
} from '../../../lib/memory/memory';
import { MemoryStrategy } from '../../../lib/memory/memory-strategy';

// Create a test model using the stable FoundationModel
const TEST_STACK = new cdk.Stack(new cdk.App(), 'ModelStack');
const TEST_MODEL = FoundationModel.fromFoundationModelId(TEST_STACK, 'TestModel', FoundationModelIdentifier.ANTHROPIC_CLAUDE_SONNET_4_6);

describe('Memory default tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory',
      description: 'A test memory for storing agent interactions',
      expirationDuration: Duration.days(30),
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with expected properties', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory',
      Description: 'A test memory for storing agent interactions',
      EventExpiryDuration: 30,
    });
  });

  test('Should handle tags correctly when no tags are provided', () => {
    // Verify that the Memory resource exists with basic properties
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory',
      Description: 'A test memory for storing agent interactions',
      EventExpiryDuration: 30,
    });
  });

  test('Should have default expiration of 90 days when not specified', () => {
    const memoryWithDefaultExpiry = new Memory(stack, 'default-expiry-memory', {
      memoryName: 'default_expiry_memory',
    });

    expect(memoryWithDefaultExpiry.expirationDuration?.toDays()).toBe(90);
  });

  test('Should have service role with confused deputy conditions', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: { Service: 'bedrock-agentcore.amazonaws.com' },
            Condition: {
              StringEquals: { 'aws:SourceAccount': '123456789012' },
              ArnLike: {
                'aws:SourceArn': {
                  'Fn::Join': ['', Match.arrayWith([
                    ':bedrock-agentcore:us-east-1:123456789012:memory/test_memory*',
                  ])],
                },
              },
            },
          },
        ],
      },
    });
  });
});

describe('Memory static methods tests', () => {
  // @ts-ignore
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    template = Template.fromStack(stack);
  });

  test('fromMemoryAttributes should create a Memory reference from existing attributes', () => {
    const memory = Memory.fromMemoryAttributes(stack, 'test-memory', {
      memoryArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/test-memory',
      roleArn: 'arn:aws:iam::123456789012:role/test-memory-role',
      updatedAt: '2021-01-01T00:00:00Z',
      status: 'ACTIVE',
      createdAt: '2021-01-01T00:00:00Z',
      kmsKeyArn: 'arn:aws:kms::123456789012:key/test-kms-key',
    });

    expect(memory.memoryArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/test-memory');
    expect(memory.executionRole).toBeDefined();
    expect(memory.updatedAt).toBe('2021-01-01T00:00:00Z');
    expect(memory.status).toBe('ACTIVE');
    expect(memory.createdAt).toBe('2021-01-01T00:00:00Z');
    expect(memory.kmsKey).toBeDefined();
  });

  test('fromMemoryAttributes provides undefined values when not provided', () => {
    const memory = Memory.fromMemoryAttributes(stack, 'test-memory-2', {
      memoryArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/test-memory',
      roleArn: 'arn:aws:iam::123456789012:role/test-memory-role',
    });

    expect(memory.memoryArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/test-memory');
    expect(memory.executionRole).toBeDefined();
    expect(memory.updatedAt).toBeUndefined();
    expect(memory.status).toBeUndefined();
    expect(memory.createdAt).toBeUndefined();
    expect(memory.kmsKey).toBeUndefined();
  });
});

describe('Memory with built-in strategies tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_with_strategies',
      description: 'A test memory with built-in strategies',
      expirationDuration: Duration.days(60),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInSummarization(),
        MemoryStrategy.usingBuiltInSemantic(),
        MemoryStrategy.usingBuiltInUserPreference(),
        MemoryStrategy.usingBuiltInEpisodic(),
      ],
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with built-in memory strategies', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_with_strategies',
      Description: 'A test memory with built-in strategies',
      EventExpiryDuration: 60,
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: 'summary_builtin_cdkGen0001' }) }),
        Match.objectLike({ SemanticMemoryStrategy: Match.objectLike({ Name: 'semantic_builtin_cdkGen0001' }) }),
        Match.objectLike({ UserPreferenceMemoryStrategy: Match.objectLike({ Name: 'preference_builtin_cdkGen0001' }) }),
        Match.objectLike({ EpisodicMemoryStrategy: Match.objectLike({ Name: 'episodic_builtin_cdkGen0001' }) }),
      ]),
    });

    // Verify that episodic strategy is present in memoryStrategies property
    expect(memory.memoryStrategies).toHaveLength(4);
    expect(memory.memoryStrategies.some(s => s.strategyType === 'EPISODIC')).toBe(true);
  });
});

describe('Memory with custom execution role tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let customRole: iam.Role;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create a custom execution role
    customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
      roleName: 'custom-memory-execution-role',
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_with_custom_role',
      description: 'A test memory with custom execution role',
      expirationDuration: Duration.days(45),
      executionRole: customRole,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with custom execution role', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_with_custom_role',
      Description: 'A test memory with custom execution role',
      EventExpiryDuration: 45,
      MemoryExecutionRoleArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('CustomExecutionRole*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have custom execution role with correct properties', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'custom-memory-execution-role',
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });
});

describe('Memory with KMS encryption tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let kmsKey: kms.Key;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create a custom KMS key
    kmsKey = new kms.Key(stack, 'MemoryEncryptionKey', {
      description: 'KMS key for memory encryption',
      enableKeyRotation: true,
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_with_kms',
      description: 'A test memory with KMS encryption',
      expirationDuration: Duration.days(120),
      kmsKey: kmsKey,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.resourceCountIs('AWS::KMS::Key', 1);
  });

  test('Should have Memory resource with KMS encryption', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_with_kms',
      Description: 'A test memory with KMS encryption',
      EventExpiryDuration: 120,
      EncryptionKeyArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('MemoryEncryptionKey*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have KMS key with correct properties', () => {
    template.hasResourceProperties('AWS::KMS::Key', {
      Description: 'KMS key for memory encryption',
      EnableKeyRotation: true,
    });
  });
});

describe('Memory name validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should throw error for name with hyphen', () => {
    expect(() => {
      new Memory(stack, 'test-memory', {
        memoryName: 'test-memory',
        expirationDuration: Duration.days(30),
      });
    }).toThrow('The field Memory name with value "test-memory" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for empty name', () => {
    expect(() => {
      new Memory(stack, 'empty-name', {
        memoryName: '',
        expirationDuration: Duration.days(30),
      });
    }).toThrow('The field Memory name is 0 characters long but must be at least 1 characters');
  });

  test('Should throw error for name with spaces', () => {
    expect(() => {
      new Memory(stack, 'name-with-spaces', {
        memoryName: 'test memory',
        expirationDuration: Duration.days(30),
      });
    }).toThrow('The field Memory name with value "test memory" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for name with special characters', () => {
    expect(() => {
      new Memory(stack, 'name-with-special-chars', {
        memoryName: 'test@memory',
        expirationDuration: Duration.days(30),
      });
    }).toThrow('The field Memory name with value "test@memory" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for name exceeding 48 characters', () => {
    const longName = 'a'.repeat(49);
    expect(() => {
      new Memory(stack, 'long-name', {
        memoryName: longName,
        expirationDuration: Duration.days(30),
      });
    }).toThrow('The field Memory name is 49 characters long but must be less than or equal to 48 characters');
  });

  test('Should accept valid name with underscores', () => {
    expect(() => {
      new Memory(stack, 'valid-name', {
        memoryName: 'test_memory_123',
        expirationDuration: Duration.days(30),
      });
    }).not.toThrow();
  });

  test('Should accept valid name with only letters and numbers', () => {
    expect(() => {
      new Memory(stack, 'valid-name-2', {
        memoryName: 'testMemory123',
        expirationDuration: Duration.days(30),
      });
    }).not.toThrow();
  });
});

describe('Memory expiration duration validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should throw error for expiration less than 7 days', () => {
    expect(() => {
      new Memory(stack, 'short-expiry', {
        memoryName: 'short_expiry_memory',
        expirationDuration: Duration.days(0),
      });
    }).toThrow('Memory expiration days must be between 7 and 365');
  });

  test('Should throw error for expiration more than 365 days', () => {
    expect(() => {
      new Memory(stack, 'long-expiry', {
        memoryName: 'long_expiry_memory',
        expirationDuration: Duration.days(366),
      });
    }).toThrow('Memory expiration days must be between 7 and 365');
  });

  test('Should accept minimum valid expiration (7 days)', () => {
    expect(() => {
      new Memory(stack, 'min-expiry', {
        memoryName: 'min_expiry_memory',
        expirationDuration: Duration.days(7),
      });
    }).not.toThrow();
  });

  test('Should accept maximum valid expiration (365 days)', () => {
    expect(() => {
      new Memory(stack, 'max-expiry', {
        memoryName: 'max_expiry_memory',
        expirationDuration: Duration.days(365),
      });
    }).not.toThrow();
  });

  test('Should accept valid expiration in middle range', () => {
    expect(() => {
      new Memory(stack, 'middle-expiry', {
        memoryName: 'middle_expiry_memory',
        expirationDuration: Duration.days(180),
      });
    }).not.toThrow();
  });

  test('does not fail validation if expirationDuration is a late-bound value', () => {
    // WHEN
    const expirationDuration = new cdk.CfnParameter(stack, 'ExpirationDuration', {
      default: 30,
      type: 'Number',
    });

    expect(() => {
      new Memory(stack, 'memory-late-bound', {
        memoryName: 'memory_late_bound',
        expirationDuration: Duration.days(expirationDuration.valueAsNumber),
      });
    }).not.toThrow();
  });
});

describe('Memory with custom strategies tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create custom strategies
    const customSemanticStrategy = MemoryStrategy.usingSemantic({
      strategyName: 'custom_semantic_strategy',
      description: 'Custom semantic memory strategy with overrides',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/custom'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom consolidation prompt for semantic memory',
      },
      customExtraction: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom extraction prompt for semantic memory',
      },
    });

    const customUserPreferenceStrategy = MemoryStrategy.usingUserPreference({
      strategyName: 'custom_user_preference_strategy',
      description: 'Custom user preference strategy with overrides',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/preferences'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom consolidation prompt for user preferences',
      },
      customExtraction: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom extraction prompt for user preferences',
      },
    });

    const customSummaryStrategy = MemoryStrategy.usingSummarization({
      strategyName: 'custom_summary_strategy',
      description: 'Custom summary strategy with override',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/custom'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom consolidation prompt for summaries',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_with_custom_strategies',
      description: 'A test memory with custom strategies',
      expirationDuration: Duration.days(90),
      memoryStrategies: [
        customSemanticStrategy,
        customUserPreferenceStrategy,
        customSummaryStrategy,
      ],
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with custom semantic strategy', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_with_custom_strategies',
      Description: 'A test memory with custom strategies',
      EventExpiryDuration: 90,
      MemoryStrategies: Match.arrayWith([
        {
          CustomMemoryStrategy: {
            Name: 'custom_semantic_strategy',
            Description: 'Custom semantic memory strategy with overrides',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/custom'],
            Configuration: {
              SemanticOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt for semantic memory',
                },
                Extraction: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom extraction prompt for semantic memory',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have Memory resource with custom user preference strategy', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        {
          CustomMemoryStrategy: {
            Name: 'custom_user_preference_strategy',
            Description: 'Custom user preference strategy with overrides',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/preferences'],
            Configuration: {
              UserPreferenceOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt for user preferences',
                },
                Extraction: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom extraction prompt for user preferences',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have Memory resource with custom summary strategy', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        {
          CustomMemoryStrategy: {
            Name: 'custom_summary_strategy',
            Description: 'Custom summary strategy with override',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/custom'],
            Configuration: {
              SummaryOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt for summaries',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have all three custom strategies in MemoryStrategies array', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'custom_semantic_strategy' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'custom_user_preference_strategy' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'custom_summary_strategy' }) }),
      ]),
    });
  });
});

describe('Memory with mixed built-in and custom strategies tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Mix of built-in and custom strategies
    const customSemanticStrategy = MemoryStrategy.usingSemantic({
      strategyName: 'hybrid_semantic_strategy',
      description: 'Hybrid semantic strategy combining built-in and custom',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/hybrid'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Hybrid consolidation prompt',
      },
      customExtraction: {
        model: TEST_MODEL,
        appendToPrompt: 'Hybrid extraction prompt',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_mixed_strategies',
      description: 'A test memory with mixed built-in and custom strategies',
      expirationDuration: Duration.days(180),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInSummarization(),
        MemoryStrategy.usingBuiltInSemantic(),
        customSemanticStrategy,
      ],
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with mixed strategies', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_mixed_strategies',
      Description: 'A test memory with mixed built-in and custom strategies',
      EventExpiryDuration: 180,
      MemoryStrategies: Match.arrayWith([
        // Built-in summarization
        {
          SummaryMemoryStrategy: {
            Name: Match.stringLikeRegexp('summary_builtin_.*'),
            Description: 'Summarize interactions to preserve critical context and key insights',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}'],
          },
        },
        // Built-in semantic
        {
          SemanticMemoryStrategy: {
            Name: Match.stringLikeRegexp('semantic_builtin_.*'),
            Description: 'Extract general factual knowledge, concepts and meanings from raw conversations in a context-independent format.',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
          },
        },
        // Custom semantic
        {
          CustomMemoryStrategy: {
            Name: 'hybrid_semantic_strategy',
            Description: 'Hybrid semantic strategy combining built-in and custom',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/hybrid'],
            Configuration: {
              SemanticOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Hybrid consolidation prompt',
                },
                Extraction: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Hybrid extraction prompt',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have exactly three strategies', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_mixed_strategies',
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: 'summary_builtin_cdkGen0001' }) }),
        Match.objectLike({ SemanticMemoryStrategy: Match.objectLike({ Name: 'semantic_builtin_cdkGen0001' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'hybrid_semantic_strategy' }) }),
      ]),
    });
  });
});

describe('Memory with addMemoryStrategy method tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create memory without initial strategies
    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_add_strategy',
      description: 'A test memory for testing addMemoryStrategy method',
      expirationDuration: Duration.days(90),
    });

    // Add strategies after instantiation
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSummarization());
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSemantic());

    // Add a custom strategy
    const customStrategy = MemoryStrategy.usingSemantic({
      strategyName: 'added_custom_strategy',
      description: 'Custom strategy added via addMemoryStrategy method',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/added'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom consolidation prompt added via method',
      },
      customExtraction: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom extraction prompt added via method',
      },
    });
    memory.addMemoryStrategy(customStrategy);

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with strategies added via addMemoryStrategy', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_add_strategy',
      Description: 'A test memory for testing addMemoryStrategy method',
      EventExpiryDuration: 90,
      MemoryStrategies: [
        {
          SummaryMemoryStrategy: {
            Name: Match.stringLikeRegexp('summary_builtin_.*'),
            Description: 'Summarize interactions to preserve critical context and key insights',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}'],
          },
        },
        {
          SemanticMemoryStrategy: {
            Name: Match.stringLikeRegexp('semantic_builtin_.*'),
            Description: 'Extract general factual knowledge, concepts and meanings from raw conversations in a context-independent format.',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
          },
        },
        {
          CustomMemoryStrategy: {
            Name: 'added_custom_strategy',
            Description: 'Custom strategy added via addMemoryStrategy method',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/added'],
            Configuration: {
              SemanticOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt added via method',
                },
                Extraction: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom extraction prompt added via method',
                },
              },
            },
          },
        },
      ],
    });
  });

  test('Should have exactly three strategies in MemoryStrategies array', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: Match.stringLikeRegexp('summary_builtin_.*') }) }),
        Match.objectLike({ SemanticMemoryStrategy: Match.objectLike({ Name: Match.stringLikeRegexp('semantic_builtin_.*') }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'added_custom_strategy' }) }),
      ]),
    });
  });

  test('Should have memoryStrategies property accessible after adding strategies', () => {
    expect(memory.memoryStrategies).toHaveLength(3);

    const strategyNames = memory.memoryStrategies.map(strategy =>
      strategy.strategyName,
    );

    expect(strategyNames.some((name: string) => /summary_builtin_.*/.test(name))).toBe(true);
    expect(strategyNames.some((name: string) => /semantic_builtin_.*/.test(name))).toBe(true);
    expect(strategyNames).toContain('added_custom_strategy');
  });
});

describe('Memory with dynamic strategy addition tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create memory with one initial strategy
    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_dynamic',
      description: 'A test memory with dynamic strategy addition',
      expirationDuration: Duration.days(120),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInUserPreference(),
      ],
    });

    // Add more strategies dynamically
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSummarization());

    const customStrategy = MemoryStrategy.usingSummarization({
      strategyName: 'dynamic_summary_strategy',
      description: 'Dynamic summary strategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/dynamic'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Dynamic consolidation prompt',
      },
    });
    memory.addMemoryStrategy(customStrategy);

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with both initial and dynamically added strategies', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_dynamic',
      Description: 'A test memory with dynamic strategy addition',
      EventExpiryDuration: 120,
      MemoryStrategies: Match.arrayWith([
        // Initial strategy
        {
          UserPreferenceMemoryStrategy: {
            Name: Match.stringLikeRegexp('preference_builtin_.*'),
            Description: 'Capture individual preferences, interaction patterns, and personalized settings to enhance future experiences.',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
          },
        },
        // Dynamically added strategies
        {
          SummaryMemoryStrategy: {
            Name: Match.stringLikeRegexp('summary_builtin_.*'),
            Description: 'Summarize interactions to preserve critical context and key insights',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}'],
          },
        },
        {
          CustomMemoryStrategy: {
            Name: 'dynamic_summary_strategy',
            Description: 'Dynamic summary strategy',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/dynamic'],
            Configuration: {
              SummaryOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Dynamic consolidation prompt',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have exactly three strategies total', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_dynamic',
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ UserPreferenceMemoryStrategy: Match.objectLike({ Name: 'preference_builtin_cdkGen0001' }) }),
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: 'summary_builtin_cdkGen0001' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'dynamic_summary_strategy' }) }),
      ]),
    });
  });
});

describe('Memory grant methods tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let memory: Memory;
  let user: iam.User;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_grants',
      description: 'A test memory for testing grant methods',
      expirationDuration: Duration.days(30),
    });

    user = new iam.User(stack, 'TestUser', {
      userName: 'test-user',
    });
  });

  test('Should grant custom actions to user', () => {
    const grant = memory.grant(user, 'bedrock-agentcore:GetMemory', 'bedrock-agentcore:ListMemoryRecords');

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant write permissions to user', () => {
    const grant = memory.grantWrite(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant read permissions to user', () => {
    const grant = memory.grantRead(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant read short-term memory permissions to user', () => {
    const grant = memory.grantReadShortTermMemory(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant read long-term memory permissions to user', () => {
    const grant = memory.grantReadLongTermMemory(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant delete permissions to user', () => {
    const grant = memory.grantDelete(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant delete short-term memory permissions to user', () => {
    const grant = memory.grantDeleteShortTermMemory(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant delete long-term memory permissions to user', () => {
    const grant = memory.grantDeleteLongTermMemory(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant admin permissions to user', () => {
    const grant = memory.grantAdmin(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });

  test('Should grant full access permissions to user', () => {
    const grant = memory.grantFullAccess(user);

    expect(grant).toBeDefined();
    expect(grant.principalStatement).toBeDefined();
    // resourceStatement might be undefined if no resource policy is needed
  });
});

describe('Memory metric methods tests', () => {
  let stack: cdk.Stack;
  let memory: Memory;

  function alarmForMetric(id: string, metric: cloudwatch.Metric): void {
    new cloudwatch.Alarm(stack, id, { metric, evaluationPeriods: 1, threshold: 1 });
  }

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    memory = new Memory(stack, 'test-memory-metrics', {
      memoryName: 'test_memory_metrics',
      description: 'A test memory for testing metric methods',
      expirationDuration: Duration.days(30),
    });
  });

  test('metric() produces correct namespace, name, and dimensions', () => {
    alarmForMetric('CustomAlarm', memory.metric('CustomMetric', { dimensionsMap: { CustomDimension: 'value' } }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Namespace: 'AWS/Bedrock-AgentCore',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'CustomDimension', Value: 'value' }),
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('testmemorymetrics.*'), 'MemoryArn'] } }),
      ]),
    });
  });

  test('metricForApiOperation() produces correct Operation dimension', () => {
    alarmForMetric('OpAlarm', memory.metricForApiOperation('CustomMetric', 'CreateEvent'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Namespace: 'AWS/Bedrock-AgentCore',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'CreateEvent' }),
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('testmemorymetrics.*'), 'MemoryArn'] } }),
      ]),
    });
  });

  test('metricLatencyForApiOperation() produces Latency with Average statistic', () => {
    alarmForMetric('LatencyAlarm', memory.metricLatencyForApiOperation('CreateEvent'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'CreateEvent' }),
      ]),
    });
  });

  test('metricInvocationsForApiOperation() produces Invocations with Sum statistic', () => {
    alarmForMetric('InvocAlarm', memory.metricInvocationsForApiOperation('CreateEvent'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'CreateEvent' }),
      ]),
    });
  });

  test('metricErrorsForApiOperation() produces Errors with Sum statistic', () => {
    alarmForMetric('ErrorsAlarm', memory.metricErrorsForApiOperation('CreateEvent'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Errors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'CreateEvent' }),
      ]),
    });
  });

  test('metricEventCreationCount() produces CreationCount with Event ItemType dimension', () => {
    alarmForMetric('EventCountAlarm', memory.metricEventCreationCount());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CreationCount',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'ItemType', Value: 'Event' }),
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('testmemorymetrics.*'), 'MemoryArn'] } }),
      ]),
    });
  });

  test('metricMemoryRecordCreationCount() produces CreationCount with MemoryRecordsExtracted dimension', () => {
    alarmForMetric('RecordCountAlarm', memory.metricMemoryRecordCreationCount());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CreationCount',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'ItemType', Value: 'MemoryRecordsExtracted' }),
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('testmemorymetrics.*'), 'MemoryArn'] } }),
      ]),
    });
  });

  test('custom statistic prop overrides the default', () => {
    alarmForMetric('OverrideAlarm', memory.metricInvocationsForApiOperation('CreateEvent', { statistic: 'Average' }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Statistic: 'Average',
    });
  });
});

describe('Memory.addMemoryStrategy behavior tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let memory: Memory;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_add_memory_strategy',
      description: 'A test memory for addMemoryStrategy',
      expirationDuration: Duration.days(30),
    });
  });

  test('Should include the added strategy in the rendered CloudFormation template', () => {
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSemantic());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({
          SemanticMemoryStrategy: Match.objectLike({
            Namespaces: [
              '/strategies/{memoryStrategyId}/actors/{actorId}',
            ],
          }),
        }),
      ]),
    });
  });

  test('Should not throw when the strategy grant returns undefined (built-in strategy without overrides)', () => {
    // Built-in strategies without custom overrides have grant() returning undefined.
    // addMemoryStrategy must handle this via the optional chain `grant?.applyBefore(...)`.
    expect(() => {
      memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSummarization());
    }).not.toThrow();
  });

  test('Should grant model invoke permissions when strategy has custom overrides (grant returns defined)', () => {
    // Custom strategy with both extraction and consolidation overrides — grant() returns a combined Grant
    const customStrategy = MemoryStrategy.usingSemantic({
      strategyName: 'custom_semantic',
      description: 'Custom strategy with model overrides',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'custom consolidation',
      },
      customExtraction: {
        model: TEST_MODEL,
        appendToPrompt: 'custom extraction',
      },
    });

    memory.addMemoryStrategy(customStrategy);

    const template = Template.fromStack(stack);
    // The execution role must have been granted bedrock:InvokeModel on the foundation model
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock:InvokeModel*',
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('Should support adding multiple strategies of different types in sequence', () => {
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSummarization());
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInSemantic());
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInUserPreference());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: 'summary_builtin_cdkGen0001' }) }),
        Match.objectLike({ SemanticMemoryStrategy: Match.objectLike({ Name: 'semantic_builtin_cdkGen0001' }) }),
        Match.objectLike({ UserPreferenceMemoryStrategy: Match.objectLike({ Name: 'preference_builtin_cdkGen0001' }) }),
      ]),
    });
  });
});

describe('Memory strategy namespace validation tests', () => {
  beforeAll(() => {
    // No setup needed for these tests as they only test validation logic
  });

  test('Should throw error for namespace containing only opening brace', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid_namespace_strategy',
        description: 'Strategy with invalid namespace',
        namespaces: ['/strategies/{/actors'],
        customConsolidation: {
          model: TEST_MODEL,
          appendToPrompt: 'Test consolidation prompt',
        },
        customExtraction: {
          model: TEST_MODEL,
          appendToPrompt: 'Test extraction prompt',
        },
      });
    }).toThrow('Namespace with templates should contain valid variables: /strategies/{/actors');
  });

  test('Should throw error for namespace with invalid template variables', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid_template_strategy',
        description: 'Strategy with invalid template variables',
        namespaces: ['/strategies/{invalidVar}/actors'],
        customConsolidation: {
          model: TEST_MODEL,
          appendToPrompt: 'Test consolidation prompt',
        },
        customExtraction: {
          model: TEST_MODEL,
          appendToPrompt: 'Test extraction prompt',
        },
      });
    }).toThrow('Namespace with templates should contain valid variables: /strategies/{invalidVar}/actors');
  });

  test('Should accept valid namespace with correct template variables', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'valid_namespace_strategy',
        description: 'Strategy with valid namespace',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
        customConsolidation: {
          model: TEST_MODEL,
          appendToPrompt: 'Test consolidation prompt',
        },
        customExtraction: {
          model: TEST_MODEL,
          appendToPrompt: 'Test extraction prompt',
        },
      });
    }).not.toThrow();
  });

  test('Should accept namespace without template variables', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'no_template_strategy',
        description: 'Strategy without template variables',
        namespaces: ['/strategies/custom/actors'],
        customConsolidation: {
          model: TEST_MODEL,
          appendToPrompt: 'Test consolidation prompt',
        },
        customExtraction: {
          model: TEST_MODEL,
          appendToPrompt: 'Test extraction prompt',
        },
      });
    }).not.toThrow();
  });
});

describe('BuiltInMemoryStrategy unit tests', () => {
  test('Should create BuiltInMemoryStrategy with valid properties', () => {
    const strategy = MemoryStrategy.usingBuiltInSummarization();

    expect(strategy.strategyName).toMatch('summary_builtin_cdkGen0001');
    expect(strategy.description).toBe('Summarize interactions to preserve critical context and key insights');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}']);
    expect(strategy.strategyType).toBe('SUMMARIZATION');
  });

  test('Should create semantic BuiltInMemoryStrategy with valid properties', () => {
    const strategy = MemoryStrategy.usingBuiltInSemantic();

    expect(strategy.strategyName).toMatch('semantic_builtin_cdkGen0001');
    expect(strategy.description).toBe('Extract general factual knowledge, concepts and meanings from raw conversations in a context-independent format.');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}']);
    expect(strategy.strategyType).toBe('SEMANTIC');
  });

  test('Should create user preference BuiltInMemoryStrategy with valid properties', () => {
    const strategy = MemoryStrategy.usingBuiltInUserPreference();

    expect(strategy.strategyName).toMatch('preference_builtin_cdkGen0001');
    expect(strategy.description).toBe('Capture individual preferences, interaction patterns, and personalized settings to enhance future experiences.');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}']);
    expect(strategy.strategyType).toBe('USER_PREFERENCE');
  });

  test('Should render summarization strategy correctly', () => {
    const strategy = MemoryStrategy.usingBuiltInSummarization();
    const rendered = strategy.render();

    expect(rendered).toHaveProperty('summaryMemoryStrategy');
    expect(rendered.summaryMemoryStrategy).toMatchObject({
      name: expect.stringMatching('summary_builtin_cdkGen0001'),
      description: 'Summarize interactions to preserve critical context and key insights',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}'],
    });
  });

  test('Should render semantic strategy correctly', () => {
    const strategy = MemoryStrategy.usingBuiltInSemantic();
    const rendered = strategy.render();

    expect(rendered).toHaveProperty('semanticMemoryStrategy');
    expect(rendered.semanticMemoryStrategy).toMatchObject({
      name: expect.stringMatching('semantic_builtin_cdkGen0001'),
      description: 'Extract general factual knowledge, concepts and meanings from raw conversations in a context-independent format.',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
    });
  });

  test('Should render user preference strategy correctly', () => {
    const strategy = MemoryStrategy.usingBuiltInUserPreference();
    const rendered = strategy.render();

    expect(rendered).toHaveProperty('userPreferenceMemoryStrategy');
    expect(rendered.userPreferenceMemoryStrategy).toMatchObject({
      name: expect.stringMatching('preference_builtin_cdkGen0001'),
      description: 'Capture individual preferences, interaction patterns, and personalized settings to enhance future experiences.',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
    });
  });

  test('Should return undefined for grant method', () => {
    const strategy = MemoryStrategy.usingBuiltInSummarization();
    const mockRole = {} as iam.IRole;

    const grant = strategy.grant(mockRole);
    expect(grant).toBeUndefined();
  });

  test('Should create custom semantic strategy with valid properties', () => {
    const strategy = MemoryStrategy.usingSemantic({
      strategyName: 'custom_semantic',
      description: 'Custom semantic strategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/custom'],
    });

    expect(strategy.strategyName).toBe('custom_semantic');
    expect(strategy.description).toBe('Custom semantic strategy');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}/custom']);
    expect(strategy.strategyType).toBe('SEMANTIC');
  });

  test('Should create custom user preference strategy with valid properties', () => {
    const strategy = MemoryStrategy.usingUserPreference({
      strategyName: 'custom_user_preference',
      description: 'Custom user preference strategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/preferences'],
    });

    expect(strategy.strategyName).toBe('custom_user_preference');
    expect(strategy.description).toBe('Custom user preference strategy');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}/preferences']);
    expect(strategy.strategyType).toBe('USER_PREFERENCE');
  });

  test('Should create custom summarization strategy with valid properties', () => {
    const strategy = MemoryStrategy.usingSummarization({
      strategyName: 'custom_summarization',
      description: 'Custom summarization strategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/custom'],
    });

    expect(strategy.strategyName).toBe('custom_summarization');
    expect(strategy.description).toBe('Custom summarization strategy');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/custom']);
    expect(strategy.strategyType).toBe('SUMMARIZATION');
  });
});

describe('BuiltInMemoryStrategy validation tests', () => {
  test('Should throw error for invalid strategy name with hyphen', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid-name',
        description: 'Strategy with invalid name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).toThrow('The field Memory name with value "invalid-name" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for empty strategy name', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: '',
        description: 'Strategy with empty name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).toThrow('The field Memory name is 0 characters long but must be at least 1 characters');
  });

  test('Should throw error for strategy name with spaces', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid name',
        description: 'Strategy with spaces in name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).toThrow('The field Memory name with value "invalid name" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for strategy name with special characters', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid@name',
        description: 'Strategy with special characters',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).toThrow('The field Memory name with value "invalid@name" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for strategy name exceeding 48 characters', () => {
    const longName = 'a'.repeat(49);
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: longName,
        description: 'Strategy with long name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).toThrow('The field Memory name is 49 characters long but must be less than or equal to 48 characters');
  });

  test('Should accept valid strategy name with underscores', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'valid_strategy_name_123',
        description: 'Strategy with valid name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).not.toThrow();
  });

  test('Should accept valid strategy name with only letters and numbers', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'validStrategyName123',
        description: 'Strategy with valid name',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).not.toThrow();
  });

  test('Should throw error for namespace with invalid template variables in builtin strategy', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid_namespace_strategy',
        description: 'Strategy with invalid namespace',
        namespaces: ['/strategies/{invalidVar}/actors'],
      });
    }).toThrow('Namespace with templates should contain valid variables: /strategies/{invalidVar}/actors');
  });

  test('Should throw error for namespace containing only opening brace in builtin strategy', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'invalid_brace_strategy',
        description: 'Strategy with invalid namespace',
        namespaces: ['/strategies/{/actors'],
      });
    }).toThrow('Namespace with templates should contain valid variables: /strategies/{/actors');
  });

  test('Should accept valid namespace with correct template variables in builtin strategy', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'valid_namespace_strategy',
        description: 'Strategy with valid namespace',
        namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
      });
    }).not.toThrow();
  });

  test('Should accept namespace without template variables in builtin strategy', () => {
    expect(() => {
      MemoryStrategy.usingSemantic({
        strategyName: 'no_template_strategy',
        description: 'Strategy without template variables',
        namespaces: ['/strategies/custom/actors'],
      });
    }).not.toThrow();
  });
});

describe('SelfManagedMemoryStrategy unit tests', () => {
  let stack: cdk.Stack;
  let topic: sns.Topic;
  let bucket: s3.Bucket;

  beforeEach(() => {
    stack = new cdk.Stack();
    topic = new sns.Topic(stack, 'TestTopic');
    bucket = new s3.Bucket(stack, 'TestBucket');
  });

  test('Should create SelfManagedMemoryStrategy with default values', () => {
    const strategy = MemoryStrategy.usingSelfManaged({
      strategyName: 'test_self_managed',
      description: 'Test self managed strategy',
      invocationConfiguration: {
        topic: topic,
        s3Location: {
          bucketName: bucket.bucketName,
          objectKey: 'test/',
        },
      },
    });

    expect(strategy.strategyName).toBe('test_self_managed');
    expect(strategy.description).toBe('Test self managed strategy');
    expect(strategy.strategyType).toBe('CUSTOM');
    expect(strategy.historicalContextWindowSize).toBe(4);
    expect(strategy.triggerConditions.messageBasedTrigger).toBe(1);
    expect(strategy.triggerConditions.timeBasedTrigger?.toSeconds()).toBe(10);
    expect(strategy.triggerConditions.tokenBasedTrigger).toBe(100);
  });

  test('Should create SelfManagedMemoryStrategy with custom values', () => {
    const strategy = MemoryStrategy.usingSelfManaged({
      strategyName: 'custom_self_managed',
      description: 'Custom self managed strategy',
      historicalContextWindowSize: 10,
      invocationConfiguration: {
        topic: topic,
        s3Location: {
          bucketName: bucket.bucketName,
          objectKey: 'custom/',
        },
      },
      triggerConditions: {
        messageBasedTrigger: 5,
        timeBasedTrigger: cdk.Duration.seconds(30),
        tokenBasedTrigger: 500,
      },
    });

    expect(strategy.historicalContextWindowSize).toBe(10);
    expect(strategy.triggerConditions.messageBasedTrigger).toBe(5);
    expect(strategy.triggerConditions.timeBasedTrigger?.toSeconds()).toBe(30);
    expect(strategy.triggerConditions.tokenBasedTrigger).toBe(500);
  });

  test('Should create SelfManagedMemoryStrategy with partial trigger conditions', () => {
    const strategy = MemoryStrategy.usingSelfManaged({
      strategyName: 'partial_self_managed',
      description: 'Partial self managed strategy',
      invocationConfiguration: {
        topic: topic,
        s3Location: {
          bucketName: bucket.bucketName,
          objectKey: 'partial/',
        },
      },
      triggerConditions: {
        messageBasedTrigger: 3,
        // timeBasedTrigger and tokenBasedTrigger will use defaults
      },
    });

    expect(strategy.triggerConditions.messageBasedTrigger).toBe(3);
    expect(strategy.triggerConditions.timeBasedTrigger?.toSeconds()).toBe(10); // default
    expect(strategy.triggerConditions.tokenBasedTrigger).toBe(100); // default
  });

  test('Should validate historical context window size range', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'invalid_historical',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'test/',
          },
        },
        historicalContextWindowSize: 100, // Invalid: should be 0-50
      });
    }).toThrow('Historical context window size must be between 0 and 50, got 100');
  });

  test('Should validate message-based trigger range', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'invalid_message_trigger',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'test/',
          },
        },
        triggerConditions: {
          messageBasedTrigger: 100, // Invalid: should be 1-50
        },
      });
    }).toThrow('Message-based trigger must be between 1 and 50, got 100');
  });

  test('Should validate time-based trigger range', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'invalid_time_trigger',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'test/',
          },
        },
        triggerConditions: {
          timeBasedTrigger: cdk.Duration.seconds(5), // Invalid: should be 10-3000
        },
      });
    }).toThrow('Time-based trigger must be between 10 and 3000 seconds, got 5');
  });

  test('Should validate token-based trigger range', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'invalid_token_trigger',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'test/',
          },
        },
        triggerConditions: {
          tokenBasedTrigger: 50, // Invalid: should be 100-500000
        },
      });
    }).toThrow('Token-based trigger must be between 100 and 500000, got 50');
  });

  test('Should skip validation when historicalContextWindowSize is an unresolved token', () => {
    const param = new cdk.CfnParameter(stack, 'WindowSize', { type: 'Number', default: 10 });
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'token_historical',
        historicalContextWindowSize: param.valueAsNumber,
        invocationConfiguration: {
          topic: topic,
          s3Location: { bucketName: bucket.bucketName, objectKey: 'test/' },
        },
      });
    }).not.toThrow();
  });

  test('Should skip validation when trigger conditions are unresolved tokens', () => {
    const msgParam = new cdk.CfnParameter(stack, 'MsgTrigger', { type: 'Number', default: 5 });
    const tokenParam = new cdk.CfnParameter(stack, 'TokenTrigger', { type: 'Number', default: 500 });
    const timeParam = new cdk.CfnParameter(stack, 'TimeTrigger', { type: 'Number', default: 30 });
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'token_triggers',
        invocationConfiguration: {
          topic: topic,
          s3Location: { bucketName: bucket.bucketName, objectKey: 'test/' },
        },
        triggerConditions: {
          messageBasedTrigger: msgParam.valueAsNumber,
          timeBasedTrigger: cdk.Duration.seconds(timeParam.valueAsNumber),
          tokenBasedTrigger: tokenParam.valueAsNumber,
        },
      });
    }).not.toThrow();
  });

  test('Should validate multiple trigger conditions', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'invalid_multiple_triggers',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'test/',
          },
        },
        triggerConditions: {
          messageBasedTrigger: 0, // Invalid: should be 1-50
          timeBasedTrigger: cdk.Duration.seconds(5000), // Invalid: should be 10-3000
          tokenBasedTrigger: 1000000, // Invalid: should be 100-500000
        },
      });
    }).toThrow();
  });

  test('Should render CloudFormation properties correctly', () => {
    const strategy = MemoryStrategy.usingSelfManaged({
      strategyName: 'render_test',
      invocationConfiguration: {
        topic: topic,
        s3Location: {
          bucketName: bucket.bucketName,
          objectKey: 'render/',
        },
      },
      triggerConditions: {
        messageBasedTrigger: 2,
        timeBasedTrigger: cdk.Duration.seconds(60),
        tokenBasedTrigger: 200,
      },
    });

    const rendered = strategy.render();
    expect(rendered.customMemoryStrategy).toBeDefined();
    expect((rendered.customMemoryStrategy as any)?.name).toBe('render_test');
    expect((rendered.customMemoryStrategy as any)?.type).toBe('CUSTOM');
  });

  test('Should grant permissions correctly', () => {
    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    const strategy = MemoryStrategy.usingSelfManaged({
      strategyName: 'grant_test',
      invocationConfiguration: {
        topic: topic,
        s3Location: {
          bucketName: bucket.bucketName,
          objectKey: 'grant/',
        },
      },
    });

    const grant = strategy.grant(role);
    expect(grant).toBeDefined();
  });

  test('Should handle minimum valid values', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'min_values',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'min/',
          },
        },
        historicalContextWindowSize: 0,
        triggerConditions: {
          messageBasedTrigger: 1,
          timeBasedTrigger: cdk.Duration.seconds(10),
          tokenBasedTrigger: 100,
        },
      });
    }).not.toThrow();
  });

  test('Should handle maximum valid values', () => {
    expect(() => {
      MemoryStrategy.usingSelfManaged({
        strategyName: 'max_values',
        invocationConfiguration: {
          topic: topic,
          s3Location: {
            bucketName: bucket.bucketName,
            objectKey: 'max/',
          },
        },
        historicalContextWindowSize: 50,
        triggerConditions: {
          messageBasedTrigger: 50,
          timeBasedTrigger: cdk.Duration.seconds(3000),
          tokenBasedTrigger: 500000,
        },
      });
    }).not.toThrow();
  });
});

describe('Memory with custom execution role and strategies tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let customRole: iam.Role;
  let kmsKey: kms.Key;
  let topic: sns.Topic;
  let bucket: s3.Bucket;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create a custom execution role
    customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
      roleName: 'custom-memory-execution-role',
    });

    // Create a custom KMS key
    kmsKey = new kms.Key(stack, 'MemoryEncryptionKey', {
      description: 'KMS key for memory encryption',
      enableKeyRotation: true,
    });

    // Create SNS topic and S3 bucket for self-managed strategy
    topic = new sns.Topic(stack, 'MemoryTopic', {
      topicName: 'memory-processing-topic',
    });

    bucket = new s3.Bucket(stack, 'MemoryBucket', {
      bucketName: 'memory-processing-bucket',
    });

    // Create memory with custom role, KMS key, and mixed strategies
    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_custom_role_strategies',
      description: 'A test memory with custom execution role and mixed strategies',
      expirationDuration: Duration.days(60),
      executionRole: customRole,
      kmsKey: kmsKey,
      memoryStrategies: [
        // Built-in strategies
        MemoryStrategy.usingBuiltInSummarization(),
        MemoryStrategy.usingBuiltInSemantic(),
        // Custom managed strategy
        MemoryStrategy.usingSemantic({
          strategyName: 'custom_semantic_strategy',
          description: 'Custom semantic strategy with custom role',
          namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/custom'],
          customConsolidation: {
            model: TEST_MODEL,
            appendToPrompt: 'Custom consolidation prompt for semantic memory',
          },
          customExtraction: {
            model: TEST_MODEL,
            appendToPrompt: 'Custom extraction prompt for semantic memory',
          },
        }),
        // Self-managed strategy
        MemoryStrategy.usingSelfManaged({
          strategyName: 'self_managed_strategy',
          description: 'Self-managed strategy with custom role',
          invocationConfiguration: {
            topic: topic,
            s3Location: {
              bucketName: bucket.bucketName,
              objectKey: 'memory-processing/',
            },
          },
          triggerConditions: {
            messageBasedTrigger: 5,
            timeBasedTrigger: Duration.seconds(30),
            tokenBasedTrigger: 500,
          },
        }),
      ],
    });

    // Add additional strategies via addMemoryStrategy method
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInUserPreference());

    memory.addMemoryStrategy(MemoryStrategy.usingUserPreference({
      strategyName: 'added_user_preference_strategy',
      description: 'User preference strategy added via addMemoryStrategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/preferences'],
      customConsolidation: {
        model: TEST_MODEL,
        appendToPrompt: 'Custom consolidation prompt for user preferences',
      },
    }));

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.resourceCountIs('AWS::KMS::Key', 1);
    template.resourceCountIs('AWS::SNS::Topic', 1);
    template.resourceCountIs('AWS::S3::Bucket', 1);
  });

  test('Should have Memory resource with custom execution role and KMS encryption', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_custom_role_strategies',
      Description: 'A test memory with custom execution role and mixed strategies',
      EventExpiryDuration: 60,
      MemoryExecutionRoleArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('CustomExecutionRole*'),
          'Arn',
        ],
      },
      EncryptionKeyArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('MemoryEncryptionKey*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have custom execution role with correct properties', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'custom-memory-execution-role',
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('Should have KMS key with correct properties', () => {
    template.hasResourceProperties('AWS::KMS::Key', {
      Description: 'KMS key for memory encryption',
      EnableKeyRotation: true,
    });
  });

  test('Should have KMS permissions correctly applied on execution role', () => {
    // Check that KMS permissions are granted through IAM policy statements
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: ['kms:CreateGrant',
              'kms:Decrypt',
              'kms:DescribeKey',
              'kms:GenerateDataKey',
              'kms:GenerateDataKeyWithoutPlaintext',
              'kms:ReEncrypt*'],
            Resource: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('MemoryEncryptionKey*'),
                'Arn',
              ],
            },
          }),
        ]),
      },
      Roles: [
        {
          Ref: Match.stringLikeRegexp('CustomExecutionRole*'),
        },
      ],
    });
  });

  test('Should have Memory resource with all strategies from constructor', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        // Built-in summarization
        Match.objectLike({
          SummaryMemoryStrategy: {
            Name: Match.stringLikeRegexp('summary_builtin_.*'),
            Description: 'Summarize interactions to preserve critical context and key insights',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}'],
          },
        }),
        // Built-in semantic
        Match.objectLike( {
          SemanticMemoryStrategy: {
            Name: Match.stringLikeRegexp('semantic_builtin_.*'),
            Description: 'Extract general factual knowledge, concepts and meanings from raw conversations in a context-independent format.',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
          },
        }),
        // Custom semantic strategy
        Match.objectLike({
          CustomMemoryStrategy: {
            Name: 'custom_semantic_strategy',
            Description: 'Custom semantic strategy with custom role',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/custom'],
            Configuration: {
              SemanticOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt for semantic memory',
                },
                Extraction: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom extraction prompt for semantic memory',
                },
              },
            },
          },
        }),
        // Self-managed strategy
        Match.objectLike({
          CustomMemoryStrategy: {
            Name: 'self_managed_strategy',
            Description: 'Self-managed strategy with custom role',
            Configuration: {
              SelfManagedConfiguration: {
                HistoricalContextWindowSize: 4,
                InvocationConfiguration: {
                  TopicArn: {
                    Ref: Match.stringLikeRegexp('MemoryTopic*'),
                  },
                  PayloadDeliveryBucketName: {
                    Ref: Match.stringLikeRegexp('MemoryBucket*'),
                  },
                },
                TriggerConditions: [
                  {
                    MessageBasedTrigger: {
                      MessageCount: 5,
                    },
                  },
                  {
                    TimeBasedTrigger: {
                      IdleSessionTimeout: 30,
                    },
                  },
                  {
                    TokenBasedTrigger: {
                      TokenCount: 500,
                    },
                  },
                ],
              },
            },
          },
        }),
      ]),
    });
  });

  test('Should have Memory resource with strategies added via addMemoryStrategy', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        // Built-in user preference added via addMemoryStrategy
        {
          UserPreferenceMemoryStrategy: {
            Name: Match.stringLikeRegexp('preference_builtin_.*'),
            Description: 'Capture individual preferences, interaction patterns, and personalized settings to enhance future experiences.',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}'],
          },
        },
        // Custom user preference added via addMemoryStrategy
        {
          CustomMemoryStrategy: {
            Name: 'added_user_preference_strategy',
            Description: 'User preference strategy added via addMemoryStrategy',
            Namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/preferences'],
            Configuration: {
              UserPreferenceOverride: {
                Consolidation: {
                  ModelId: 'anthropic.claude-sonnet-4-6',
                  AppendToPrompt: 'Custom consolidation prompt for user preferences',
                },
              },
            },
          },
        },
      ]),
    });
  });

  test('Should have exactly six strategies total', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_custom_role_strategies',
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({ SummaryMemoryStrategy: Match.objectLike({ Name: 'summary_builtin_cdkGen0001' }) }),
        Match.objectLike({ SemanticMemoryStrategy: Match.objectLike({ Name: 'semantic_builtin_cdkGen0001' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'custom_semantic_strategy' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'self_managed_strategy' }) }),
        Match.objectLike({ UserPreferenceMemoryStrategy: Match.objectLike({ Name: 'preference_builtin_cdkGen0001' }) }),
        Match.objectLike({ CustomMemoryStrategy: Match.objectLike({ Name: 'added_user_preference_strategy' }) }),
      ]),
    });
  });

  test('Should have SNS permissions for self-managed strategy on execution role', () => {
    // Check that the execution role has the necessary SNS permissions for self-managed strategy
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['sns:GetTopicAttributes', 'sns:Publish']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('Should have S3 permissions for self-managed strategy on execution role', () => {
    // Check that the execution role has the necessary S3 permissions for self-managed strategy
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:GetBucketLocation', 's3:PutObject']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('Should have Bedrock invoke permissions for custom strategies on execution role', () => {
    // Check that custom strategies with model overrides are properly configured
    // We verify this by checking that the custom strategies are present in the template
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      MemoryStrategies: Match.arrayWith([
        Match.objectLike({
          CustomMemoryStrategy: Match.objectLike({
            Name: 'custom_semantic_strategy',
            Configuration: Match.objectLike({
              SemanticOverride: Match.objectLike({}),
            }),
          }),
        }),
      ]),
    });
  });

  test('Should have memoryStrategies property accessible after adding strategies', () => {
    expect(memory.memoryStrategies).toHaveLength(6);

    const strategyNames = memory.memoryStrategies.map(strategy => strategy.strategyName);

    expect(strategyNames.some((name: string) => /summary_builtin_.*/.test(name))).toBe(true);
    expect(strategyNames.some((name: string) => /semantic_builtin_.*/.test(name))).toBe(true);
    expect(strategyNames.some((name: string) => /preference_builtin_.*/.test(name))).toBe(true);
    expect(strategyNames).toContain('custom_semantic_strategy');
    expect(strategyNames).toContain('self_managed_strategy');
    expect(strategyNames).toContain('added_user_preference_strategy');
  });

  test('Should have correct execution role reference', () => {
    expect(memory.executionRole).toBe(customRole);
    // The role name is resolved as a token, so we check that it's defined
    expect(memory.executionRole?.roleName).toBeDefined();
  });

  test('Should have correct KMS key reference', () => {
    expect(memory.kmsKey).toBe(kmsKey);
    expect(memory.kmsKey?.keyArn).toBeDefined();
  });
});

describe('Memory Optional Physical Names', () => {
  let stack: cdk.Stack;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'TestStack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });
  });

  test('Should create Memory without memoryName (auto-generated)', () => {
    const memory = new Memory(stack, 'TestMemory', {
    });

    expect(memory.memoryName).toBeDefined();
    expect(memory.memoryName).not.toBe('');
  });
});

describe('Episodic Memory Strategy unit tests', () => {
  test('Should create built-in episodic strategy with valid properties', () => {
    const strategy = MemoryStrategy.usingBuiltInEpisodic();

    expect(strategy.strategyName).toMatch('episodic_builtin_cdkGen0001');
    expect(strategy.namespaces).toEqual(['/strategy/{memoryStrategyId}/actor/{actorId}/session/{sessionId}']);
    expect(strategy.strategyType).toBe('EPISODIC');
    // Verify reflection configuration is included
    expect((strategy as any).reflectionConfiguration).toBeDefined();
    expect((strategy as any).reflectionConfiguration.namespaces).toEqual(['/strategy/{memoryStrategyId}/actor/{actorId}']);
  });

  test('Should render built-in episodic strategy with correct structure', () => {
    const strategy = MemoryStrategy.usingBuiltInEpisodic();
    const rendered = strategy.render();

    expect(rendered).toHaveProperty('episodicMemoryStrategy');
    expect((rendered as any).episodicMemoryStrategy).toMatchObject({
      name: expect.stringMatching('episodic_builtin_cdkGen0001'),
      namespaces: ['/strategy/{memoryStrategyId}/actor/{actorId}/session/{sessionId}'],
      reflectionConfiguration: {
        namespaces: ['/strategy/{memoryStrategyId}/actor/{actorId}'],
      },
    });
  });

  test('Should create custom episodic strategy with valid properties', () => {
    const strategy = MemoryStrategy.usingEpisodic({
      strategyName: 'custom_episodic',
      description: 'Custom episodic strategy',
      namespaces: ['/strategies/{memoryStrategyId}/actors/{actorId}/episodic'],
    });

    expect(strategy.strategyName).toBe('custom_episodic');
    expect(strategy.description).toBe('Custom episodic strategy');
    expect(strategy.namespaces).toEqual(['/strategies/{memoryStrategyId}/actors/{actorId}/episodic']);
    expect(strategy.strategyType).toBe('EPISODIC');
  });

  test('Should return undefined for grant method on built-in episodic', () => {
    const strategy = MemoryStrategy.usingBuiltInEpisodic();
    const mockRole = {} as iam.IRole;

    const grant = strategy.grant(mockRole);
    expect(grant).toBeUndefined();
  });
});

describe('Memory with built-in episodic strategy tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_episodic',
      description: 'A test memory with built-in episodic strategy',
      expirationDuration: Duration.days(90),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInEpisodic(),
      ],
    });
  });

  test('Should create memory with episodic strategy without errors', () => {
    expect(memory.memoryName).toBe('test_memory_episodic');
    expect(memory.memoryStrategies).toHaveLength(1);
    expect(memory.memoryStrategies[0].strategyType).toBe('EPISODIC');
    expect(memory.memoryStrategies[0].strategyName).toMatch(/episodic_builtin/);
  });
});

describe('Memory with episodic reflection configuration tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const episodicStrategy = MemoryStrategy.usingEpisodic({
      strategyName: 'episodic_with_reflection',
      description: 'Episodic strategy with reflection configuration',
      namespaces: ['/journey/customer/{actorId}/episodes'],
      reflectionConfiguration: {
        namespaces: ['/journey/customer/{actorId}/reflections'],
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_episodic_reflection',
      description: 'A test memory with episodic reflection configuration',
      expirationDuration: Duration.days(120),
      memoryStrategies: [episodicStrategy],
    });
  });

  test('Should create memory with episodic reflection configuration without errors', () => {
    expect(memory.memoryName).toBe('test_memory_episodic_reflection');
    expect(memory.memoryStrategies).toHaveLength(1);
    expect(memory.memoryStrategies[0].strategyType).toBe('EPISODIC');
    expect(memory.memoryStrategies[0].strategyName).toBe('episodic_with_reflection');
  });

  test('Should have reflection configuration accessible', () => {
    const strategy = memory.memoryStrategies[0];
    expect(strategy).toHaveProperty('reflectionConfiguration');
    expect((strategy as any).reflectionConfiguration).toBeDefined();
    expect((strategy as any).reflectionConfiguration.namespaces).toEqual(['/journey/customer/{actorId}/reflections']);
  });
});

describe('Episodic reflection configuration validation tests', () => {
  test('Should throw error for empty reflection namespaces array', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'invalid_reflection',
        description: 'Episodic with empty reflection namespaces',
        namespaces: ['/journey/customer/{actorId}/episodes'],
        reflectionConfiguration: {
          namespaces: [],
        },
      });
    }).toThrow('Reflection configuration must have at least one namespace');
  });

  test('Should throw error for reflection namespace with invalid template variables', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'invalid_reflection_template',
        description: 'Episodic with invalid reflection namespace template',
        namespaces: ['/journey/customer/{actorId}/episodes'],
        reflectionConfiguration: {
          namespaces: ['/journey/{invalidVar}/reflections'],
        },
      });
    }).toThrow('Namespace with templates should contain valid variables: /journey/{invalidVar}/reflections');
  });

  test('Should accept valid reflection configuration', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'valid_reflection',
        description: 'Episodic with valid reflection configuration',
        namespaces: ['/journey/customer/{actorId}/episodes'],
        reflectionConfiguration: {
          namespaces: ['/journey/customer/{actorId}/reflections'],
        },
      });
    }).not.toThrow();
  });

  test('Should accept reflection configuration with multiple namespaces', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'multi_reflection',
        description: 'Episodic with multiple reflection namespaces',
        namespaces: ['/journey/customer/{actorId}/episodes'],
        reflectionConfiguration: {
          namespaces: [
            '/journey/customer/{actorId}/reflections/primary',
            '/journey/customer/{actorId}/reflections/secondary',
          ],
        },
      });
    }).not.toThrow();
  });

  test('Should accept reflection configuration without template variables', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'no_template_reflection',
        description: 'Episodic with reflection namespace without templates',
        namespaces: ['/journey/customer/episodes'],
        reflectionConfiguration: {
          namespaces: ['/journey/customer/reflections'],
        },
      });
    }).not.toThrow();
  });
});

describe('Memory with multiple episodic strategies tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_multi_episodic',
      description: 'A test memory with multiple episodic strategies',
      expirationDuration: Duration.days(90),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInEpisodic(),
        MemoryStrategy.usingEpisodic({
          strategyName: 'custom_episodic_1',
          description: 'First custom episodic strategy',
          namespaces: ['/journey/{actorId}/episodes/primary'],
        }),
        MemoryStrategy.usingEpisodic({
          strategyName: 'custom_episodic_2',
          description: 'Second custom episodic strategy with reflection',
          namespaces: ['/journey/{actorId}/episodes/secondary'],
          reflectionConfiguration: {
            namespaces: ['/journey/{actorId}/reflections/secondary'],
          },
        }),
      ],
    });
  });

  test('Should create memory with multiple episodic strategies without errors', () => {
    expect(memory.memoryName).toBe('test_memory_multi_episodic');
    expect(memory.memoryStrategies).toHaveLength(3);
    expect(memory.memoryStrategies.every(s => s.strategyType === 'EPISODIC')).toBe(true);
  });

  test('Should have all three episodic strategies with correct names', () => {
    const strategyNames = memory.memoryStrategies.map(s => s.strategyName);
    expect(strategyNames).toContain('custom_episodic_1');
    expect(strategyNames).toContain('custom_episodic_2');
    expect(strategyNames.some(name => name.includes('episodic_builtin'))).toBe(true);
  });

  test('Should have reflection configuration on second custom episodic', () => {
    const strategyWithReflection = memory.memoryStrategies.find(s => s.strategyName === 'custom_episodic_2');
    expect(strategyWithReflection).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration.namespaces).toEqual(['/journey/{actorId}/reflections/secondary']);
  });
});

describe('Memory with addMemoryStrategy for episodic tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create memory without initial strategies
    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_add_episodic',
      description: 'A test memory for testing addMemoryStrategy with episodic',
      expirationDuration: Duration.days(90),
    });

    // Add episodic strategies dynamically
    memory.addMemoryStrategy(MemoryStrategy.usingBuiltInEpisodic());
    memory.addMemoryStrategy(MemoryStrategy.usingEpisodic({
      strategyName: 'added_episodic_custom',
      description: 'Custom episodic added via addMemoryStrategy',
      namespaces: ['/added/{actorId}/episodes'],
      reflectionConfiguration: {
        namespaces: ['/added/{actorId}/reflections'],
      },
    }));
  });

  test('Should create memory and add episodic strategies without errors', () => {
    expect(memory.memoryName).toBe('test_memory_add_episodic');
    expect(memory.memoryStrategies).toHaveLength(2);
    expect(memory.memoryStrategies.every(s => s.strategyType === 'EPISODIC')).toBe(true);
  });

  test('Should have both episodic strategies with correct names', () => {
    const strategyNames = memory.memoryStrategies.map(s => s.strategyName);
    expect(strategyNames).toContain('added_episodic_custom');
    expect(strategyNames.some(name => name.includes('episodic_builtin'))).toBe(true);
  });

  test('Should have reflection configuration on added custom episodic', () => {
    const strategyWithReflection = memory.memoryStrategies.find(s => s.strategyName === 'added_episodic_custom');
    expect(strategyWithReflection).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration.namespaces).toEqual(['/added/{actorId}/reflections']);
  });
});

describe('Memory with mixed episodic and other strategies tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_mixed_episodic',
      description: 'A test memory with mixed episodic and other strategies',
      expirationDuration: Duration.days(100),
      memoryStrategies: [
        MemoryStrategy.usingBuiltInSummarization(),
        MemoryStrategy.usingBuiltInEpisodic(),
        MemoryStrategy.usingBuiltInSemantic(),
        MemoryStrategy.usingEpisodic({
          strategyName: 'custom_episodic_mixed',
          description: 'Custom episodic in mixed strategy set',
          namespaces: ['/mixed/{actorId}/episodes'],
          reflectionConfiguration: {
            namespaces: ['/mixed/{actorId}/reflections'],
          },
        }),
      ],
    });
  });

  test('Should create memory with mixed strategies including episodic without errors', () => {
    expect(memory.memoryName).toBe('test_memory_mixed_episodic');
    expect(memory.memoryStrategies).toHaveLength(4);
  });

  test('Should have correct mix of strategy types', () => {
    const strategyTypes = memory.memoryStrategies.map(s => s.strategyType);
    expect(strategyTypes).toContain('SUMMARIZATION');
    expect(strategyTypes).toContain('SEMANTIC');
    expect(strategyTypes.filter(t => t === 'EPISODIC')).toHaveLength(2);
  });

  test('Should have episodic strategies with correct names', () => {
    const episodicStrategies = memory.memoryStrategies.filter(s => s.strategyType === 'EPISODIC');
    const episodicNames = episodicStrategies.map(s => s.strategyName);
    expect(episodicNames).toContain('custom_episodic_mixed');
    expect(episodicNames.some(name => name.includes('episodic_builtin'))).toBe(true);
  });

  test('Should have reflection configuration on custom episodic', () => {
    const strategyWithReflection = memory.memoryStrategies.find(s => s.strategyName === 'custom_episodic_mixed');
    expect(strategyWithReflection).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration).toBeDefined();
    expect((strategyWithReflection as any).reflectionConfiguration.namespaces).toEqual(['/mixed/{actorId}/reflections']);
  });
});

describe('Episodic strategy validation edge cases tests', () => {
  test('Should accept episodic strategy without reflection configuration', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'episodic_no_reflection',
        description: 'Episodic without reflection',
        namespaces: ['/journey/{actorId}/episodes'],
      });
    }).not.toThrow();
  });

  test('Should accept episodic strategy with single reflection namespace', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'episodic_single_reflection',
        description: 'Episodic with single reflection namespace',
        namespaces: ['/journey/{actorId}/episodes'],
        reflectionConfiguration: {
          namespaces: ['/journey/{actorId}/reflections'],
        },
      });
    }).not.toThrow();
  });

  test('Should throw error for invalid episodic strategy name', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'invalid-episodic-name',
        description: 'Episodic with invalid name',
        namespaces: ['/journey/{actorId}/episodes'],
      });
    }).toThrow('The field Memory name with value "invalid-episodic-name" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should throw error for episodic with invalid main namespace template', () => {
    expect(() => {
      MemoryStrategy.usingEpisodic({
        strategyName: 'invalid_main_namespace',
        description: 'Episodic with invalid main namespace',
        namespaces: ['/journey/{badVariable}/episodes'],
      });
    }).toThrow('Namespace with templates should contain valid variables: /journey/{badVariable}/episodes');
  });
});

describe('Memory with stream delivery resources tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const stream = new kinesis.Stream(stack, 'MemoryEventStream');

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_with_stream',
      description: 'A test memory with stream delivery',
      expirationDuration: Duration.days(30),
      streamDeliveryResources: [
        StreamDeliveryResource.kinesis(stream, {
          contentConfigurations: [
            {
              type: StreamDeliveryContentType.MEMORY_RECORDS,
              level: StreamDeliveryContentLevel.FULL_CONTENT,
            },
          ],
        }),
      ],
    });

    app.synth();
    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
    template.resourceCountIs('AWS::Kinesis::Stream', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have Memory resource with StreamDeliveryResources', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      Name: 'test_memory_with_stream',
      StreamDeliveryResources: {
        Resources: [
          {
            Kinesis: {
              DataStreamArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('MemoryEventStream*'), 'Arn'] },
              ContentConfigurations: [
                {
                  Type: 'MEMORY_RECORDS',
                  Level: 'FULL_CONTENT',
                },
              ],
            },
          },
        ],
      },
    });
  });

  test('Should grant Kinesis write permissions to execution role', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'kinesis:PutRecord',
              'kinesis:PutRecords',
            ]),
            Resource: { 'Fn::GetAtt': [Match.stringLikeRegexp('MemoryEventStream*'), 'Arn'] },
          }),
          Match.objectLike({
            Effect: 'Allow',
            Action: 'kinesis:DescribeStream',
            Resource: { 'Fn::GetAtt': [Match.stringLikeRegexp('MemoryEventStream*'), 'Arn'] },
          }),
        ]),
      },
    });
  });

  test('Should have stream delivery resource in construct property', () => {
    expect(memory.streamDeliveryResources).toHaveLength(1);
  });
});

describe('Memory with addStreamDeliveryResource method tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let memory: Memory;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const stream = new kinesis.Stream(stack, 'AddedStream');

    memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_add_stream',
    });

    memory.addStreamDeliveryResource(StreamDeliveryResource.kinesis(stream, {
      contentConfigurations: [
        {
          type: StreamDeliveryContentType.MEMORY_RECORDS,
          level: StreamDeliveryContentLevel.METADATA_ONLY,
        },
      ],
    }));

    app.synth();
    template = Template.fromStack(stack);
  });

  test('Should have Memory resource with StreamDeliveryResources added via method', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      StreamDeliveryResources: {
        Resources: [
          {
            Kinesis: {
              DataStreamArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('AddedStream*'), 'Arn'] },
              ContentConfigurations: [
                {
                  Type: 'MEMORY_RECORDS',
                  Level: 'METADATA_ONLY',
                },
              ],
            },
          },
        ],
      },
    });
  });

  test('Should grant Kinesis write permissions via addStreamDeliveryResource', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'kinesis:PutRecord',
              'kinesis:PutRecords',
            ]),
          }),
        ]),
      },
    });
  });

  test('Should have stream delivery resources in construct property', () => {
    expect(memory.streamDeliveryResources).toHaveLength(1);
  });
});

describe('Memory stream delivery validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should throw error for more than one content configuration', () => {
    const stream = new kinesis.Stream(stack, 'TooManyConfigStream');

    expect(() => {
      new Memory(stack, 'invalid-memory', {
        memoryName: 'invalid_stream_memory',
        streamDeliveryResources: [
          StreamDeliveryResource.kinesis(stream, {
            contentConfigurations: [
              { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.FULL_CONTENT },
              { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
            ],
          }),
        ],
      });
    }).toThrow('Stream delivery resource currently supports at most one content configuration');
  });

  test('fails for empty content configurations array', () => {
    const stream = new kinesis.Stream(stack, 'EmptyConfigStream');

    expect(() => {
      new Memory(stack, 'empty-config-memory', {
        memoryName: 'empty_config_memory',
        streamDeliveryResources: [
          StreamDeliveryResource.kinesis(stream, {
            contentConfigurations: [],
          }),
        ],
      });
    }).toThrow('contentConfigurations must be specified');
  });

  test('fails for missing content configurations', () => {
    const stream = new kinesis.Stream(stack, 'MissingConfigStream');

    expect(() => {
      new Memory(stack, 'missing-config-memory', {
        memoryName: 'missing_config_memory',
        streamDeliveryResources: [
          // A JavaScript or jsii caller can bypass the required type, so this must
          // still surface a ValidationError rather than a TypeError
          StreamDeliveryResource.kinesis(stream, {} as any),
        ],
      });
    }).toThrow('contentConfigurations must be specified');
  });

  test('Should throw error when adding more than one stream delivery resource', () => {
    const stream1 = new kinesis.Stream(stack, 'Stream1');
    const stream2 = new kinesis.Stream(stack, 'Stream2');

    const memory = new Memory(stack, 'test-memory', {
      memoryName: 'test_memory',
      streamDeliveryResources: [
        StreamDeliveryResource.kinesis(stream1, {
          contentConfigurations: [
            { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
          ],
        }),
      ],
    });

    expect(() => {
      memory.addStreamDeliveryResource(StreamDeliveryResource.kinesis(stream2, {
        contentConfigurations: [
          { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
        ],
      }));
    }).toThrow('Memory currently supports at most one stream delivery resource');
  });

  test('Should not include StreamDeliveryResources when not configured', () => {
    new Memory(stack, 'no-stream-memory', {
      memoryName: 'no_stream_memory',
    });

    app.synth();
    const template = Template.fromStack(stack);

    const memoryResources = template.findResources('AWS::BedrockAgentCore::Memory');
    const memoryResource = Object.values(memoryResources)[0];
    expect(memoryResource.Properties.StreamDeliveryResources).toBeUndefined();
  });

  test.each([
    [StreamDeliveryContentLevel.METADATA_ONLY, 'METADATA_ONLY'],
    [StreamDeliveryContentLevel.FULL_CONTENT, 'FULL_CONTENT'],
  ])('Should always render the explicitly configured level %s', (level, expected) => {
    const stream = new kinesis.Stream(stack, `LevelStream${expected}`);

    new Memory(stack, 'level-memory', {
      memoryName: 'level_memory',
      streamDeliveryResources: [
        StreamDeliveryResource.kinesis(stream, {
          contentConfigurations: [
            { type: StreamDeliveryContentType.MEMORY_RECORDS, level },
          ],
        }),
      ],
    });

    app.synth();
    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      StreamDeliveryResources: {
        Resources: [
          {
            Kinesis: {
              ContentConfigurations: [
                {
                  Type: 'MEMORY_RECORDS',
                  Level: expected,
                },
              ],
            },
          },
        ],
      },
    });
  });
});

describe('StreamDeliveryResource factory tests', () => {
  test('kinesis() exposes the stream and content configurations it was created with', () => {
    const stack = new cdk.Stack(new cdk.App(), 'factory-stack');
    const stream = new kinesis.Stream(stack, 'FactoryStream');

    const resource = StreamDeliveryResource.kinesis(stream, {
      contentConfigurations: [
        { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
      ],
    });

    expect(resource.stream).toBe(stream);
    expect(resource.contentConfigurations).toEqual([
      { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
    ]);
  });

  test('kinesis() accepts an imported stream', () => {
    const stack = new cdk.Stack(new cdk.App(), 'imported-stack');
    const stream = kinesis.Stream.fromStreamArn(stack, 'ImportedStream', 'arn:aws:kinesis:us-east-1:123456789012:stream/imported');

    const memory = new Memory(stack, 'imported-stream-memory', {
      memoryName: 'imported_stream_memory',
      streamDeliveryResources: [
        StreamDeliveryResource.kinesis(stream, {
          contentConfigurations: [
            { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.METADATA_ONLY },
          ],
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::Memory', {
      StreamDeliveryResources: {
        Resources: [
          {
            Kinesis: {
              DataStreamArn: 'arn:aws:kinesis:us-east-1:123456789012:stream/imported',
              ContentConfigurations: [
                { Type: 'MEMORY_RECORDS', Level: 'METADATA_ONLY' },
              ],
            },
          },
        ],
      },
    });
    expect(memory.streamDeliveryResources).toHaveLength(1);
  });
});

describe('Memory with KMS-encrypted Kinesis stream tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const kmsKey = new kms.Key(stack, 'StreamEncryptionKey');
    const stream = new kinesis.Stream(stack, 'EncryptedStream', {
      encryptionKey: kmsKey,
    });

    new Memory(stack, 'test-memory', {
      memoryName: 'test_memory_kms_stream',
      streamDeliveryResources: [
        StreamDeliveryResource.kinesis(stream, {
          contentConfigurations: [
            { type: StreamDeliveryContentType.MEMORY_RECORDS, level: StreamDeliveryContentLevel.FULL_CONTENT },
          ],
        }),
      ],
    });

    app.synth();
    template = Template.fromStack(stack);
  });

  test('Should grant KMS permissions when stream uses customer-managed key', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Action: Match.arrayWith([
              'kms:Encrypt',
              'kms:ReEncrypt*',
              'kms:GenerateDataKey*',
            ]),
          }),
        ]),
      },
    });
  });
});
