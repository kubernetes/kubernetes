import * as fs from 'node:fs';
import * as path from 'node:path';
import { Match, Template } from '../../assertions';
import { AssertionError } from '../../assertions/lib/private/error';
import * as notifications from '../../aws-codestarnotifications';
import * as iam from '../../aws-iam';
import { ServicePrincipal } from '../../aws-iam';
import * as kms from '../../aws-kms';
import { CfnKey } from '../../aws-kms';
import * as cdk from '../../core';
import { Stage } from '../../core';
import * as sns from '../lib';
import { TopicGrants } from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('Topic', () => {
  describe('topic tests', () => {
    test('all defaults', () => {
      const stack = new cdk.Stack();
      new sns.Topic(stack, 'MyTopic');

      Template.fromStack(stack).resourceCountIs('AWS::SNS::Topic', 1);
    });

    test('topic without logging configs omits DeliveryStatusLogging', () => {
      const stack = new cdk.Stack();
      new sns.Topic(stack, 'MyTopic');

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        DeliveryStatusLogging: Match.absent(),
      });
    });

    test('specify topicName', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        topicName: 'topicName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'TopicName': 'topicName',
      });
    });

    test('specify displayName', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        displayName: 'displayName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'DisplayName': 'displayName',
      });
    });

    test('specify kmsMasterKey', () => {
      const stack = new cdk.Stack();
      const key = new kms.Key(stack, 'CustomKey');

      new sns.Topic(stack, 'MyTopic', {
        masterKey: key,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'KmsMasterKeyId': { 'Fn::GetAtt': ['CustomKey1E6D0D07', 'Arn'] },
      });
    });

    test('specify displayName and topicName', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        topicName: 'topicName',
        displayName: 'displayName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'DisplayName': 'displayName',
        'TopicName': 'topicName',
      });
    });

    test('Adds .fifo suffix when no topicName is passed', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'FifoTopic': true,
        'TopicName': 'MyTopic.fifo',
      });
    });

    test('specify fifo without .fifo suffix in topicName', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        topicName: 'topicName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'FifoTopic': true,
        'TopicName': 'topicName.fifo',
      });
    });

    test('specify fifo with .fifo suffix in topicName', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        topicName: 'topicName.fifo',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'FifoTopic': true,
        'TopicName': 'topicName.fifo',
      });
    });

    test('specify fifo without contentBasedDeduplication', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        topicName: 'topicName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'FifoTopic': true,
        'TopicName': 'topicName.fifo',
      });
    });

    test('specify fifo with contentBasedDeduplication', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        contentBasedDeduplication: true,
        fifo: true,
        topicName: 'topicName',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'ContentBasedDeduplication': true,
        'FifoTopic': true,
        'TopicName': 'topicName.fifo',
      });
    });

    test('throw with contentBasedDeduplication on non-fifo topic', () => {
      const stack = new cdk.Stack();

      expect(() => new sns.Topic(stack, 'MyTopic', {
        contentBasedDeduplication: true,
      })).toThrow(/Content based deduplication can only be enabled for FIFO SNS topics./);
    });

    test('specify signatureVersion', () => {
      const stack = new cdk.Stack();

      new sns.Topic(stack, 'MyTopic', {
        signatureVersion: '2',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'SignatureVersion': '2',
      });
    });

    test('throw with incorrect signatureVersion', () => {
      const stack = new cdk.Stack();

      expect(() => new sns.Topic(stack, 'MyTopic', {
        signatureVersion: '3',
      })).toThrow(/signatureVersion must be "1" or "2", received: "3"/);
    });

    test('throw error when displayName is too long', () => {
      const stack = new cdk.Stack();

      expect(() => {
        new sns.Topic(stack, 'MyTopic', {
          displayName: 'a'.repeat(101),
        });
      }).toThrow('displayName must be less than or equal to 100 characters, got 101');
    });
  });

  test('can add a policy to the topic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic');

    // WHEN
    topic.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['sns:*'],
      principals: [new iam.ArnPrincipal('arn')],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [{
          'Sid': '0',
          'Action': 'sns:*',
          'Effect': 'Allow',
          'Principal': { 'AWS': 'arn' },
          'Resource': '*',
        }],
      },
    });
  });

  test('can enforce ssl when creating the topic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    new sns.Topic(stack, 'Topic', {
      enforceSSL: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            'Sid': 'AllowPublishThroughSSLOnly',
            'Action': 'sns:Publish',
            'Effect': 'Deny',
            'Resource': {
              'Ref': 'TopicBFC7AF6E',
            },
            'Condition': {
              'Bool': {
                'aws:SecureTransport': 'false',
              },
            },
            'Principal': '*',
          },
        ],
      },
    });
  });

  test('can enforce ssl with addToResourcePolicy method', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic', {
      enforceSSL: true,
    });

    // WHEN
    topic.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['sns:*'],
      principals: [new iam.ArnPrincipal('arn')],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            'Sid': 'AllowPublishThroughSSLOnly',
            'Action': 'sns:Publish',
            'Effect': 'Deny',
            'Resource': {
              'Ref': 'TopicBFC7AF6E',
            },
            'Condition': {
              'Bool': {
                'aws:SecureTransport': 'false',
              },
            },
            'Principal': '*',
          },
          {
            'Sid': '1',
            'Action': 'sns:*',
            'Effect': 'Allow',
            'Principal': { 'AWS': 'arn' },
            'Resource': '*',
          },
        ],
      },
    });
  });

  test('give publishing permissions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic');
    const user = new iam.User(stack, 'User');

    // WHEN
    topic.grants.publish(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        Version: '2012-10-17',
        'Statement': [
          {
            'Action': 'sns:Publish',
            'Effect': 'Allow',
            'Resource': stack.resolve(topic.topicArn),
          },
        ],
      },
    });
  });

  test('give publishing permissions to CfnTopic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.CfnTopic(stack, 'Topic');
    const user = new iam.User(stack, 'User');

    // WHEN
    TopicGrants.fromTopic(topic).publish(user);

    // THEN
    let template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        Version: '2012-10-17',
        'Statement': [
          {
            'Action': 'sns:Publish',
            'Effect': 'Allow',
            'Resource': { Ref: 'Topic' },
          },
        ],
      },
    });
  });

  test('give arbitrary set of permissions to CfnTopic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key = new kms.CfnKey(stack, 'Key');
    const topic = new sns.CfnTopic(stack, 'Topic', {
      kmsMasterKeyId: key.attrKeyId,
    });
    const user = new iam.User(stack, 'User');

    // WHEN
    TopicGrants.fromTopic(topic).actions(user, ['sns:Foo'], {
      keyActions: ['kms:Bar', 'kms:Zee'],
    });

    // THEN
    let template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 'sns:Foo',
            Effect: 'Allow',
            Resource: { Ref: 'Topic' },
          },
          {
            Action: ['kms:Bar', 'kms:Zee'],
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': ['Key', 'Arn'],
            },
          },
        ],
      },
    });

    template.hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: [{
          Action: ['kms:Bar', 'kms:Zee'],
          Effect: 'Allow',
          Principal: { AWS: { 'Fn::GetAtt': ['User00B015A1', 'Arn'] } },
          Resource: '*',
        }],
        Version: '2012-10-17',
      },
    });
  });

  test('give service principal permissions to publish to CfnTopic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.CfnTopic(stack, 'Topic');
    const principal = new ServicePrincipal('some.service.amazonaws.com');

    // WHEN
    TopicGrants.fromTopic(topic).publish(principal);

    // THEN
    let template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::SNS::TopicPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 'sns:Publish',
            Effect: 'Allow',
            Resource: { Ref: 'Topic' },
            Principal: { Service: 'some.service.amazonaws.com' },
          },
        ],
      },
    });
  });

  test('give service principal permissions to publish to CfnTopic with encryption key', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key = new CfnKey(stack, 'CustomKey', {
      keyPolicy: { Statement: [] },
    });
    const topic = new sns.CfnTopic(stack, 'Topic', {
      kmsMasterKeyId: key.attrKeyId,
    });
    const principal = new ServicePrincipal('some.service.amazonaws.com');

    // WHEN
    TopicGrants.fromTopic(topic).publish(principal);

    // THEN
    let template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: ['kms:Decrypt', 'kms:GenerateDataKey*'],
            Effect: 'Allow',
            Resource: '*',
            Principal: { Service: 'some.service.amazonaws.com' },
          },
        ],
      },
    });
  });

  test('refer to masterKey', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'CustomKey');
    const topic = new sns.Topic(stack, 'Topic', { masterKey: key });

    // THEN
    expect(topic.masterKey!).toEqual(key);
  });

  test('give publishing permissions with masterKey', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'CustomKey');
    const topic = new sns.Topic(stack, 'Topic', { masterKey: key });
    const user = new iam.User(stack, 'User');

    // WHEN
    topic.grantPublish(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        Version: '2012-10-17',
        'Statement': [
          {
            'Action': 'sns:Publish',
            'Effect': 'Allow',
            'Resource': stack.resolve(topic.topicArn),
          },
          {
            'Action': ['kms:Decrypt', 'kms:GenerateDataKey*'],
            'Effect': 'Allow',
            'Resource': {
              'Fn::GetAtt': ['CustomKey1E6D0D07', 'Arn'],
            },
          },
        ],
      },
    });
  });

  test('give subscribing permissions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic');
    const user = new iam.User(stack, 'User');

    // WHEN
    topic.grantSubscribe(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        Version: '2012-10-17',
        'Statement': [
          {
            'Action': 'sns:Subscribe',
            'Effect': 'Allow',
            'Resource': stack.resolve(topic.topicArn),
          },
        ],
      },
    });
  });

  test('TopicPolicy passed document', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'MyTopic');
    const ps = new iam.PolicyStatement({
      actions: ['service:statement0'],
      principals: [new iam.ArnPrincipal('arn')],
    });

    // WHEN
    new sns.TopicPolicy(stack, 'topicpolicy', { topics: [topic], policyDocument: new iam.PolicyDocument({ assignSids: true, statements: [ps] }) });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'service:statement0',
            'Effect': 'Allow',
            'Principal': { 'AWS': 'arn' },
            'Sid': '0',
          },
        ],
        'Version': '2012-10-17',
      },
      'Topics': [
        {
          'Ref': 'MyTopic86869434',
        },
      ],
    });
  });

  test('Add statements to policy', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    const topicPolicy = new sns.TopicPolicy(stack, 'TopicPolicy', {
      topics: [topic],
    });
    topicPolicy.document.addStatements(new iam.PolicyStatement({
      actions: ['service:statement0'],
      principals: [new iam.ArnPrincipal('arn')],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'service:statement0',
            'Effect': 'Allow',
            'Principal': { 'AWS': 'arn' },
            'Sid': '0',
          },
        ],
        'Version': '2012-10-17',
      },
      'Topics': [
        {
          'Ref': 'MyTopic86869434',
        },
      ],
    });
  });

  test('Create topic policy and enforce ssl', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    new sns.TopicPolicy(stack, 'TopicPolicy', {
      topics: [topic],
      enforceSSL: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Sid': 'AllowPublishThroughSSLOnly',
            'Action': 'sns:Publish',
            'Effect': 'Deny',
            'Resource': {
              'Ref': 'MyTopic86869434',
            },
            'Condition': {
              'Bool': {
                'aws:SecureTransport': 'false',
              },
            },
            'Principal': '*',
          },
        ],
        'Version': '2012-10-17',
      },
      'Topics': [
        {
          'Ref': 'MyTopic86869434',
        },
      ],
    });
  });

  test('topic resource policy includes unique SIDs', () => {
    const stack = new cdk.Stack();

    const topic = new sns.Topic(stack, 'MyTopic');

    topic.addToResourcePolicy(new iam.PolicyStatement({
      actions: ['service:statement0'],
      principals: [new iam.ArnPrincipal('arn')],
      resources: ['*'],
    }));
    topic.addToResourcePolicy(new iam.PolicyStatement({
      actions: ['service:statement1'],
      principals: [new iam.ArnPrincipal('arn')],
      resources: ['*'],
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'service:statement0',
            'Effect': 'Allow',
            'Principal': { 'AWS': 'arn' },
            'Sid': '0',
          },
          {
            'Action': 'service:statement1',
            'Effect': 'Allow',
            'Principal': { 'AWS': 'arn' },
            'Sid': '1',
          },
        ],
        'Version': '2012-10-17',
      },
      'Topics': [
        {
          'Ref': 'MyTopic86869434',
        },
      ],
    });
  });

  test('fromTopicArn', () => {
    // GIVEN
    const stack2 = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicArn(stack2, 'Imported', 'arn:aws:sns:*:123456789012:my_corporate_topic');

    // THEN
    expect(imported.topicName).toEqual('my_corporate_topic');
    expect(imported.topicArn).toEqual('arn:aws:sns:*:123456789012:my_corporate_topic');
    expect(imported.fifo).toEqual(false);
  });

  test('fromTopicArn fifo', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicArn(stack, 'Imported', 'arn:aws:sns:*:123456789012:mytopic.fifo');

    // THEN
    expect(imported.topicName).toEqual('mytopic.fifo');
    expect(imported.topicArn).toEqual('arn:aws:sns:*:123456789012:mytopic.fifo');
    expect(imported.fifo).toEqual(true);
  });

  test('fromTopicAttributes contentBasedDeduplication false', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicAttributes(stack, 'Imported', {
      topicArn: 'arn:aws:sns:*:123456789012:mytopic',
    });

    // THEN
    expect(imported.topicName).toEqual('mytopic');
    expect(imported.topicArn).toEqual('arn:aws:sns:*:123456789012:mytopic');
    expect(imported.contentBasedDeduplication).toEqual(false);
  });

  test('fromTopicAttributes contentBasedDeduplication true', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicAttributes(stack, 'Imported', {
      topicArn: 'arn:aws:sns:*:123456789012:mytopic.fifo',
      contentBasedDeduplication: true,
    });

    // THEN
    expect(imported.topicName).toEqual('mytopic.fifo');
    expect(imported.topicArn).toEqual('arn:aws:sns:*:123456789012:mytopic.fifo');
    expect(imported.contentBasedDeduplication).toEqual(true);
  });

  test('fromTopicAttributes throws with contentBasedDeduplication on non-fifo topic', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    expect(() => sns.Topic.fromTopicAttributes(stack, 'Imported', {
      topicArn: 'arn:aws:sns:*:123456789012:mytopic',
      contentBasedDeduplication: true,
    })).toThrow(/Cannot import topic; contentBasedDeduplication is only available for FIFO SNS topics./);
  });

  test('fromTopicAttributes keyArn', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const keyArn = 'arn:aws:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab';

    // WHEN
    const imported = sns.Topic.fromTopicAttributes(stack, 'Imported', {
      topicArn: 'arn:aws:sns:*:123456789012:mytopic',
      keyArn,
    });

    // THEN
    expect(imported.masterKey?.keyArn).toEqual(keyArn);
  });

  test('sets account for imported topic env', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicArn(stack, 'Imported', 'arn:aws:sns:us-west-2:123456789012:my-topic');

    // THEN
    expect(imported.env.account).toEqual('123456789012');
  });

  test('sets region for imported topic env', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = sns.Topic.fromTopicArn(stack, 'Imported', 'arn:aws:sns:us-west-2:123456789012:my-topic');

    // THEN
    expect(imported.env.region).toEqual('us-west-2');
  });

  test('test metrics', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic');

    // THEN
    expect(stack.resolve(topic.metricNumberOfMessagesPublished())).toEqual({
      dimensions: { TopicName: { 'Fn::GetAtt': ['TopicBFC7AF6E', 'TopicName'] } },
      namespace: 'AWS/SNS',
      metricName: 'NumberOfMessagesPublished',
      period: cdk.Duration.minutes(5),
      statistic: 'Sum',
    });

    expect(stack.resolve(topic.metricPublishSize())).toEqual({
      dimensions: { TopicName: { 'Fn::GetAtt': ['TopicBFC7AF6E', 'TopicName'] } },
      namespace: 'AWS/SNS',
      metricName: 'PublishSize',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });
  });

  test('subscription is created under the topic scope by default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const topic = new sns.Topic(stack, 'Topic');

    // WHEN
    topic.addSubscription({
      bind: () => ({
        protocol: sns.SubscriptionProtocol.HTTP,
        endpoint: 'http://foo/bar',
        subscriberId: 'my-subscription',
      }),
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::SNS::Subscription', 1);
  });

  test('if "scope" is defined, subscription will be created under that scope', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'A');
    const stack2 = new cdk.Stack(app, 'B');
    const topic = new sns.Topic(stack, 'Topic');

    // WHEN
    topic.addSubscription({
      bind: () => ({
        protocol: sns.SubscriptionProtocol.HTTP,
        endpoint: 'http://foo/bar',
        subscriberScope: stack2,
        subscriberId: 'subscriberId',
      }),
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::SNS::Subscription', 0);
    Template.fromStack(stack2).resourceCountIs('AWS::SNS::Subscription', 1);
  });

  test('fails if topic policy has no actions', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'my-stack');
    const topic = new sns.Topic(stack, 'Topic');

    // WHEN
    topic.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      principals: [new iam.ArnPrincipal('arn')],
    }));

    // THEN
    expect(() => app.synth()).toThrow(/A PolicyStatement must specify at least one \'action\' or \'notAction\'/);
  });

  test('fails if topic policy has no IAM principals', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'my-stack');
    const topic = new sns.Topic(stack, 'Topic');

    // WHEN
    topic.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['sns:*'],
    }));

    // THEN
    expect(() => app.synth()).toThrow(/A PolicyStatement used in a resource-based policy must specify at least one IAM principal/);
  });

  test('topic policy should be set if topic as a notifications rule target', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'my-stack');
    const topic = new sns.Topic(stack, 'Topic');
    const rule = new notifications.NotificationRule(stack, 'MyNotificationRule', {
      source: {
        bindAsNotificationRuleSource: () => ({
          sourceArn: 'ARN',
        }),
      },
      events: ['codebuild-project-build-state-succeeded'],
    });

    rule.addTarget(topic);

    Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [{
          'Sid': '0',
          'Action': 'sns:Publish',
          'Effect': 'Allow',
          'Principal': { 'Service': 'codestar-notifications.amazonaws.com' },
          'Resource': { 'Ref': 'TopicBFC7AF6E' },
        }],
      },
      Topics: [{
        Ref: 'TopicBFC7AF6E',
      }],
    });
  });

  test('specify delivery status logging configuration through construct props', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const feedbackRole = new iam.Role(stack, 'feedbackRole', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });

    // WHEN
    new sns.Topic(stack, 'MyTopic', {
      loggingConfigs: [
        {
          protocol: sns.LoggingProtocol.SQS,
          failureFeedbackRole: feedbackRole,
          successFeedbackRole: feedbackRole,
          successFeedbackSampleRate: 50,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
      'DeliveryStatusLogging': [{
        'Protocol': 'sqs',
        'SuccessFeedbackRoleArn': { 'Fn::GetAtt': ['feedbackRole2010903F', 'Arn'] },
        'FailureFeedbackRoleArn': { 'Fn::GetAtt': ['feedbackRole2010903F', 'Arn'] },
        'SuccessFeedbackSampleRate': '50',
      }],
    });
  });

  test('add delivery status logging configuration to a topic', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const feedbackRole = new iam.Role(stack, 'feedbackRole', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    topic.addLoggingConfig({
      protocol: sns.LoggingProtocol.HTTP,
      failureFeedbackRole: feedbackRole,
      successFeedbackRole: feedbackRole,
      successFeedbackSampleRate: 50,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
      'DeliveryStatusLogging': [{
        'Protocol': 'http/s',
        'SuccessFeedbackRoleArn': { 'Fn::GetAtt': ['feedbackRole2010903F', 'Arn'] },
        'FailureFeedbackRoleArn': { 'Fn::GetAtt': ['feedbackRole2010903F', 'Arn'] },
        'SuccessFeedbackSampleRate': '50',
      }],
    });
  });

  test('fails if success feedback sample rate is outside the appropriate range', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'my-stack');
    const feedbackRole = new iam.Role(stack, 'feedbackRole', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });

    // WHEN
    new sns.Topic(stack, 'MyTopic', {
      loggingConfigs: [
        {
          protocol: sns.LoggingProtocol.SQS,
          failureFeedbackRole: feedbackRole,
          successFeedbackRole: feedbackRole,
          successFeedbackSampleRate: 110,
        },
      ],
    });

    // THEN
    expect(() => app.synth()).toThrow(/Success feedback sample rate must be an integer between 0 and 100/);
  });

  test('fails if success feedback sample rate is decimal', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'my-stack');
    const feedbackRole = new iam.Role(stack, 'feedbackRole', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });

    // WHEN
    new sns.Topic(stack, 'MyTopic', {
      loggingConfigs: [
        {
          protocol: sns.LoggingProtocol.SQS,
          failureFeedbackRole: feedbackRole,
          successFeedbackRole: feedbackRole,
          successFeedbackSampleRate: 50.4,
        },
      ],
    });

    // THEN
    expect(() => app.synth()).toThrow(/Success feedback sample rate must be an integer between 0 and 100/);
  });

  describe('message retention period', () => {
    test('specify message retention period in days', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      // WHEN
      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        messageRetentionPeriodInDays: 10,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        ArchivePolicy: {
          MessageRetentionPeriod: 10,
        },
        FifoTopic: true,
      });
    });

    test.each([0, 366, 12.3, NaN])('throw error if message retention period is invalid value "%s"', (days) => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      // THEN
      expect(() => new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        messageRetentionPeriodInDays: days,
      })).toThrow(/`messageRetentionPeriodInDays` must be an integer between 1 and 365/);
    });

    test('throw error when specify messageRetentionPeriodInDays to standard topic', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      expect(
        () => new sns.Topic(stack, 'MyTopic', {
          messageRetentionPeriodInDays: 12,
        }),
      ).toThrow('`messageRetentionPeriodInDays` is only valid for FIFO SNS topics');
    });
  });

  describe('tracingConfig', () => {
    test('specify tracingConfig', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      // WHEN
      new sns.Topic(stack, 'MyTopic', {
        tracingConfig: sns.TracingConfig.ACTIVE,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        'TracingConfig': 'Active',
      });
    });
  });

  describe('fifoThroughputScope', () => {
    test.each([sns.FifoThroughputScope.MESSAGE_GROUP, sns.FifoThroughputScope.TOPIC])('set fifoThroughputScope to %s', (fifoThroughputScope) => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      // WHEN
      new sns.Topic(stack, 'MyTopic', {
        fifo: true,
        fifoThroughputScope,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        FifoTopic: true,
        FifoThroughputScope: fifoThroughputScope,
      });
    });

    test('throw error when specify fifoThroughputScope to standard topic', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app);

      expect(
        () => new sns.Topic(stack, 'MyTopic', {
          fifoThroughputScope: sns.FifoThroughputScope.MESSAGE_GROUP,
        }),
      ).toThrow('`fifoThroughputScope` can only be set for FIFO SNS topics.');
    });
  });
});

/*
This is a representative test suite for source tracing.
What we are asserting here about CfnTopic applies to all L1 constructs.
 */
describe('Source tracing', () => {
  test('Metadata contains propertyAssignment and stack trace with CDK_DEBUG=1', () => {
    try {
      process.env.CDK_DEBUG = '1';
      const stack = new cdk.Stack();

      const topic = new sns.CfnTopic(stack, 'MyTopic', {
        topicName: 'topicName',
      });

      topic.displayName = 'something';
      const lineWherePropertyWasSet = getLineNumber() - 1; // the one before this one

      const asm = synth(stack);
      const metadata = JSON.parse(fs.readFileSync(path.join(asm.directory, 'Default.metadata.json'), 'utf8'));
      const propertyAssignmentEntry = metadata['/Default/MyTopic'].find((e: any) => e.type === 'aws:cdk:propertyAssignment');

      expect(propertyAssignmentEntry).toBeDefined();
      expect(propertyAssignmentEntry.data.propertyName).toEqual('DisplayName');
      expect(propertyAssignmentEntry.data.stackTrace.some(
        (t: string) => t.includes(`${__filename}:${lineWherePropertyWasSet}`)),
      ).toBe(true);
    } finally {
      delete process.env.CDK_DEBUG;
    }
  });

  test('Metadata does not contain propertyAssignment by default', () => {
    const stack = new cdk.Stack();

    const topic = new sns.CfnTopic(stack, 'MyTopic', {
      topicName: 'topicName',
    });

    topic.displayName = 'something';

    const asm = synth(stack);
    const metadata = JSON.parse(fs.readFileSync(path.join(asm.directory, 'Default.metadata.json'), 'utf8'));
    const propertyAssignmentEntry = metadata['/Default/MyTopic'].find((e: any) => e.type === 'aws:cdk:propertyAssignment');

    expect(propertyAssignmentEntry).toBeUndefined();
  });
});

function synth(stack: cdk.Stack) {
  const stage = Stage.of(stack);
  if (!Stage.isStage(stage)) {
    throw new AssertionError('unexpected: all stacks must be part of a Stage or an App');
  }

  return stage.synth();
}

function getLineNumber(): number {
  const err = new Error();
  const line = err.stack?.split('\n')[2]?.match(/:(\d+):\d+\)?$/)?.[1];
  return Number(line);
}
