import type * as notifications from '../lib';

export class FakeCodeBuild implements notifications.INotificationRuleSource {
  readonly projectArn = 'arn:aws:codebuild::123456789012:project/MyCodebuildProject';
  readonly projectName = 'test-project';

  bindAsNotificationRuleSource(): notifications.NotificationRuleSourceConfig {
    return {
      sourceArn: this.projectArn,
    };
  }
}

export class FakeCodePipeline implements notifications.INotificationRuleSource {
  readonly pipelineArn = 'arn:aws:codepipeline::123456789012:MyCodepipelineProject';
  readonly pipelineName = 'test-pipeline';

  bindAsNotificationRuleSource(): notifications.NotificationRuleSourceConfig {
    return {
      sourceArn: this.pipelineArn,
    };
  }
}

export class FakeCodeCommit implements notifications.INotificationRuleSource {
  readonly repositoryArn = 'arn:aws:codecommit::123456789012:MyCodecommitProject';
  readonly repositoryName = 'test-repository';

  bindAsNotificationRuleSource(): notifications.NotificationRuleSourceConfig {
    return {
      sourceArn: this.repositoryArn,
    };
  }
}

export class FakeSnsTopicTarget implements notifications.INotificationRuleTarget {
  readonly topicArn = 'arn:aws:sns::123456789012:MyTopic';

  bindAsNotificationRuleTarget(): notifications.NotificationRuleTargetConfig {
    return {
      targetType: 'SNS',
      targetAddress: this.topicArn,
    };
  }
}

export class FakeSlackTarget implements notifications.INotificationRuleTarget {
  readonly slackChannelConfigurationArn = 'arn:aws:chatbot::123456789012:chat-configuration/slack-channel/MySlackChannel';

  bindAsNotificationRuleTarget(): notifications.NotificationRuleTargetConfig {
    return {
      targetType: 'AWSChatbotSlack',
      targetAddress: this.slackChannelConfigurationArn,
    };
  }
}
