import type { IConstruct } from 'constructs';
import { FakeBuildAction } from './fake-build-action';
import { FakeSourceAction } from './fake-source-action';
import { Match, Template } from '../../assertions';
import { CodeStarConnectionsSourceAction } from '../../aws-codepipeline-actions';
import * as cdk from '../../core';
import * as codepipeline from '../lib';

const connectionArn = 'arn:aws:codestar-connections:us-east-1:111111111111:connection/ConnectionId';

describe('triggers', () => {
  let stack: cdk.Stack;
  let sourceArtifact: codepipeline.Artifact;
  let sourceAction: codepipeline.Action;
  let buildAction: codepipeline.Action;

  beforeEach(() => {
    stack = new cdk.Stack();
    sourceArtifact = new codepipeline.Artifact();
    sourceAction = new CodeStarConnectionsSourceAction({
      actionName: 'CodeStarConnectionsSourceAction',
      output: sourceArtifact,
      connectionArn,
      owner: 'owner',
      repo: 'repo',
    });
    buildAction = new FakeBuildAction({
      actionName: 'FakeBuild',
      input: sourceArtifact,
    });
  });

  test('can specify triggers with tags in pushFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: ['exclude1', 'exclude2'],
            tagsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Tags: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify triggers with branches in pushFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify triggers with branches and file paths in pushFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            filePathsExcludes: ['exclude1', 'exclude2'],
            filePathsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            FilePaths: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify triggers with branches in pullRequestFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify triggers with branches and file paths in pullRequestFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            filePathsExcludes: ['exclude1', 'exclude2'],
            filePathsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            FilePaths: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify triggers with branches and events in pullRequestFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            events: [
              codepipeline.GitPullRequestEvent.OPEN,
              codepipeline.GitPullRequestEvent.UPDATED,
              codepipeline.GitPullRequestEvent.CLOSED,
            ],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            Events: ['OPEN', 'UPDATED', 'CLOSED'],
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('can specify multiple triggers', () => {
    const sourceArtifact2 = new codepipeline.Artifact();
    const sourceAction2 = new CodeStarConnectionsSourceAction({
      actionName: 'CodeStarConnectionsSourceAction2',
      output: sourceArtifact2,
      connectionArn: 'arn:aws:codestar-connections:us-east-1:111111111111:connection/ConnectionId2',
      owner: 'owner',
      repo: 'repo',
    });

    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [
        {
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              tagsExcludes: ['exclude1', 'exclude2'],
              tagsIncludes: ['include1', 'include2'],
            }],
          },
        },
        {
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction: sourceAction2,
            pushFilter: [{
              tagsExcludes: ['exclude1', 'exclude2'],
              tagsIncludes: ['include1', 'include2'],
            }],
          },
        },
      ],
    });

    testPipelineSetup(pipeline, [sourceAction, sourceAction2], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [
        {
          GitConfiguration: {
            SourceActionName: 'CodeStarConnectionsSourceAction',
            Push: [{
              Tags: {
                Excludes: ['exclude1', 'exclude2'],
                Includes: ['include1', 'include2'],
              },
            }],
          },
          ProviderType: 'CodeStarSourceConnection',
        },
        {
          GitConfiguration: {
            SourceActionName: 'CodeStarConnectionsSourceAction2',
            Push: [{
              Tags: {
                Excludes: ['exclude1', 'exclude2'],
                Includes: ['include1', 'include2'],
              },
            }],
          },
          ProviderType: 'CodeStarSourceConnection',
        },
      ],
    });
  });

  test('can specify triggers by addTrigger method', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
    });
    pipeline.addTrigger({
      providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
      gitConfiguration: {
        sourceAction,
        pushFilter: [{
          tagsExcludes: ['exclude1', 'exclude2'],
          tagsIncludes: ['include1', 'include2'],
        }],
      },
    });
    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Tags: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty tagsExcludes in GitPushFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: [],
            tagsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Tags: {
              Excludes: Match.absent(),
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty tagsIncludes in GitPushFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: ['excluded1', 'excluded2'],
            tagsIncludes: [],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: [{
            Tags: {
              Excludes: ['excluded1', 'excluded2'],
              Includes: Match.absent(),
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty branchesExcludes in GitPullRequestFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: [],
            branchesIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: Match.absent(),
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty branchesIncludes in GitPullRequestFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['excluded1', 'excluded2'],
            branchesIncludes: [],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['excluded1', 'excluded2'],
              Includes: Match.absent(),
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty filePathsExcludes in GitPullRequestFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            filePathsExcludes: [],
            filePathsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            FilePaths: {
              Excludes: Match.absent(),
              Includes: ['include1', 'include2'],
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty filePathsIncludes in GitPullRequestFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            filePathsExcludes: ['exclude1', 'exclude2'],
            filePathsIncludes: [],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            FilePaths: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: Match.absent(),
            },
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('undefined events in GitPullRequestFilter for trigger is set to all events', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            Events: ['OPEN', 'UPDATED', 'CLOSED'],
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty events in GitPullRequestFilter for trigger is set to all events', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            events: [],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            Events: ['OPEN', 'UPDATED', 'CLOSED'],
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('throw if length of tagsExcludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              tagsExcludes: ['exclude1', 'exclude2', 'exclude3', 'exclude4', 'exclude5', 'exclude6', 'exclude7', 'exclude8', 'exclude9'],
              tagsIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of tagsExcludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of tagsIncludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              tagsExcludes: ['exclude1', 'exclude2'],
              tagsIncludes: ['include1', 'include2', 'include3', 'include4', 'include5', 'include6', 'include7', 'include8', 'include9'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of tagsIncludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of branchesExcludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              branchesExcludes: ['exclude1', 'exclude2', 'exclude3', 'exclude4', 'exclude5', 'exclude6', 'exclude7', 'exclude8', 'exclude9'],
              branchesIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of branchesExcludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of branchesIncludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2', 'include3', 'include4', 'include5', 'include6', 'include7', 'include8', 'include9'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of branchesIncludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of filePathsExcludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2'],
              filePathsExcludes: ['exclude1', 'exclude2', 'exclude3', 'exclude4', 'exclude5', 'exclude6', 'exclude7', 'exclude8', 'exclude9'],
              filePathsIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of filePathsExcludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of filePathsIncludes in GitPushFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2'],
              filePathsExcludes: ['exclude1', 'exclude2'],
              filePathsIncludes: ['include1', 'include2', 'include3', 'include4', 'include5', 'include6', 'include7', 'include8', 'include9'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of filePathsIncludes in GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of branchesExcludes in GitPullRequestFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [{
              branchesExcludes: ['exclude1', 'exclude2', 'exclude3', 'exclude4', 'exclude5', 'exclude6', 'exclude7', 'exclude8', 'exclude9'],
              branchesIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of branchesExcludes in GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of branchesIncludes in GitPullRequestFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2', 'include3', 'include4', 'include5', 'include6', 'include7', 'include8', 'include9'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of branchesIncludes in GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of filePathsExcludes in GitPullRequestFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2'],
              filePathsExcludes: ['exclude1', 'exclude2', 'exclude3', 'exclude4', 'exclude5', 'exclude6', 'exclude7', 'exclude8', 'exclude9'],
              filePathsIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of filePathsExcludes in GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if length of filePathsIncludes in GitPullRequestFilter is greater than 8', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [{
              branchesExcludes: ['exclude1', 'exclude2'],
              branchesIncludes: ['include1', 'include2'],
              filePathsExcludes: ['exclude1', 'exclude2'],
              filePathsIncludes: ['include1', 'include2', 'include3', 'include4', 'include5', 'include6', 'include7', 'include8', 'include9'],
            }],
          },
        }],
      });
    }).toThrow(/maximum length of filePathsIncludes in GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction' is 8, got 9/);
  });

  test('throw if branches is not specified in GitPullRequestFilter', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [{}],
          },
        }],
      });
    }).toThrow(/must specify branches in GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction'/);
  });

  test('can eliminate duplicates events in GitPullRequestFilter', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pullRequestFilter: [{
            branchesExcludes: ['exclude1', 'exclude2'],
            branchesIncludes: ['include1', 'include2'],
            events: [
              codepipeline.GitPullRequestEvent.OPEN,
              codepipeline.GitPullRequestEvent.OPEN,
              codepipeline.GitPullRequestEvent.CLOSED,
            ],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: [{
            Branches: {
              Excludes: ['exclude1', 'exclude2'],
              Includes: ['include1', 'include2'],
            },
            Events: ['OPEN', 'CLOSED'],
          }],
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty GitPushFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [],
          pullRequestFilter: [
            {
              branchesExcludes: ['exclude1'],
            },
          ],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          Push: Match.absent(),
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('empty GitPullRequestFilter for trigger is set to undefined', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [
            {
              tagsExcludes: ['exclude1'],
            },
          ],
          pullRequestFilter: [],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
      PipelineType: 'V2',
      Triggers: [{
        GitConfiguration: {
          SourceActionName: 'CodeStarConnectionsSourceAction',
          PullRequest: Match.absent(),
        },
        ProviderType: 'CodeStarSourceConnection',
      }],
    });
  });

  test('throw if length of GitPushFilter is greater than 3', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [
              {
                tagsExcludes: ['exclude1'],
                tagsIncludes: ['include1'],
              },
              {
                tagsExcludes: ['exclude2'],
                tagsIncludes: ['include2'],
              },
              {
                tagsExcludes: ['exclude3'],
                tagsIncludes: ['include3'],
              },
              {
                tagsExcludes: ['exclude4'],
                tagsIncludes: ['include4'],
              },
            ],
          },
        }],
      });
    }).toThrow(/length of GitPushFilter for sourceAction with name 'CodeStarConnectionsSourceAction' must be less than or equal to 3, got 4/);
  });

  test('throw if length of GitPullRequestFilter is greater than 3', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pullRequestFilter: [
              {
                branchesExcludes: ['exclude1'],
                branchesIncludes: ['include1'],
              },
              {
                branchesExcludes: ['exclude2'],
                branchesIncludes: ['include2'],
              },
              {
                branchesExcludes: ['exclude3'],
                branchesIncludes: ['include3'],
              },
              {
                branchesExcludes: ['exclude4'],
                branchesIncludes: ['include4'],
              },
            ],
          },
        }],
      });
    }).toThrow(/length of GitPullRequestFilter for sourceAction with name 'CodeStarConnectionsSourceAction' must be less than or equal to 3, got 4/);
  });

  test('throw if neither GitPushFilter nor GitPullRequestFilter are specified', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
          },
        }],
      });
    }).toThrow(/must specify either GitPushFilter or GitPullRequestFilter for the trigger with sourceAction with name 'CodeStarConnectionsSourceAction'/);
  });

  test('throw if both GitPushFilter and GitPullRequestFilter are empty arrays', () => {
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction,
            pushFilter: [],
            pullRequestFilter: [],
          },
        }],
      });
    }).toThrow(/must specify either GitPushFilter or GitPullRequestFilter for the trigger with sourceAction with name 'CodeStarConnectionsSourceAction'/);
  });

  test('throw if provider of sourceAction is not \'CodeStarSourceConnection\'', () => {
    const fakeAction = new FakeSourceAction({
      actionName: 'FakeSource',
      output: sourceArtifact,
    });
    expect(() => {
      new codepipeline.Pipeline(stack, 'Pipeline', {
        pipelineType: codepipeline.PipelineType.V2,
        triggers: [{
          providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
          gitConfiguration: {
            sourceAction: fakeAction,
            pushFilter: [{
              tagsExcludes: ['exclude1', 'exclude2'],
              tagsIncludes: ['include1', 'include2'],
            }],
          },
        }],
      });
    }).toThrow(/provider for actionProperties in sourceAction with name 'FakeSource' must be 'CodeStarSourceConnection', got 'Fake'/);
  });

  test('throw if source action with duplicate action name added to the Pipeline', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V2,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: ['exclude1', 'exclude2'],
            tagsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });
    expect(() => {
      pipeline.addTrigger({
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: ['exclude1', 'exclude2'],
            tagsIncludes: ['include1', 'include2'],
          }],
        },
      });
    }).toThrow(/Trigger with duplicate source action 'CodeStarConnectionsSourceAction' added to the Pipeline/);
  });

  test('throw if triggers are specified when pipelineType is not set to V2', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V1,
      triggers: [{
        providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
        gitConfiguration: {
          sourceAction,
          pushFilter: [{
            tagsExcludes: ['exclude1', 'exclude2'],
            tagsIncludes: ['include1', 'include2'],
          }],
        },
      }],
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    const errors = validate(stack);

    expect(errors.length).toEqual(1);
    const error = errors[0];
    expect(error).toMatch(/Triggers can only be used with V2 pipelines, `PipelineType.V2` must be specified for `pipelineType`/);
  });

  test('throw if triggers are specified when pipelineType is not set to V2 and addTrigger method is used', () => {
    const pipeline = new codepipeline.Pipeline(stack, 'Pipeline', {
      pipelineType: codepipeline.PipelineType.V1,
    });
    pipeline.addTrigger({
      providerType: codepipeline.ProviderType.CODE_STAR_SOURCE_CONNECTION,
      gitConfiguration: {
        sourceAction,
        pushFilter: [{
          tagsExcludes: ['exclude1', 'exclude2'],
          tagsIncludes: ['include1', 'include2'],
        }],
      },
    });

    testPipelineSetup(pipeline, [sourceAction], [buildAction]);

    const errors = validate(stack);

    expect(errors.length).toEqual(1);
    const error = errors[0];
    expect(error).toMatch(/Triggers can only be used with V2 pipelines, `PipelineType.V2` must be specified for `pipelineType`/);
  });
});

function validate(construct: IConstruct): string[] {
  try {
    (construct.node.root as cdk.App).synth();
    return [];
  } catch (err: any) {
    if (!('message' in err) || !err.message.startsWith('Validation failed')) {
      throw err;
    }
    return err.message.split('\n').slice(1);
  }
}

// Adding 2 stages with actions so pipeline validation will pass
function testPipelineSetup(pipeline: codepipeline.Pipeline, sourceActions?: codepipeline.IAction[], buildActions?: codepipeline.IAction[]) {
  pipeline.addStage({
    stageName: 'Source',
    actions: sourceActions,
  });

  pipeline.addStage({
    stageName: 'Build',
    actions: buildActions,
  });
}
