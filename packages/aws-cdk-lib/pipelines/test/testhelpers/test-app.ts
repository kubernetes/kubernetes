import * as fs from 'fs';
import * as path from 'path';
import type { Construct } from 'constructs';
import * as ecr_assets from '../../../aws-ecr-assets';
import * as s3 from '../../../aws-s3';
import * as s3_assets from '../../../aws-s3-assets';
import type { AppProps, Environment, StageProps, StackProps } from '../../../core';
import { App, CfnOutput, Stage, Stack, ValidationError } from '../../../core';
import { lit } from '../../../core/lib/private/literal-string';
import { assemblyBuilderOf } from '../../lib/private/construct-internals';

export const PIPELINE_ENV: Environment = {
  account: '123456789012',
  region: 'us-pipeline',
};

export class TestApp extends App {
  constructor(props?: Partial<AppProps>) {
    super({
      context: {
        '@aws-cdk/core:newStyleStackSynthesis': '1',
      },
      stackTraces: false,
      autoSynth: false,
      treeMetadata: false,
      ...props,
    });
  }

  public stackArtifact(stackName: string | Stack) {
    if (typeof stackName !== 'string') {
      stackName = stackName.stackName;
    }

    this.synth();
    const supportStack = this.node.findAll().filter(Stack.isStack).find(s => s.stackName === stackName);
    expect(supportStack).not.toBeUndefined();
    return supportStack;
  }

  public cleanup() {
    rimraf(assemblyBuilderOf(this).outdir);
  }
}

export class AppWithExposedStacks extends Stage {
  public readonly stacks: Stack[];
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    this.stacks = new Array<Stack>();
    this.stacks.push(new BucketStack(this, 'Stack1'));
    this.stacks.push(new BucketStack(this, 'Stack2'));
    this.stacks.push(new BucketStack(this, 'Stack3'));
  }
}

export class OneStackApp extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);

    new BucketStack(this, 'Stack');
  }
}

export interface AppWithOutputProps extends StageProps {
  readonly stackId?: string;
}

export class AppWithOutput extends Stage {
  public readonly theOutput: CfnOutput;

  constructor(scope: Construct, id: string, props: AppWithOutputProps = {}) {
    super(scope, id, props);

    const stack = new BucketStack(this, props.stackId ?? 'Stack');
    this.theOutput = new CfnOutput(stack, 'MyOutput', { value: stack.bucket.bucketRef.bucketName });
  }
}

export interface TwoStackAppProps extends StageProps {
  /**
   * Create a dependency between the two stacks
   *
   * @default true
   */
  readonly withDependency?: boolean;
}

export class TwoStackApp extends Stage {
  public readonly stack1: Stack;
  public readonly stack2: Stack;

  constructor(scope: Construct, id: string, props?: TwoStackAppProps) {
    super(scope, id, props);

    this.stack2 = new BucketStack(this, 'Stack2');
    this.stack1 = new BucketStack(this, 'Stack1');

    if (props?.withDependency ?? true) {
      this.stack2.addStackDependency(this.stack1);
    }
  }
}

/**
 * Three stacks where the last one depends on the earlier 2
 */
export class ThreeStackApp extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);

    const stack1 = new BucketStack(this, 'Stack1');
    const stack2 = new BucketStack(this, 'Stack2');
    const stack3 = new BucketStack(this, 'Stack3');

    stack3.addStackDependency(stack1);
    stack3.addStackDependency(stack2);
  }
}

/**
 * A test stack
 *
 * It contains a single Bucket. Such robust. Much uptime.
 */
export class BucketStack extends Stack {
  public readonly bucket: s3.IBucketRef;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    this.bucket = new s3.Bucket(this, 'Bucket');
  }
}

/**
 * rm -rf reimplementation, don't want to depend on an NPM package for this
 */
export function rimraf(fsPath: string) {
  try {
    const isDir = fs.lstatSync(fsPath).isDirectory();

    if (isDir) {
      for (const file of fs.readdirSync(fsPath)) {
        rimraf(path.join(fsPath, file));
      }
      fs.rmdirSync(fsPath);
    } else {
      fs.unlinkSync(fsPath);
    }
  } catch (e: any) {
    // We will survive ENOENT
    if (e.code !== 'ENOENT') { throw e; }
  }
}

export function stackTemplate(stack: Stack) {
  const stage = Stage.of(stack);
  if (!stage) { throw new Error('stack not in a Stage'); }
  return stage.synth().getStackArtifact(stack.artifactId);
}

export class StageWithStackOutput extends Stage {
  public readonly output: CfnOutput;

  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    const stack = new BucketStack(this, 'Stack');

    this.output = new CfnOutput(stack, 'BucketName', {
      value: stack.bucket.bucketRef.bucketName,
    });
  }
}

export interface FileAssetAppProps extends StageProps {
  readonly displayName?: string;
}

export class FileAssetApp extends Stage {
  constructor(scope: Construct, id: string, props?: FileAssetAppProps) {
    super(scope, id, props);
    const stack = new Stack(this, 'Stack');
    new s3_assets.Asset(stack, 'Asset', {
      path: path.join(__dirname, 'assets', 'test-file-asset.txt'),
      displayName: props?.displayName,
    });
  }
}

export interface MultipleFileAssetsProps extends StageProps {
  readonly n: number;

  /**
   * Up to 3 display names for equally many assets
   */
  readonly displayNames?: string[];
}

/**
 * Supports up to 3 file assets
 */
export class MultipleFileAssetsApp extends Stage {
  constructor(scope: Construct, id: string, props: MultipleFileAssetsProps) {
    super(scope, id, props);
    const stack = new Stack(this, 'Stack');

    const fileNames = ['test-file-asset.txt', 'test-file-asset-two.txt', 'test-file-asset-three.txt'];
    if (props.displayNames && props.displayNames.length !== props.n) {
      throw new Error('Incorrect displayNames length');
    }

    for (let i = 0; i < props.n; i++) {
      const displayName = props.displayNames ? props.displayNames[i] : undefined;
      const fn = fileNames[i];
      if (!fn) {
        throw new ValidationError(lit`DisplayNamesFileNames`, `Got more displayNames than we have fileNames: ${i + 1}`, this);
      }

      new s3_assets.Asset(stack, `Asset${i + 1}`, {
        path: path.join(__dirname, 'assets', fn),
        displayName,
      });
    }
  }
}

export class DockerAssetApp extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    const stack = new Stack(this, 'Stack');
    new ecr_assets.DockerImageAsset(stack, 'Asset', {
      directory: path.join(__dirname, 'assets', 'test-docker-asset'),
    });
  }
}

export interface MegaAssetsAppProps extends StageProps {
  readonly numAssets: number;
}

// Creates a mix of file and image assets, up to a specified count
export class MegaAssetsApp extends Stage {
  constructor(scope: Construct, id: string, props: MegaAssetsAppProps) {
    super(scope, id, props);
    const stack = new Stack(this, 'Stack');

    let assetCount = 0;
    for (; assetCount < props.numAssets / 2; assetCount++) {
      new s3_assets.Asset(stack, `Asset${assetCount}`, {
        path: path.join(__dirname, 'assets', 'test-file-asset.txt'),
        assetHash: `FileAsset${assetCount}`,
      });
    }
    for (; assetCount < props.numAssets; assetCount++) {
      new ecr_assets.DockerImageAsset(stack, `Asset${assetCount}`, {
        directory: path.join(__dirname, 'assets', 'test-docker-asset'),
        extraHash: `FileAsset${assetCount}`,
      });
    }
  }
}

export class PlainStackApp extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    new BucketStack(this, 'Stack');
  }
}

export class MultiStackApp extends Stage {
  constructor(scope: Construct, id: string, props?: StageProps) {
    super(scope, id, props);
    new BucketStack(this, 'Stack1');
    new BucketStack(this, 'Stack2');
  }
}
